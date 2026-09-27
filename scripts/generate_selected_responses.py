#!/usr/bin/env python3
"""Generate answer text at classifier-selected Response decisions."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from stream_gemma.inference.qwen_stream import FrameVisionCache, QwenRuntime

from pace_clm.config import load_mvp_config
from pace_clm.experiment import (
    absolute_timeline,
    annotation_records,
    decode_indices,
    direct_inputs,
    gold_messages,
    read_jsonl,
    state_timeline,
)
from pace_clm.labels import Action


def forced_response(runtime, messages: list[dict], vision, max_new_tokens: int) -> str:
    inputs = direct_inputs(runtime, messages, vision)
    prefix = runtime.tokenizer("</Response>", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(runtime.device)
    input_ids = torch.cat([inputs["input_ids"], prefix], dim=1)
    attention = torch.cat(
        [inputs["attention_mask"], torch.ones_like(prefix, device=runtime.device)], dim=1
    )
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(prefix, device=runtime.device)], dim=1
        )
    with runtime._cached_image_features(vision), torch.inference_mode():
        output = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    continuation = runtime.processor.batch_decode(
        output[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0].strip()
    return continuation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-fires",
        type=int,
        default=None,
        help="Deterministically sample at most this many Response decisions.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--selection-mode",
        choices=("random", "shortest"),
        default="random",
        help="How to choose a bounded smoke sample from predicted fires.",
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)

    prediction_artifact = torch.load(
        args.prediction_checkpoint, map_location="cpu", weights_only=False
    )
    predictions = prediction_artifact["predictions"].tolist()
    pred_by_video: dict[str, dict[int, int]] = {}
    for pred, row in zip(predictions, prediction_artifact["metadata"], strict=True):
        pred_by_video.setdefault(row["video_id"], {})[int(row["frame_index"])] = int(pred)
    val_path = (
        STREAM_GEMMA
        / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl"
    )
    val_rows = read_jsonl(val_path)
    frame_counts = {row["video_id"]: int(row["inference_frame_count"]) for row in val_rows}
    candidate_fires = sorted(
        (video_id, frame_index)
        for video_id, by_frame in pred_by_video.items()
        for frame_index, value in by_frame.items()
        if value == int(Action.Response)
    )
    selected_fires = candidate_fires
    if args.max_fires is not None and len(candidate_fires) > args.max_fires:
        if args.selection_mode == "random":
            selected_fires = sorted(
                random.Random(args.sample_seed).sample(candidate_fires, args.max_fires)
            )
        else:
            ordered = sorted(
                candidate_fires,
                key=lambda item: (frame_counts[item[0]], item[0], item[1]),
            )
            selected_fires = []
            selected_videos = set()
            for event in ordered:
                if event[0] not in selected_videos:
                    selected_fires.append(event)
                    selected_videos.add(event[0])
                if len(selected_fires) == args.max_fires:
                    break
    selected_fire_set = set(selected_fires)

    bins: list[list[dict]] = [[] for _ in range(world)]
    loads = [0] * world
    for row in sorted(
        val_rows, key=lambda item: (-int(item["inference_frame_count"]), item["video_id"])
    ):
        target = min(range(world), key=lambda item: (loads[item], item))
        bins[target].append(row)
        loads[target] += int(row["inference_frame_count"])
    assigned = sorted(bins[rank], key=lambda row: row["video_id"])

    config = load_mvp_config(ROOT / "configs/qwen_mvp.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.output_dir / "selection.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.prediction_checkpoint),
                    "candidate_fires": len(candidate_fires),
                    "selected_fires": len(selected_fires),
                    "sample_seed": args.sample_seed,
                    "selection_mode": args.selection_mode,
                    "max_new_tokens": args.max_new_tokens,
                    "events": [
                        {"video_id": video_id, "frame_index": frame_index}
                        for video_id, frame_index in selected_fires
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    runtime = QwenRuntime.load(
        checkpoint=str((ROOT / config["model"]["adapter"]).resolve()),
        model_name=config["model"]["name"],
        model_revision=config["model"]["revision"],
        frame_num_tokens=int(config["streaming"]["frame_num_tokens"]),
        generation_prefix_mode="direct",
    )
    records = annotation_records(STREAM_GEMMA / "datasets/markii/val.jsonl")
    metadata = json.loads((STREAM_GEMMA / "datasets/markii/metadata.json").read_text())
    output = []
    started = time.monotonic()
    for number, source in enumerate(assigned, 1):
        fires = [
            index
            for index, value in sorted(pred_by_video[source["video_id"]].items())
            if value == int(Action.Response) and (source["video_id"], index) in selected_fire_set
        ]
        if not fires:
            output.append(
                {
                    "video_id": source["video_id"],
                    "question_id": source["question_id"],
                    "question": source["question"],
                    "inference_timestamps": source["inference_timestamps"],
                    "inference_source_indices": source["inference_source_indices"],
                    "model_response_list": [],
                    "telemetry": {
                        "inference_frames": int(source["inference_frame_count"]),
                        "number_of_model_fires": 0,
                        "total_seconds": time.monotonic() - started,
                    },
                }
            )
            print(
                f"[rank {rank}] {number}/{len(assigned)} videos fires=0",
                flush=True,
            )
            continue
        timeline = state_timeline(records[int(source["raw_index"])], metadata)
        count = int(source["inference_frame_count"])
        emits, states = absolute_timeline(timeline, count)
        images = runtime._resize(
            decode_indices(
                source["resolved_video_path"],
                [int(value) for value in source["inference_source_indices"]],
            )
        )
        cache = FrameVisionCache(runtime, limit=max(32, count))
        cache.prefetch(images)
        responses = []
        for index in fires:
            messages, start = gold_messages(
                question=source["question"],
                images=images,
                emits=emits,
                states=states,
                index=index,
            )
            content = forced_response(
                runtime,
                messages,
                cache.window(images, start, index + 1),
                args.max_new_tokens,
            )
            responses.append({"time": float(index), "content": content})
        output.append(
            {
                "video_id": source["video_id"],
                "question_id": source["question_id"],
                "question": source["question"],
                "inference_timestamps": source["inference_timestamps"],
                "inference_source_indices": source["inference_source_indices"],
                "model_response_list": responses,
                "telemetry": {
                    "inference_frames": count,
                    "number_of_model_fires": len(responses),
                    "total_seconds": time.monotonic() - started,
                },
            }
        )
        print(
            f"[rank {rank}] {number}/{len(assigned)} videos fires={len(responses)}",
            flush=True,
        )
    path = args.output_dir / f"responses.rank{rank:02d}.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
    print(f"[rank {rank}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
