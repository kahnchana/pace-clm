#!/usr/bin/env python3
"""Paired decision latency on cached PACE-200 frame vision features."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(STREAM_GEMMA / "src")]

from stream_gemma.inference.qwen_stream import FrameVisionCache

from pace_clm.experiment import (
    absolute_timeline, annotation_records, decode_indices, direct_inputs, gold_messages,
    read_jsonl, state_timeline,
)
from train_and_eval_heads import build_model, logits_for
from extract_feature_shards import load_runtime
from pace_clm.config import load_mvp_config


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return time.perf_counter() - start, result


def describe(values):
    return {"count": len(values), "mean_ms": 1000 * statistics.mean(values),
            "median_ms": 1000 * statistics.median(values),
            "p90_ms": 1000 * sorted(values)[int(0.9 * (len(values) - 1))]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=4)
    parser.add_argument("--decisions-per-video", type=int, default=8)
    args = parser.parse_args()
    runtime, extractor = load_runtime(load_mvp_config(ROOT / "configs/qwen_mvp.yaml"))
    val = torch.load(args.feature_dir / "pace200.pt", map_location="cpu", weights_only=False)
    actions = val["action_features"].float().to(runtime.device)
    variants = torch.load(args.feature_dir / "action_variants.pt", map_location="cpu", weights_only=False)
    short_actions = variants["short_labels"].float().to(runtime.device)
    models = {}
    for name, dim in (("linear", 0), ("mlp", 0), ("clm_128", 128),
                      ("clm_256", 256), ("clm_short_256", 256)):
        artifact = torch.load(args.result_dir / f"{name}.n24300.seed42.pt", map_location="cpu", weights_only=False)
        model = build_model(name, int(val["hidden_size"]), dim).to(runtime.device)
        model.load_state_dict(artifact["model"])
        model.eval()
        models[name] = model
    source = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    manifest = read_jsonl(source / "subset/subset_200_manifest.jsonl")[:args.videos]
    records = annotation_records(STREAM_GEMMA / "datasets/markii/val.jsonl")
    metadata = json.loads((STREAM_GEMMA / "datasets/markii/metadata.json").read_text())
    timings = {name: [] for name in ("pace_token", *models)}
    order = []
    with torch.no_grad():
        for source_row in manifest:
            timeline = state_timeline(records[int(source_row["raw_index"])], metadata)
            count = int(source_row["inference_frame_count"])
            emits, states = absolute_timeline(timeline, count)
            images = runtime._resize(decode_indices(
                source_row["resolved_video_path"],
                [int(value) for value in source_row["inference_source_indices"]],
            ))
            cache = FrameVisionCache(runtime, limit=max(32, count))
            cache.prefetch(images)
            indices = sorted(set(round(i * (count - 1) / max(1, args.decisions_per_video - 1))
                                 for i in range(args.decisions_per_video)))
            for index in indices:
                messages, start = gold_messages(
                    question=source_row["question"], images=images, emits=emits,
                    states=states, index=index,
                )
                vision = cache.window(images, start, index + 1)
                def feature_call():
                    inputs = direct_inputs(runtime, messages, vision)
                    with runtime._cached_image_features(vision):
                        return extractor.extract(inputs).float()
                # Paired calls share the same decoded and prefetched frames.
                token_seconds, _ = timed(lambda: runtime.generate_turn(
                    messages, max_new_tokens=4, vision=vision
                ))
                feature_seconds, feature = timed(feature_call)
                timings["pace_token"].append(token_seconds)
                for name, model in models.items():
                    action_vectors = short_actions if name == "clm_short_256" else actions
                    head_seconds, _ = timed(lambda model=model, action_vectors=action_vectors:
                                            logits_for(model, feature, action_vectors))
                    timings[name].append(feature_seconds + head_seconds)
                order.append({"video_id": source_row["video_id"], "frame_index": index,
                              "token_s": token_seconds, "feature_s": feature_seconds})
                print(f"{len(order)} decisions", flush=True)
    report = {"protocol": "same gold history and cached frame vision; 4-token PACE tag vs frozen final-token feature plus head",
              "methods": {name: describe(values) for name, values in timings.items()},
              "paired_samples": order,
              "head_only_parameters": {name: sum(p.numel() for p in model.parameters())
                                       for name, model in models.items()}}
    (args.result_dir / "serving_benchmark.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["methods"], indent=2), flush=True)


if __name__ == "__main__":
    main()
