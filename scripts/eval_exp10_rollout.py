#!/usr/bin/env python3
"""Resumable, full-text PACE-CLM rollout on a frozen Markii cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(SG / "src"), str(SG)]

from generate_selected_responses import forced_response
from stream_gemma.config import SYSTEM_PROMPT
from stream_gemma.episode import STATE_TAGS
from stream_gemma.inference.qwen_stream import QwenRuntime
from train_and_eval_heads import build_model, logits_for
from zs_infer.adapters import web

from pace_clm.backbone import FrozenFeatureExtractor
from pace_clm.experiment import decode_indices, direct_inputs
from pace_clm.labels import Action
from pace_clm.stream_gemma_adapter import validate_promoted_contract
from pace_clm.temporal import CausalTemporalAdapter, TemporalHead
from pace_clm.utility import UtilityHead


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--token-baseline", action="store_true")
    parser.add_argument("--capture-features-manifest", type=Path,
                        help="Run5 positions to cache under generated response history on training videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=0, help="Training-only dry run")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.token_baseline == bool(args.checkpoint):
        parser.error("specify exactly one of --checkpoint or --token-baseline")
    if args.capture_features_manifest and args.token_baseline:
        parser.error("feature replay capture requires a head checkpoint")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    rows = read_jsonl(args.manifest)
    capture_by_video: dict[str, set[int]] = {}
    capture_hash = sha(args.capture_features_manifest) if args.capture_features_manifest else None
    if args.capture_features_manifest:
        for source in read_jsonl(args.capture_features_manifest):
            capture_by_video.setdefault(source["video_id"], set()).add(int(source["frame_index"]))
    if len({r["video_id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate manifest video")
    if args.max_videos:
        rows = rows[:args.max_videos]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = validate_promoted_contract(SG)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    method = checkpoint["run"]["method"] if checkpoint else "pace_token"
    if method not in ("mlp", "clm_short_256", "pace_token"):
        raise RuntimeError(f"unsupported head checkpoint: {method}")
    temporal = bool(checkpoint and checkpoint["run"].get("temporal_hidden_width"))
    utility = bool(checkpoint and checkpoint["run"].get("utility_hidden_width"))
    if temporal and utility:
        raise RuntimeError("combined temporal and utility heads are outside exp10")
    protocol = {
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": sha(args.checkpoint) if args.checkpoint else None,
        "capture_features_manifest": str(args.capture_features_manifest.resolve()) if args.capture_features_manifest else None,
        "capture_features_sha256": capture_hash,
        "model": contract, "method": method, "temporal": temporal, "utility": utility,
        "max_new_tokens": args.max_new_tokens,
        "max_videos": args.max_videos, "question_ids": [r["question_id"] for r in rows],
        "target_fps": 1, "window_frames": 32, "prefix_mode": "direct",
        "memory_mode": "last-response", "memory_instruction_version": "v2",
        "memory_select": "aged-out",
    }
    protocol_path = args.output_dir / "manifest.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("output manifest differs from frozen protocol")
    if rank == 0:
        protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    shard = args.output_dir / f"predictions.rank{rank:02d}.jsonl"
    existing = read_jsonl(shard) if shard.exists() else []
    completed = {r["question_id"] for r in existing}
    if len(existing) != len(completed):
        raise RuntimeError("duplicate completed question")
    assigned = [r for i, r in enumerate(rows) if i % world == rank]
    if completed - {r["question_id"] for r in assigned}:
        raise RuntimeError("existing shard has questions outside assignment")
    feature_dir = args.output_dir / "replay_features"
    if args.capture_features_manifest:
        feature_dir.mkdir(exist_ok=True)

    def feature_path(video_id: str) -> Path:
        return feature_dir / (hashlib.sha256(video_id.encode()).hexdigest() + ".pt")

    assigned = [r for r in assigned if r["question_id"] not in completed or
                (args.capture_features_manifest and not feature_path(r["video_id"]).exists())]
    if not assigned:
        return
    runtime = QwenRuntime.load(
        checkpoint=contract["adapter"], model_name=contract["model_name"],
        model_revision=contract["model_revision"], frame_num_tokens=144,
        generation_prefix_mode="direct")
    counts: Counter[str] = Counter()
    current_video = ""
    current_frame_index = 0
    current_features: list[dict] = []
    head_seconds = 0.0
    extraction_seconds = 0.0
    if checkpoint is not None:
        extractor = FrozenFeatureExtractor(runtime.model, vision_token_ids=(runtime.processor.image_token_id,))
        base_head = build_model(method, extractor.hidden_size, 256 if method == "clm_short_256" else 0)
        head = (TemporalHead(CausalTemporalAdapter(extractor.hidden_size, 256), base_head)
                if temporal else UtilityHead(base_head, extractor.hidden_size, 256)
                if utility else base_head)
        head.load_state_dict(checkpoint["model"])
        head = head.to(runtime.device).eval()
        short = None
        if method == "clm_short_256":
            short = torch.load(Path("/fsx/sfr/data/kanchana/stream-gemma/pace-clm-10pct/features/action_variants.pt"), map_location="cpu",
                               weights_only=False)["short_labels"].float().to(runtime.device)
        margin = float(checkpoint["run"]["response_margin"])
        temporal_states: deque[torch.Tensor] = deque(maxlen=4)

        def choose_turn(messages, *, max_new_tokens, vision=None):
            nonlocal current_frame_index, head_seconds, extraction_seconds
            if vision is None:
                raise RuntimeError("vision cache required")
            extraction_started = time.monotonic()
            with runtime._cached_image_features(vision):
                pooled = extractor.extract(direct_inputs(runtime, messages, vision))
            extraction_seconds += time.monotonic() - extraction_started
            if args.capture_features_manifest and current_frame_index in capture_by_video.get(current_video, set()):
                current_features.append({"video_id": current_video, "frame_index": current_frame_index,
                                         "feature": pooled.squeeze(0).half().cpu()})
            current_frame_index += 1
            head_started = time.monotonic()
            with torch.inference_mode():
                if temporal:
                    temporal_states.append(pooled.float().squeeze(0))
                    states = [temporal_states[0]] * (4-len(temporal_states)) + list(temporal_states)
                    logits = head(torch.stack(states).unsqueeze(0), short)
                elif utility:
                    logits, predicted_utility = head(pooled.float(), short)
                    logits[:, int(Action.Response)] += float(checkpoint["run"]["utility_alpha"]) * predicted_utility
                else:
                    logits = logits_for(head, pooled.float(), short)
                logits[:, int(Action.Response)] -= margin
                action = Action(int(logits.argmax(-1).item()))
            head_seconds += time.monotonic() - head_started
            counts[action.name] += 1
            if action == Action.Response:
                content = forced_response(runtime, messages, vision, min(max_new_tokens, args.max_new_tokens))
                if not content or not content.strip():
                    raise RuntimeError("empty Response text")
                return f"{STATE_TAGS['Response']} {content}".strip()
            return STATE_TAGS["Silence" if action == Action.Silence else "Standby"]

        runtime.generate_turn = choose_turn
    started = time.monotonic()
    with shard.open("a", encoding="utf-8") as f:
        for index, row in enumerate(assigned, 1):
            video_started = time.monotonic()
            current_video = row["video_id"]
            current_frame_index = 0
            current_features = []
            head_seconds = 0.0
            extraction_seconds = 0.0
            if checkpoint is not None and temporal:
                temporal_states.clear()
            indices = [int(x) for x in row["inference_source_indices"]]
            times = [float(x) for x in row["inference_timestamps"]]
            if len(indices) != len(times) or times != sorted(set(times)):
                raise RuntimeError("invalid causal frame times")
            images = runtime._resize(decode_indices(row["resolved_video_path"], indices))
            prepared = {"images": images, "frame_times": times}
            record = web.generate(
                runtime, {"question_id": row["question_id"], "question": row["question"],
                          "path": row["resolved_video_path"]},
                mode="v24", source_fps=0, frame_cap=None, v24_system_prompt=SYSTEM_PROMPT,
                v24_memory_mode="last-response", v24_memory_instruction_version="v2",
                v24_memory_select="aged-out", prepared=prepared)
            record["video_id"] = row["video_id"]
            record["exp10_video_wall_seconds"] = time.monotonic() - video_started
            record["exp10_frame_count"] = len(indices)
            if checkpoint is not None:
                record["pace_clm_actions"] = dict(counts)
                record["exp10_head_seconds"] = head_seconds
                record["exp10_extraction_seconds"] = extraction_seconds
            if int(record.get("v24_parse", {}).get("malformed", 0)):
                raise RuntimeError("malformed action emission")
            if any(not str(r.get("content", "")).strip() for r in record["model_response_list"]):
                raise RuntimeError("malformed or empty Response")
            if args.capture_features_manifest:
                expected_keys = capture_by_video.get(current_video, set())
                observed_keys = {r["frame_index"] for r in current_features}
                if expected_keys != observed_keys:
                    raise RuntimeError(f"replay feature positions differ: {current_video}")
                path = feature_path(current_video)
                if not path.exists():
                    temp = path.with_suffix(".pt.tmp")
                    torch.save({"video_id": current_video, "rows": current_features,
                                "capture_manifest_sha256": capture_hash}, temp)
                    os.replace(temp, path)
            if row["question_id"] not in completed:
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
            print(f"[rank {rank}] {index}/{len(assigned)} {row['video_id']} "
                  f"fires={len(record['model_response_list'])} elapsed={time.monotonic()-started:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
