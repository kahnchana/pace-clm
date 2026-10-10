#!/usr/bin/env python3
"""Freeze two fresh disjoint Markii validation cohorts and timing gold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(SG))
from zs_infer.markii_paucf_200 import read_jsonl, sha256_file


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def write_new(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != data:
            raise FileExistsError(f"frozen artifact differs: {path}")
        return
    path.write_text(data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--videos-per-set", type=int, default=200)
    p.add_argument("--offset", type=int, default=0,
                   help="Skip this many hash-ordered eligible videos before making the two sets")
    args = p.parse_args()
    if args.offset < 0:
        raise ValueError("offset must be nonnegative")
    source = SG / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    population = read_jsonl(source / "manifests/valid_population.jsonl")
    old = {r["video_id"] for r in read_jsonl(source / "subset/subset_200_manifest.jsonl")}
    available = [r for r in population if r["video_id"] not in old]
    available.sort(key=lambda r: digest("pace-clm-mixture-val-20260928:" + r["video_id"]))
    available = available[args.offset:]
    if len(available) < args.videos_per_set * 2:
        raise RuntimeError("insufficient fresh validation videos")
    cohorts = {"selection": available[:args.videos_per_set],
               "final": available[args.videos_per_set:args.videos_per_set * 2]}
    for name, rows in cohorts.items():
        manifest = args.output_dir / f"{name}.jsonl"
        write_new(manifest, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        records = []
        for row in rows:
            runs = row["exact_caption_response_runs"]
            windows = []
            end_of_video = float(row["physical_duration"])
            for i, run in enumerate(runs):
                start = float(run["response_mark"])
                next_start = float(runs[i + 1]["response_mark"]) if i + 1 < len(runs) else math.inf
                end = min(start + 4.0, next_start, end_of_video)
                windows.append({"window_index": i, "response_mark": start, "window_start": start,
                                "window_end": end, "end_inclusive": False, "length_seconds": max(0.0, end-start),
                                "gold_text": run["gold_text"], "annotation_end_time": run["annotation_end_time"],
                                "decision_timestamps": [t for t in row["inference_timestamps"] if start <= float(t) < end]})
            records.append({"question_id": row["question_id"], "video_id": row["video_id"],
                            "question": row["question"], "physical_video_end": end_of_video, "windows": windows})
        gold = args.output_dir / f"{name}.gold.json"
        write_new(gold, json.dumps({"protocol_version": "markii-paucf-200-zss7-yesno-v1",
                                    "window_seconds": 4.0, "boundary_semantics": "[start,end)",
                                    "records": records}, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"set": name, "videos": len(rows), "frames": sum(r["inference_frame_count"] for r in rows),
                          "response_windows": sum(len(r["windows"]) for r in records),
                          "manifest_sha256": sha256_file(manifest), "gold_sha256": sha256_file(gold)}), flush=True)


if __name__ == "__main__":
    main()
