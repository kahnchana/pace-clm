#!/usr/bin/env python3
"""Merge generated responses and score with unchanged Markii PAUC-F code."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(STREAM_GEMMA))

from zs_infer.markii_paucf_200 import _score_records, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    records = []
    for rank in range(args.world_size):
        records.extend(read_jsonl(args.input_dir / f"responses.rank{rank:02d}.jsonl"))
    records.sort(key=lambda row: row["video_id"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    generation = args.output_root / "generation"
    generation.mkdir(exist_ok=True)
    predictions_path = generation / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    source_root = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    gold_dir = args.output_root / "gold"
    gold_dir.mkdir(exist_ok=True)
    for name in ("paucf_4s_gold.json", "paucf_4s_window_audit.json"):
        shutil.copy2(source_root / "gold" / name, gold_dir / name)
    gold = read_json(gold_dir / "paucf_4s_gold.json")["records"]
    timing, per_video = _score_records(records, gold, None)
    results = args.output_root / "results"
    results.mkdir(exist_ok=True)
    (results / "paucf_4s_timing_diagnostic.json").write_text(
        json.dumps({**timing, "per_video": per_video}, indent=2, sort_keys=True) + "\n"
    )
    (generation / "generation_summary.json").write_text(
        json.dumps(
            {
                "num_records": len(records),
                "num_streaming_decisions": sum(
                    row["telemetry"]["inference_frames"] for row in records
                ),
                "num_physical_fires": sum(
                    row["telemetry"]["number_of_model_fires"] for row in records
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(timing, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
