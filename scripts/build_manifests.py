#!/usr/bin/env python3
"""Freeze the balanced PACE training selection and the PACE-200 reference."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from pace_clm.experiment import build_balanced_train_manifest, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    val = (
        STREAM_GEMMA
        / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl"
    )
    val_rows = read_jsonl(val)
    train = ROOT / "manifests/pace_train_balanced_v1.jsonl"
    summary = build_balanced_train_manifest(
        annotation_path=STREAM_GEMMA / "datasets/markii/train.jsonl",
        metadata_path=STREAM_GEMMA / "datasets/markii/metadata.json",
        validation_video_ids={row["video_id"] for row in val_rows},
        output_path=train,
    )
    reference = {
        "version": "pace-manifests-v1",
        "train_manifest": str(train.relative_to(ROOT)),
        "train_sha256": sha256(train),
        "train": summary,
        "validation_manifest": str(val),
        "validation_sha256": sha256(val),
        "validation_videos": len(val_rows),
        "validation_decisions": sum(row["inference_frame_count"] for row in val_rows),
        "validation_gold_response_runs": sum(row["gold_response_run_count"] for row in val_rows),
        "overlap": sorted(
            {row["video_id"] for row in val_rows} & {row["video_id"] for row in read_jsonl(train)}
        ),
    }
    output = ROOT / "manifests/pace_manifests_v1.json"
    output.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n")
    print(json.dumps(reference, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
