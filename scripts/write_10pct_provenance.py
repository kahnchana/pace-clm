#!/usr/bin/env python3
"""Hash the frozen cohort, code, features, and scaled experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source_files = [
        ROOT / "configs/qwen_mvp.yaml", ROOT / "configs/actions_v1.yaml",
        ROOT / "manifests/pace_train_stream_gemma_10pct.jsonl",
        ROOT / "manifests/pace_train_10pct_decisions.jsonl",
        ROOT / "manifests/pace_train_10pct_summary.json",
    ] + [ROOT / "scripts" / name for name in (
        "build_10pct_manifests.py", "extract_feature_shards.py", "merge_10pct_parallel.py",
        "run_10pct_tail_assigned.sh", "sbatch_extract_10pct.sh",
        "train_10pct_curve.py", "eval_10pct_frozen.py", "benchmark_serving_10pct.py",
        "finalize_10pct.py", "score_timing_paucf.py",
    )]
    artifact_files = [
        args.output_root / "features" / name for name in
        ("train.pt", "pace200.pt", "action_variants.pt", "train.summary.json")
    ] + [args.output_root / "results" / name for name in
         ("pilot_calibration.json", "curve.json", "pace200_results.json",
          "serving_benchmark.json", "decision_summary.json", "RESULTS.md")]
    missing = [str(path) for path in source_files + artifact_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    payload = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "pace_clm_commit": commit(ROOT),
        "stream_gemma_commit": commit(STREAM_GEMMA),
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "artifact_sha256": {str(path): sha256(path) for path in artifact_files},
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "adapter": str(STREAM_GEMMA / "outputs/lora-qwen-w1-long-neg"),
        "training_seeds": [42, 43, 44],
        "nested_sizes": [768, 3000, 12000, 24300],
        "feature_contract": "32-frame causal window; final valid language token; common gold prior-action history",
    }
    (args.output_root / "results" / "run_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print("wrote run_provenance.json", flush=True)


if __name__ == "__main__":
    main()
