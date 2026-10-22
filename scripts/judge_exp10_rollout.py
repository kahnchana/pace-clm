#!/usr/bin/env python3
"""Judge and score one complete exp10 rollout through the authorized portal."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(ROOT / "scripts"), str(SG)]
from judge_exp10_utility import load_gateway_key
from zs_infer.markii_paucf_200 import run_judge


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    root = a.rollout_dir / "official"
    if not (root / "judge/pauc_batch_input.jsonl").is_file():
        raise FileNotFoundError("build and validate judge requests before portal submission")
    load_gateway_key()
    run_judge(SimpleNamespace(output_root=root, concurrency=a.concurrency))
    subprocess.run([sys.executable, str(ROOT / "scripts/score_exp10.py"),
                    "--rollout-dir", str(a.rollout_dir)], check=True)


if __name__ == "__main__":
    main()
