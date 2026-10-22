#!/usr/bin/env python3
"""Run unchanged WEB/EGO GPT-4.1 official scorer for a frozen exp10 rollout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(ROOT / "scripts"), str(SG)]
from judge_exp10_utility import load_gateway_key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("benchmark", choices=("web", "ego"))
    p.add_argument("--pred-dir", type=Path, required=True)
    a = p.parse_args()
    load_gateway_key()
    gold = SG / f"datasets/proactive-videoqa-{a.benchmark}/anno.json"
    subprocess.run([sys.executable, "-m", "zs_infer.scoring", "web", "all",
                    "--pred-dir", str(a.pred_dir), "--gold", str(gold),
                    "--judge-model", "gpt-4.1", "--benchmark-name",
                    f"ProactiveVideoQA {a.benchmark.upper()}"], cwd=SG, check=True)
    subprocess.run([sys.executable, "-m", "zs_infer.pauc_f",
                    "--predictions", str(a.pred_dir / "predictions.jsonl"),
                    "--scores", str(a.pred_dir / "pauc_llm_scores-gpt-4.1.json"),
                    "--output", str(a.pred_dir / "pauc_f_summary-gpt-4.1-v1.json"),
                    "--omega", "0.5", "--judge-model", "gpt-4.1"], cwd=SG, check=True)


if __name__ == "__main__":
    main()
