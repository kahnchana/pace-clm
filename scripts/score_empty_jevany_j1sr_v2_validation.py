#!/usr/bin/env python3
"""Finalize J1-S-R validation with no semantic requests to submit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / "outputs/jevany-j1sr-v2/evaluation/validation/j1sr"
JUDGE = ROLLOUT / "official/judge"


def main() -> None:
    requests = JUDGE / "pauc_batch_input.jsonl"
    provenance = json.loads((JUDGE / "judge_provenance.json").read_text())
    if not requests.is_file() or requests.read_text().strip():
        raise ValueError("expected a prepared, empty semantic request batch")
    if provenance["num_unique_requests"] or provenance["request_mappings"]:
        raise ValueError("judge provenance contains requests")
    scores = {"judge_model": "gpt-4.1", "scores_by_request_key": {},
              "request_mappings": []}
    path = JUDGE / "pauc_llm_scores.json"
    body = json.dumps(scores, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != body:
        raise ValueError("existing judge score artifact differs")
    path.write_text(body)
    subprocess.run([sys.executable, str(ROOT / "scripts/score_exp10.py"),
                    "--rollout-dir", str(ROLLOUT)], check=True)


if __name__ == "__main__":
    main()
