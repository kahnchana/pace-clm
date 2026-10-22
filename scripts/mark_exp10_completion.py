#!/usr/bin/env python3
"""Write an immutable completion marker for one fully judged exp10 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pace-clm-exp10"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", choices=[f"E{i:02d}" for i in range(1, 11)], required=True)
    a = p.parse_args()
    checkpoint = OUT / a.run / "checkpoint.pt"
    selection = OUT / "selection" / a.run / "official/results/exp10_judged_summary.json"
    score = json.loads(selection.read_text())
    if not score.get("headline") or score["num_sampled_videos"] != 200 or score["judge_model"] != "gpt-4.1":
        raise RuntimeError("not a complete judged selection result")
    result = {"run": a.run, "trained_checkpoint_sha256": sha(checkpoint),
              "selection_judged_score_sha256": sha(selection),
              "selection_judged_pauc_f": score["pauc_f"], "status": "selection_complete"}
    path = OUT / a.run / "completion.json"
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError("completion marker changed")
    path.write_text(content)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
