#!/usr/bin/env python3
"""Check the preregistered early response and coverage stop gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    timing = json.loads(args.timing.read_text())
    if timing["num_sampled_videos"] != gate["videos"] or timing["num_gold_response_windows"] != gate["gold_windows"]:
        raise ValueError("canary cohort or gold count differs")
    passed = (timing["total_physical_fires"] >= gate["minimum_physical_fires"] and
              timing["covered_gold_windows"] >= gate["minimum_covered_gold_windows"])
    result = {"passed": passed, "fires": timing["total_physical_fires"],
              "covered_gold_windows": timing["covered_gold_windows"],
              "minimum_physical_fires": gate["minimum_physical_fires"],
              "minimum_covered_gold_windows": gate["minimum_covered_gold_windows"],
              "videos": gate["videos"], "checkpoint_step": gate["checkpoint_step"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != result:
        raise ValueError("existing gate result differs")
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
