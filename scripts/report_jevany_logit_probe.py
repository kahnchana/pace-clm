#!/usr/bin/env python3
"""Compare response margins where two rollouts still share action history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def read_rollout(directory: Path):
    result = {}
    for path in sorted(directory.glob("predictions.rank*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = row["question_id"]
            if question in result:
                raise ValueError(f"duplicate question: {question}")
            trace = row.get("jevany_action_trace")
            if not trace:
                raise ValueError(f"missing action trace: {question}")
            if sum(entry["action"] == "Response" for entry in trace) != len(row["model_response_list"]):
                raise ValueError(f"response trace is misaligned: {question}")
            result[question] = row
    return result


def response_margin(entry):
    silence, hold, response = entry["logits"]
    return response - max(silence, hold)


def compare(left, right):
    if set(left) != set(right):
        raise ValueError("probe cohorts differ")
    first_decision = []
    divergence = []
    shared_history_prefix_decisions = 0
    for question in sorted(left):
        a, b = left[question], right[question]
        first_decision.append({"question_id": question,
                               "left_margin": response_margin(a["jevany_action_trace"][0]),
                               "right_margin": response_margin(b["jevany_action_trace"][0])})
        if len(a["jevany_action_trace"]) != len(b["jevany_action_trace"]):
            raise ValueError(f"decision counts differ: {question}")
        response_index = 0
        for index, (entry_a, entry_b) in enumerate(zip(a["jevany_action_trace"],
                                                       b["jevany_action_trace"], strict=True)):
            shared_history_prefix_decisions += 1
            action_differs = entry_a["action"] != entry_b["action"]
            answer_differs = False
            if not action_differs and entry_a["action"] == "Response":
                answer_differs = (a["model_response_list"][response_index] !=
                                  b["model_response_list"][response_index])
                response_index += 1
            if action_differs or answer_differs:
                divergence.append({"question_id": question, "decision_index": index,
                                   "reason": "action" if action_differs else "answer",
                                   "left_action": entry_a["action"],
                                   "right_action": entry_b["action"],
                                   "left_margin": response_margin(entry_a),
                                   "right_margin": response_margin(entry_b)})
                break
    return {"videos": len(left), "first_divergence_videos": len(divergence),
            "shared_prefix_decisions_including_divergence": shared_history_prefix_decisions,
            "median_first_decision_response_margin_left": median(x["left_margin"] for x in first_decision),
            "median_first_decision_response_margin_right": median(x["right_margin"] for x in first_decision),
            "median_first_divergence_response_margin_left":
                median(x["left_margin"] for x in divergence) if divergence else None,
            "median_first_divergence_response_margin_right":
                median(x["right_margin"] for x in divergence) if divergence else None,
            "first_divergences": divergence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pre, final, replay = (read_rollout(path) for path in (args.pre, args.final, args.replay))
    result = {"pre_vs_final": compare(pre, final), "pre_vs_replay": compare(pre, replay)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {key: value for key, value in summary.items()
                             if key != "first_divergences"}
                      for name, summary in result.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
