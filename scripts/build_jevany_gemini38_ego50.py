#!/usr/bin/env python3
"""Freeze a matched diagnostic EGO cohort and reuse existing GPT-4.1 judgments."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path[:0] = [str(SG), str(SG / "src")]
from zs_infer.pauc_f import score_pauc_f

BASE = ROOT / "outputs/jevany-j1s/evaluation/benchmarks"
QWEN = BASE / "ego"
GEMINI = BASE / "ego-gemini38/full"
OUT = BASE / "ego-gemini38/prompt50"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> None:
    qwen_rows = read_jsonl(QWEN / "predictions.jsonl")
    gemini_rows = read_jsonl(GEMINI / "predictions.jsonl")
    qwen = {row["question_id"]: row for row in qwen_rows}
    gemini = {row["question_id"]: row for row in gemini_rows}
    if len(qwen) != 326 or set(qwen) != set(gemini):
        raise ValueError("complete matched EGO baselines are required")
    delta = {
        question_id: len(gemini[question_id]["model_response_list"])
        - len(qwen[question_id]["model_response_list"])
        for question_id in qwen
    }
    more = [question_id for question_id in qwen if delta[question_id] > 0]
    equal = [question_id for question_id in qwen if delta[question_id] == 0]
    fewer = [question_id for question_id in qwen if delta[question_id] < 0]
    if (len(more), len(equal), len(fewer)) != (119, 99, 108):
        raise ValueError("baseline fire-difference strata changed")
    rng = random.Random(42)
    selected = set(rng.sample(list(qwen), 50))
    ids = [row["question_id"] for row in gemini_rows if row["question_id"] in selected]
    if len(ids) != 50:
        raise ValueError(f"expected 50 selected IDs, got {len(ids)}")
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "selected_ids.json"
    body = json.dumps(ids, indent=2) + "\n"
    if selection_path.is_file() and selection_path.read_text() != body:
        raise ValueError("frozen selection differs")
    selection_path.write_text(body)
    audit = {
        "selection_seed": 42,
        "purpose": "fixed-seed random diagnostic EGO sample; no score-based selection",
        "population_strata": {"gemini_more": len(more), "equal": len(equal), "gemini_fewer": len(fewer)},
        "selected_strata": {
            "gemini_more": sum(delta[question_id] > 0 for question_id in ids),
            "equal": sum(delta[question_id] == 0 for question_id in ids),
            "gemini_fewer": sum(delta[question_id] < 0 for question_id in ids),
        },
        "num_videos": len(ids),
    }
    (OUT / "selection_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    for name, source in (("qwen", QWEN), ("gemini_baseline", GEMINI)):
        destination = OUT / "baselines" / name
        destination.mkdir(parents=True, exist_ok=True)
        predictions = {row["question_id"]: row for row in read_jsonl(source / "predictions.jsonl")}
        scored = {row["question_id"]: row for row in json.loads(
            (source / "pauc_llm_scores-gpt-4.1.json").read_text()
        )}
        (destination / "predictions.jsonl").write_text(
            "".join(json.dumps(predictions[question_id], sort_keys=True) + "\n" for question_id in ids)
        )
        (destination / "pauc_llm_scores-gpt-4.1.json").write_text(
            json.dumps([scored[question_id] for question_id in ids], indent=2) + "\n"
        )
        summary = score_pauc_f(
            destination / "predictions.jsonl",
            destination / "pauc_llm_scores-gpt-4.1.json",
            omega=0.5,
            judge_model="gpt-4.1",
        )
        (destination / "pauc_f_summary-gpt-4.1-v1.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(name, summary)
    print("selection", selection_path)


if __name__ == "__main__":
    main()
