#!/usr/bin/env python3
"""Paired J1/J1-S diagnostics from frozen judged WEB/EGO artifacts."""

import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from zs_infer.pauc_f import assign_value_adding_fires, fire_times


ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
OFFICIAL = SG / "third_party/proactive-videoqa/pauc/judge_llm_scores_openai.py"
spec = importlib.util.spec_from_file_location("official_pauc", OFFICIAL)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def video_stats(prediction, score):
    fires = fire_times(prediction)
    useful = len(assign_value_adding_fires(fires, score["answer"]))
    auc = []
    repeats = 0
    for turn in score["answer"]:
        repeats += max(0, len(turn.get("preds", [])) - 1)
        judged = turn.get("judge_scores") or {}
        if not judged:
            auc.append(0.25)
        else:
            points = [[float(time) - turn["reply_timespan"][0], value]
                      for time, value in judged.items()]
            auc.append(official.area_under_line_ratio(
                points, turn["reply_timespan"][1] - turn["reply_timespan"][0],
                2, 0.5, 0.5,
            ))
    return {"fires": len(fires), "useful": useful, "wasted": len(fires) - useful,
            "repeat_fires_in_gold_windows": repeats, "auc_sum": sum(auc),
            "gold_turns": len(auc)}


def pooled_score(rows):
    counts = Counter()
    for row in rows:
        counts.update({key: row[key] for key in ("fires", "useful", "wasted", "auc_sum", "gold_turns")})
    precision = counts["useful"] / counts["fires"] if counts["fires"] else 0
    recall = counts["auc_sum"] / counts["gold_turns"] if counts["gold_turns"] else 0
    score = 200 * precision * recall / (precision + recall) if precision + recall else 0
    return score


def main():
    report = {}
    rng = random.Random(20261028)
    for split in ("web", "ego"):
        source = SG / f"datasets/proactive-videoqa-{split}/anno.json"
        duration = {str(row["question_id"]): float(row["duration"]) for row in json.loads(source.read_text())}
        cells = {}
        for name, path in (("J1", ROOT / "outputs/jevany/evaluation/benchmarks"),
                           ("J1-S", ROOT / "outputs/jevany-j1s/evaluation/benchmarks")):
            folder = path / split
            predictions = {str(r["question_id"]): r for r in read_jsonl(folder / "predictions.jsonl")}
            scores = {str(r["question_id"]): r for r in json.loads((folder / "pauc_llm_scores-gpt-4.1.json").read_text())}
            assert predictions.keys() == scores.keys() == duration.keys()
            cells[name] = {qid: video_stats(predictions[qid], scores[qid]) for qid in duration}
        ids = sorted(duration)
        delta = {metric: sum(cells["J1-S"][qid][metric] - cells["J1"][qid][metric] for qid in ids)
                 for metric in ("fires", "useful", "wasted", "repeat_fires_in_gold_windows")}
        paired = Counter()
        by_bucket = defaultdict(lambda: {"videos": 0, "J1": Counter(), "J1-S": Counter()})
        for qid in ids:
            old, new = cells["J1"][qid], cells["J1-S"][qid]
            paired["fewer_fires" if new["fires"] < old["fires"] else
                   "more_fires" if new["fires"] > old["fires"] else "equal_fires"] += 1
            bucket = "<=60s" if duration[qid] <= 60 else "60-180s" if duration[qid] <= 180 else ">180s"
            by_bucket[bucket]["videos"] += 1
            for name, row in (("J1", old), ("J1-S", new)):
                by_bucket[bucket][name].update({k: row[k] for k in ("fires", "useful", "wasted", "repeat_fires_in_gold_windows")})
        point = pooled_score([cells["J1-S"][qid] for qid in ids]) - pooled_score([cells["J1"][qid] for qid in ids])
        bootstrap = []
        for _ in range(5000):
            sample = rng.choices(ids, k=len(ids))
            bootstrap.append(pooled_score([cells["J1-S"][qid] for qid in sample]) -
                             pooled_score([cells["J1"][qid] for qid in sample]))
        bootstrap.sort()
        report[split] = {
            "paired_pauc_f_delta_j1s_minus_j1": point,
            "paired_video_bootstrap_95_ci": [bootstrap[125], bootstrap[4874]],
            "bootstrap_samples": 5000, "bootstrap_seed": 20261028,
            "aggregate_delta_j1s_minus_j1": delta,
            "paired_fire_counts": dict(paired),
            "duration_buckets": {
                bucket: {"videos": values["videos"],
                         **{name: dict(values[name]) for name in ("J1", "J1-S")}}
                for bucket, values in by_bucket.items()
            },
        }
    output = ROOT / "outputs/jevany-j1s/reports/paired_diagnostics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
