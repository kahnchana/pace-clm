#!/usr/bin/env python3
"""Validate rollout shards and prepare the unchanged Markii judge/scorer inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.markii_paucf_200 import _score_records, build_judge


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"refusing to change existing scoring artifact: {path}")
    else:
        path.write_text(content)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--skip-judge", action="store_true", help="Prepare timing only for training-video rollouts")
    a = p.parse_args()
    wanted = read_jsonl(a.manifest)
    gold = json.loads(a.gold.read_text())
    if len(wanted) != len(gold["records"]):
        raise RuntimeError("manifest/gold count mismatch")
    records = []
    for rank in range(a.world_size):
        records += read_jsonl(a.rollout_dir / f"predictions.rank{rank:02d}.jsonl")
    by_id = {r["question_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {r["question_id"] for r in wanted}:
        raise RuntimeError("missing, duplicate, or unexpected rollout question")
    ordered = [by_id[r["question_id"]] for r in wanted]
    for source, row in zip(wanted, ordered, strict=True):
        if source["video_id"] != row["video_id"]:
            raise RuntimeError("rollout video mismatch")
        last = float("-inf")
        for fire in row["model_response_list"]:
            if not str(fire.get("content", "")).strip():
                raise RuntimeError("empty Response text")
            time = float(fire["time"])
            if time < last or time not in source["inference_timestamps"]:
                raise RuntimeError("noncausal or unsorted Response time")
            last = time
    official = a.rollout_dir / "official"
    write_frozen(official / "generation/predictions.jsonl",
                 "".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered))
    write_frozen(official / "gold/paucf_4s_gold.json", json.dumps(gold, indent=2, sort_keys=True) + "\n")
    timing, per_video = _score_records(ordered, gold["records"], None)
    write_frozen(official / "results/timing_prejudge.json",
                 json.dumps({**timing, "per_video": per_video}, indent=2, sort_keys=True) + "\n")
    if not a.skip_judge:
        requests = official / "judge/pauc_batch_input.jsonl"
        provenance = official / "judge/judge_provenance.json"
        if requests.exists() or provenance.exists():
            if not requests.exists() or not provenance.exists():
                raise RuntimeError("incomplete existing judge preparation")
            if json.loads(provenance.read_text())["batch_input_sha256"] != sha(requests):
                raise RuntimeError("existing judge request hash differs")
        else:
            build_judge(SimpleNamespace(output_root=official))
    print(json.dumps({"videos": len(ordered), "fires": timing["total_physical_fires"],
                      "judge_requests": (len(read_jsonl(official / "judge/pauc_batch_input.jsonl"))
                                         if not a.skip_judge else 0)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
