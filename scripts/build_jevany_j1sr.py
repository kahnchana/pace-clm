#!/usr/bin/env python3
"""Freeze training-video mining inputs and audit J1-S-R causal replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/jevany-j1s/manifests/train.jsonl"
OUT = ROOT / "outputs/jevany-j1sr"
quota_labels = frozenset(("Silence", "Response"))


def token_overlap(left: str, right: str) -> float:
    first = set(re.findall(r"[a-z0-9]+", left.lower()))
    second = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(first & second) / len(first | second) if first and second else 0.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_new(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != body:
        raise FileExistsError(f"frozen artifact differs: {path}")
    path.write_text(body)


def select(args):
    by_video = defaultdict(list)
    for row in rows(SOURCE):
        if row["label"] in ("Silence", "Response"):
            by_video[row["video_id"]].append(row)
    excluded = set()
    for path in (ROOT / "outputs/pace-clm-mixtures/validation/selection.jsonl",
                 ROOT / "outputs/pace-clm-mixtures/validation/final.jsonl",
                 OUT / "validation/frozen.jsonl"):
        if path.exists():
            excluded.update(row["video_id"] for row in rows(path))
    eligible = []
    for video, group in by_video.items():
        if (video in excluded or len({r["question"] for r in group}) != 1 or
                len({r["video_path"] for r in group}) != 1 or
                not any(r["label"] == "Response" for r in group)):
            continue
        last = max(r["frame_index"] for r in group)
        if last < 32 or last >= args.max_frames:
            continue
        priority = hashlib.sha256(f"j1sr:{args.seed}:{video}".encode()).hexdigest()
        eligible.append((priority, video, group, last))
    eligible.sort()
    selected = eligible[:args.videos]
    if len(selected) != args.videos:
        raise ValueError(f"only {len(selected)} eligible training videos")
    manifest = []
    for _, video, group, last in selected:
        exemplar = group[0]
        manifest.append({"question_id": video, "video_id": video,
                         "question": exemplar["question"],
                         "resolved_video_path": exemplar["video_path"],
                         "inference_source_indices": list(range(last + 1)),
                         "inference_timestamps": [float(i) for i in range(last + 1)]})
    path = OUT / "mining/rollout_manifest.jsonl"
    write_new(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest))
    print(json.dumps({"videos": len(manifest), "frames": sum(len(r["inference_timestamps"]) for r in manifest),
                      "source_sha256": sha(SOURCE), "rollout_manifest_sha256": sha(path)}, sort_keys=True))


def mine(args):
    rollout_dir = OUT / "mining/rollout"
    rollout = {}
    for rank in range(8):
        for row in rows(rollout_dir / f"predictions.rank{rank:02d}.jsonl"):
            if row["question_id"] in rollout:
                raise ValueError("duplicate rollout video")
            rollout[row["question_id"]] = row
    selected = list(rows(OUT / "mining/rollout_manifest.jsonl"))
    if set(rollout) != {r["video_id"] for r in selected}:
        raise ValueError("incomplete training rollout")
    selected_by_id = {r["video_id"]: r for r in selected}
    candidates = []
    rejected = Counter()
    original = []
    by_video = defaultdict(list)
    for row in rows(SOURCE):
        original.append(row)
        if row["video_id"] in rollout:
            by_video[row["video_id"]].append(row)
    for video, group in by_video.items():
        emits = rollout[video].get("v24_emits")
        expected = len(selected_by_id[video]["inference_timestamps"])
        if not isinstance(emits, list) or len(emits) != expected:
            raise ValueError(f"missing per-frame emissions for {video}")
        gold_response = {r["frame_index"] for r in group if r["label"] == "Response"}
        fires = [i for i, e in enumerate(emits) if e.startswith("</Response>")]
        if len(fires) != len(rollout[video]["model_response_list"]):
            raise ValueError(f"fire trace mismatch for {video}")
        labels_at = defaultdict(set)
        for r in group:
            labels_at[r["frame_index"]].add(r["label"])
        seen_position = set()
        for row in group:
            pos, label = row["frame_index"], row["label"]
            if label not in quota_labels:
                continue
            if pos in seen_position:
                rejected["duplicate_source_position"] += 1
                continue
            seen_position.add(pos)
            if labels_at[pos] != {label}:
                rejected["conflicting_targets"] += 1
                continue
            fired = pos in fires
            if label == "Silence" and not fired or label == "Response" and fired:
                rejected["correct_action"] += 1
                continue
            if label == "Silence" and any(abs(pos - g) <= 8 for g in gold_response):
                rejected["silence_near_gold_event"] += 1
                continue
            if label == "Response" and any(pos - 32 <= f < pos for f in fires):
                rejected["missed_response_prior_fire"] += 1
                continue
            gold_answer = row["turns"][-1]["emit"].replace("</Response>", " ").strip()
            if label == "Response" and any(
                token_overlap(emits[f].replace("</Response>", " "), gold_answer) >= 0.30
                for f in fires if f < pos
            ):
                rejected["missed_response_answer_already_covered"] += 1
                continue
            start = max(0, pos - 31)
            if label == "Response" and any(start <= g < pos for g in gold_response):
                rejected["multiple_gold_events_in_window"] += 1
                continue
            prior = [(i, emits[i]) for i in fires if i < pos]
            memory = next(({"frame_index": i, "time_seconds": float(i),
                            "content": " ".join(e[len("</Response>"):].replace("</Response>", " ").split())}
                           for i, e in reversed(prior) if i < start), None)
            turns = []
            for i in range(start, pos + 1):
                emission = emits[i] if i < pos else "</Silence>"
                state = ("Response" if emission.startswith("</Response>") else
                         "Standby" if emission.startswith("</Standby>") else "Silence")
                turns.append({"frame_index": i, "ts": f"{i // 60:02d}:{i % 60:02d}",
                              "emit": emission, "state": state, "learn": i == pos,
                              "role": "frame", "kind": "mined"})
            candidate = dict(row, memory=memory, turns=turns, window_start=start,
                             turn_index=len(turns)-1, episode_id=-1)
            candidate["mining"] = {"question_id": video, "source_episode_id": row["episode_id"],
                                   "source_turn_index": row["turn_index"],
                                   "source_gold_answer": gold_answer if label == "Response" else None,
                                   "prior_actions": [{"frame_index": i, "emit": e} for i, e in prior],
                                   "observed_action": "Response" if fired else "Silence_or_Hold"}
            candidates.append(candidate)
    rng = random.Random(args.seed)
    selected_candidates = []
    quota = {"Silence": args.silence_quota, "Response": args.response_quota}
    gate = {"quota": quota, "candidate_count": dict(Counter(r["label"] for r in candidates)),
            "candidate_videos": {label: len({r["video_id"] for r in candidates if r["label"] == label})
                                 for label in quota},
            "discarded": dict(rejected), "rollout_videos": len(rollout),
            "rollout_manifest_sha256": sha(OUT / "mining/rollout_manifest.jsonl")}
    write_new(OUT / "mining/gate.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True), flush=True)
    for label, count in quota.items():
        choices = [r for r in candidates if r["label"] == label]
        rng.shuffle(choices)
        choices.sort(key=lambda r: (len(r["mining"]["prior_actions"]) == 0,
                                    r["frame_index"] < 64))
        per_video = Counter()
        picked = []
        for row in choices:
            if per_video[row["video_id"]] >= args.cap_per_video:
                continue
            picked.append(row)
            per_video[row["video_id"]] += 1
            if len(picked) == count:
                break
        if len(picked) != count or len(per_video) < args.min_videos_per_class:
            raise ValueError(f"quota gate failed for {label}: {len(picked)}/{count} examples, {len(per_video)} videos")
        selected_candidates.extend(picked)
    # Retain every J1-S post-response negative. Exchange only easy, response-free
    # Silence positions and a matched number of original Response presentations.
    removals = set()
    for label, count in quota.items():
        pool = [r for r in original if r["label"] == label and
                (label == "Response" or (r["memory"] is None and not any(
                    t["state"] == "Response" for t in r["turns"][:-1])))]
        rng.shuffle(pool)
        if len(pool) < count:
            raise ValueError(f"insufficient replaceable {label} examples")
        removals.update((r["episode_id"], r["turn_index"]) for r in pool[:count])
    for index, row in enumerate(selected_candidates):
        row["episode_id"] = 1_000_000 + index
    result = [r for r in original if (r["episode_id"], r["turn_index"]) not in removals]
    result.extend(selected_candidates)
    result.sort(key=lambda r: (r["episode_id"], r["turn_index"]))
    counts = Counter(r["label"] for r in result)
    if len(result) != 263760 or counts["Response"] != 41220:
        raise ValueError("matched budget or class counts failed")
    path = OUT / "manifests/train.jsonl"
    write_new(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in result))
    audit = {"seed": args.seed, "quota": quota, "discarded": dict(rejected),
             "candidate_count": dict(Counter(r["label"] for r in candidates)),
             "selected_videos": {label: len({r["video_id"] for r in selected_candidates if r["label"] == label})
                                 for label in quota},
             "labels": dict(counts), "presentations": len(result),
             "source_sha256": sha(SOURCE), "manifest_sha256": sha(path),
             "rollout_manifest_sha256": sha(OUT / "mining/rollout_manifest.jsonl"),
             "rollout_protocol_sha256": sha(rollout_dir / "manifest.json"),
             "rollout_shard_sha256": {str(i): sha(rollout_dir / f"predictions.rank{i:02d}.jsonl") for i in range(8)},
             "audit_rule": "Source-verified targets only; Silence >=9s from supervised Response; missed Response no fire in prior 32s and no earlier answer with token overlap >=0.30 to source gold answer; unique source target; full self-generated 32-frame history"}
    write_new(OUT / "manifests/audit.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("select", "mine"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--videos", type=int, default=1024)
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--silence-quota", type=int, default=512)
    p.add_argument("--response-quota", type=int, default=128)
    p.add_argument("--cap-per-video", type=int, default=4)
    p.add_argument("--min-videos-per-class", type=int, default=64)
    args = p.parse_args()
    (select if args.phase == "select" else mine)(args)


if __name__ == "__main__":
    main()
