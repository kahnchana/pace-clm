#!/usr/bin/env python3
"""Event-aware, frozen J1-S-R mining extension. The original pilot is immutable."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from build_jevany_j1sr import ROOT, SOURCE, OUT as PILOT, rows, sha, write_new, token_overlap

OUT = ROOT / "outputs/jevany-j1sr-v2"
QUOTA = {"Silence": 512, "Response": 128}


def sources():
    original, by_video = [], defaultdict(list)
    for row in rows(SOURCE):
        original.append(row)
        by_video[row["video_id"]].append(row)
    return original, by_video


def select(args):
    _, by_video = sources()
    pilot = {r["video_id"] for r in rows(PILOT / "mining/rollout_manifest.jsonl")}
    excluded = pilot.copy()
    for path in (ROOT / "outputs/pace-clm-mixtures/validation/selection.jsonl",
                 ROOT / "outputs/pace-clm-mixtures/validation/final.jsonl",
                 PILOT / "validation/frozen.jsonl"):
        if path.exists():
            excluded.update(r["video_id"] for r in rows(path))
    eligible = []
    for video, group in by_video.items():
        if (video in excluded or len({r["question"] for r in group}) != 1 or
                len({r["video_path"] for r in group}) != 1 or
                not any(r["label"] == "Response" for r in group)):
            continue
        last = max(t["frame_index"] for r in group for t in r["turns"])
        if not 32 <= last < args.max_frames:
            continue
        priority = hashlib.sha256(f"j1sr-v2:{args.seed}:{video}".encode()).hexdigest()
        eligible.append((priority, video, group[0], last))
    eligible.sort()
    chosen = eligible[:args.videos]
    manifest = [{"question_id": video, "video_id": video, "question": exemplar["question"],
                 "resolved_video_path": exemplar["video_path"],
                 "inference_source_indices": list(range(last + 1)),
                 "inference_timestamps": [float(i) for i in range(last + 1)]}
                for _, video, exemplar, last in chosen]
    path = OUT / "mining/extra_manifest.jsonl"
    write_new(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest))
    print(json.dumps({"eligible_extra": len(eligible), "selected": len(chosen),
                      "frames": sum(len(r["inference_timestamps"]) for r in manifest),
                      "manifest_sha256": sha(path)}))


def load_rollout(path):
    result = {}
    for shard in sorted(path.glob("predictions.rank*.jsonl")):
        for row in rows(shard):
            video = row["question_id"]
            if video in result:
                raise ValueError(f"duplicate video {video}")
            result[video] = row
    return result


def gold_timeline(group, length):
    labels = defaultdict(set)
    answers = defaultdict(set)
    for row in group:
        for turn in row["turns"]:
            pos = turn["frame_index"]
            if pos < length:
                labels[pos].add(turn["state"])
                if turn["state"] == "Response":
                    answers[pos].add(turn["emit"].removeprefix("</Response>").strip())
        if row["label"] == "Response" and row["frame_index"] < length:
            answers[row["frame_index"]].add(row["turns"][-1]["emit"].removeprefix("</Response>").strip())
    onsets = sorted(pos for pos, value in labels.items() if value == {"Response"})
    events = []
    for index, start in enumerate(onsets):
        end = start
        while end + 1 < length and labels.get(end + 1) == {"Standby"}:
            end += 1
        next_start = onsets[index + 1] if index + 1 < len(onsets) else length
        event_answers = sorted(answers[start])
        events.append({"start": start, "end": end, "next_start": next_start,
                       "answer": event_answers[0] if len(event_answers) == 1 else None})
    return labels, events


def audit_video(video, group, rollout, *, pre=4, post=4):
    emits = rollout["v24_emits"]
    labels, events = gold_timeline(group, len(emits))
    fires = [i for i, e in enumerate(emits) if e.startswith("</Response>")]
    if len(fires) != len(rollout["model_response_list"]):
        raise ValueError(f"fire trace mismatch: {video}")
    if not events:
        return [], Counter({"no_verified_event": 1})
    # Each model fire can cover only an event whose scored temporal window it intersects.
    # Ambiguous nearby events mask the affected gold event rather than infer coverage.
    coverage = defaultdict(list)
    ambiguous_fires = set()
    for fire in fires:
        matches = [i for i, e in enumerate(events)
                   if e["start"] - pre <= fire <= min(e["end"] + post, e["next_start"] - 1)]
        if len(matches) == 1:
            coverage[matches[0]].append(fire)
        elif len(matches) > 1:
            ambiguous_fires.add(fire)
    candidates, rejected = [], Counter()
    # The source episodes contain painted, annotated positions that were not
    # sampled as supervised presentations. They are valid gold positions too.
    source_at = {}
    for source in group:
        for offset, turn in enumerate(source["turns"]):
            pos, label = turn["frame_index"], turn["state"]
            if pos < len(emits) and label in QUOTA:
                source_at.setdefault((pos, label), (source, offset))
    for (pos, label), (source, source_offset) in sorted(source_at.items()):
        if labels[pos] != {label}:
            rejected["conflicting_or_missing_gold_state"] += 1
            continue
        fired = pos in fires
        if fired == (label == "Response"):
            rejected["correct_action"] += 1
            continue
        if label == "Silence":
            if any(e["start"] - pre <= pos <= e["end"] + post for e in events):
                rejected["silence_event_boundary_or_scoring_window"] += 1
                continue
        else:
            event = next((e for e in events if e["start"] == pos), None)
            if event is None or event["answer"] is None:
                rejected["missing_or_conflicting_event_answer"] += 1
                continue
            index = events.index(event)
            if coverage[index]:
                rejected["response_event_already_covered"] += 1
                continue
            if any(event["start"] - pre <= f <= event["end"] + post for f in ambiguous_fires):
                rejected["response_ambiguous_fire_assignment"] += 1
                continue
            # Earlier content that describes this event makes the target ambiguous
            # even if it fell outside the narrow scoring window.
            if any(token_overlap(emits[f].removeprefix("</Response>").strip(), event["answer"]) >= .30
                   for f in fires if f < pos and f >= max(0, pos - 32)):
                rejected["response_prior_answer_overlap"] += 1
                continue
            if any(e["start"] > pos - 32 and e["start"] < pos for e in events):
                # Previous event is acceptable only if any earlier fire is assigned to it.
                prior = [i for i, e in enumerate(events) if pos - 32 < e["start"] < pos]
                if any(not coverage[i] for i in prior):
                    rejected["prior_gold_event_uncovered"] += 1
                    continue
        start = max(0, pos - 31)
        prior = [(i, emits[i]) for i in fires if i < pos]
        memory = next(({"frame_index": i, "time_seconds": float(i),
                        "content": " ".join(e.removeprefix("</Response>").split())}
                       for i, e in reversed(prior) if i < start), None)
        turns = []
        for i in range(start, pos + 1):
            emission = emits[i] if i < pos else "</Silence>"
            state = ("Response" if emission.startswith("</Response>") else
                     "Standby" if emission.startswith("</Standby>") else "Silence")
            turns.append({"frame_index": i, "ts": f"{i // 60:02d}:{i % 60:02d}",
                          "emit": emission, "state": state, "learn": i == pos,
                          "role": "frame", "kind": "mined"})
        candidate = dict(source, frame_index=pos, label=label,
                         memory=memory, turns=turns, window_start=start,
                         turn_index=len(turns)-1, episode_id=-1)
        candidate["mining"] = {"question_id": video, "source_episode_id": source["episode_id"],
                               "source_turn_index": source_offset,
                               "source_was_supervised": bool(source["turns"][source_offset]["learn"]),
                               "gold_event": event if label == "Response" else None,
                               "prior_actions": [{"frame_index": i, "emit": e} for i, e in prior],
                               "observed_action": "Response" if fired else "Silence_or_Hold"}
        candidates.append(candidate)
    return candidates, rejected


def mine(args):
    original, by_video = sources()
    pilot = load_rollout(PILOT / "mining/rollout")
    extra = load_rollout(OUT / "mining/extra_rollout")
    expected_pilot = {r["video_id"] for r in rows(PILOT / "mining/rollout_manifest.jsonl")}
    expected_extra = {r["video_id"] for r in rows(OUT / "mining/extra_manifest.jsonl")}
    if set(pilot) != expected_pilot or set(extra) != expected_extra or set(pilot) & set(extra):
        raise ValueError("incomplete or overlapping rollout")
    candidates, rejected = [], Counter()
    for video, rollout in {**pilot, **extra}.items():
        if len(rollout.get("v24_emits", [])) == 0:
            raise ValueError(f"missing emissions: {video}")
        found, discarded = audit_video(video, by_video[video], rollout)
        candidates.extend(found)
        rejected.update(discarded)
    rng = random.Random(args.seed)
    selected = []
    gate = {"quota": QUOTA, "candidate_count": dict(Counter(r["label"] for r in candidates)),
            "candidate_videos": {label: len({r["video_id"] for r in candidates if r["label"] == label}) for label in QUOTA},
            "discarded": dict(rejected), "pilot_videos": len(pilot), "extra_videos": len(extra),
            "source_sha256": sha(SOURCE), "extra_manifest_sha256": sha(OUT / "mining/extra_manifest.jsonl"),
            "pilot_rollout_protocol_sha256": sha(PILOT / "mining/rollout/manifest.json"),
            "extra_rollout_protocol_sha256": sha(OUT / "mining/extra_rollout/manifest.json"),
            "pilot_shard_sha256": {str(i): sha(PILOT / f"mining/rollout/predictions.rank{i:02d}.jsonl") for i in range(8)},
            "extra_shard_sha256": {str(i): sha(OUT / f"mining/extra_rollout/predictions.rank{i:02d}.jsonl") for i in range(8)},
            "rule": "Event span from source Response onset and consecutive Standby; fire associated only within onset-4 to end+4, unique event; Silence outside any such event window; Response event has no associated fire, ambiguous fire, overlapping prior answer, or uncovered gold event in prefix"}
    for label, count in QUOTA.items():
        choices = [r for r in candidates if r["label"] == label]
        rng.shuffle(choices)
        choices.sort(key=lambda r: (not r["mining"]["prior_actions"], r["frame_index"] < 64))
        picked, per_video = [], Counter()
        for row in choices:
            if per_video[row["video_id"]] >= args.cap_per_video:
                continue
            picked.append(row)
            per_video[row["video_id"]] += 1
            if len(picked) == count:
                break
        gate[f"selected_{label}"] = {"examples": len(picked), "videos": len(per_video)}
        selected.extend(picked)
    gate["passed"] = all(gate[f"selected_{label}"]["examples"] == count and
                         gate[f"selected_{label}"]["videos"] >= args.min_videos_per_class
                         for label, count in QUOTA.items())
    write_new(OUT / "mining/gate.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True), flush=True)
    if not gate["passed"]:
        raise ValueError("fixed quota/diversity gate failed")
    removals = set()
    for label, count in QUOTA.items():
        pool = [r for r in original if r["label"] == label and
                (label == "Response" or (r["memory"] is None and not any(
                    t["state"] == "Response" for t in r["turns"][:-1])))]
        rng.shuffle(pool)
        removals.update((r["episode_id"], r["turn_index"]) for r in pool[:count])
    for index, row in enumerate(selected):
        row["episode_id"] = 1_000_000 + index
    result = [r for r in original if (r["episode_id"], r["turn_index"]) not in removals] + selected
    result.sort(key=lambda r: (r["episode_id"], r["turn_index"]))
    counts = Counter(r["label"] for r in result)
    if len(result) != 263760 or counts["Response"] != 41220:
        raise ValueError("training budget changed")
    path = OUT / "manifests/train.jsonl"
    write_new(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in result))
    gate["train_manifest_sha256"] = sha(path)
    write_new(OUT / "manifests/audit.json", json.dumps(gate, indent=2, sort_keys=True) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("select", "pilot_audit", "mine"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--videos", type=int, default=3800)
    p.add_argument("--max-frames", type=int, default=180)
    p.add_argument("--cap-per-video", type=int, default=4)
    p.add_argument("--min-videos-per-class", type=int, default=64)
    args = p.parse_args()
    if args.phase == "select":
        select(args)
    elif args.phase == "pilot_audit":
        _, by_video = sources()
        rollout = load_rollout(PILOT / "mining/rollout")
        total, rejected = Counter(), Counter()
        videos = defaultdict(set)
        for video, trace in rollout.items():
            found, discarded = audit_video(video, by_video[video], trace)
            total.update(r["label"] for r in found)
            for r in found:
                videos[r["label"]].add(video)
            rejected.update(discarded)
        print(json.dumps({"candidates": dict(total), "videos": {k: len(v) for k, v in videos.items()},
                          "rejected": dict(rejected)}, sort_keys=True))
    else:
        mine(args)


if __name__ == "__main__":
    main()
