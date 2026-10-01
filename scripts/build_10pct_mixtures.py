#!/usr/bin/env python3
"""Freeze mask-derived 24.3k decision manifests for the mixture ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(SG / "src")]

from stream_gemma.data import build_episode_specs
from pace_clm.experiment import read_jsonl, state_timeline
from pace_clm.labels import Action


def digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, rows: list[dict]) -> None:
    rows.sort(key=lambda r: (r["video_id"], r["frame_index"]))
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def stratum(states: list[str], index: int) -> str:
    state = states[index]
    if state == "Response":
        return "response"
    if state == "Silence":
        nearest = min((abs(index - j) for j, s in enumerate(states) if s == "Response"), default=9999)
        return "boundary" if nearest <= 4 else "other"
    first = index
    while first and states[first - 1] == "Standby":
        first -= 1
    offset = index - first
    return "early" if offset < 3 else "middle" if offset < 8 else "late"


def select(pool: list[dict], shares: dict[str, float], *, label: str, budget: int = 24300) -> tuple[list[dict], dict]:
    by_action = defaultdict(list)
    for row in pool:
        by_action[row["gold_action"]].append(row)
    targets = {name: round(budget * share) for name, share in shares.items()}
    targets["Silence"] += budget - sum(targets.values())
    for name, rows in by_action.items():
        rows.sort(key=lambda r: digest(f"{label}:{r['video_id']}:{r['frame_index']}"))
    selected, counts, per_video = [], Counter(), Counter()
    for name in ("Hold", "Response", "Silence"):
        for row in by_action[name]:
            if counts[name] == targets[name]:
                break
            if per_video[row["video_id"]] == 14:
                continue
            selected.append(row)
            counts[name] += 1
            per_video[row["video_id"]] += 1
    if len(selected) < budget:
        chosen = {(r["video_id"], r["frame_index"]) for r in selected}
        remaining = sorted(pool, key=lambda r: digest(f"{label}:fallback:{r['video_id']}:{r['frame_index']}"))
        for row in remaining:
            if len(selected) == budget:
                break
            key = row["video_id"], row["frame_index"]
            if key in chosen or per_video[row["video_id"]] == 14:
                continue
            selected.append(row)
            chosen.add(key)
            counts[row["gold_action"]] += 1
            per_video[row["video_id"]] += 1
    report = {"target": targets, "achieved": dict(counts), "eligible_unique": {k: len(v) for k, v in by_action.items()},
              "unique_videos": len(per_video), "repeated_positions": len(selected) - len({(r['video_id'], r['frame_index']) for r in selected}),
              "maximum_per_video": max(per_video.values(), default=0)}
    if len(selected) != budget:
        raise RuntimeError(f"{label}: only {len(selected)} of {budget} unique capped decisions: {report}")
    return selected, report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = read_jsonl(ROOT / "manifests/pace_train_10pct_decisions.jsonl")
    train_videos = {r["video_id"] for r in frozen if r["split"] == "train"}
    dev_videos = {r["video_id"] for r in frozen if r["split"] == "dev"}
    pool = read_jsonl(ROOT / "manifests/pace_train_stream_gemma_10pct.jsonl")
    metadata = json.loads((SG / "datasets/markii/metadata.json").read_text())
    records = [json.loads(line) for line in (SG / "datasets/markii/train.jsonl").read_text().splitlines()]
    timeline_by_id = {}
    long_free = {}
    for source in pool:
        raw = source["raw_index"]
        timeline = state_timeline(records[raw], metadata)
        video_id = timeline["video_id"]
        if video_id not in train_videos:
            continue
        timeline_by_id[video_id] = (raw, timeline)
        states = timeline["states"]
        eligible = set()
        for i in range(len(states)):
            if states[i] != "Silence":
                continue
            start = i
            while start and states[start-1] == "Silence":
                start -= 1
            end = i + 1
            while end < len(states) and states[end] == "Silence":
                end += 1
            if end - start >= 32:
                eligible.add(timeline["source_start"] + i)
        long_free[video_id] = eligible
    if train_videos != set(timeline_by_id):
        raise RuntimeError("training video cohort mismatch")
    video_by_path = {
        str(Path(metadata[video_id].get("path") or SG / "datasets/markii/videos" / f"{video_id}.mp4")): video_id
        for video_id in train_videos
    }

    loaded = torch.load(SG / "outputs/lora-qwen-w1-long-neg/training_args.bin", map_location="cpu", weights_only=False)
    loaded.annotation_path = str(SG / "datasets/markii/train.jsonl")
    loaded.metadata_path = str(SG / "datasets/markii/metadata.json")
    loaded.video_root = str(SG / "datasets/markii/videos")
    loaded.target_episode_count = 0
    condition_pool = {}
    preflight = {}
    for condition in ("base", "w1"):
        recipe = copy.copy(loaded)
        recipe.keep_trailing_silence = condition == "w1"
        recipe.post_response_negative_min = 4 if condition == "w1" else 0
        recipe.post_response_negative_max = 16 if condition == "w1" else 0
        specs, report = build_episode_specs(recipe)
        positions = {}
        presentations = Counter()
        for spec in specs:
            video_id = video_by_path.get(str(Path(spec.video_path)))
            if video_id is None:
                continue
            raw, timeline = timeline_by_id[video_id]
            states = timeline["states"]
            for turn in spec.turns:
                if not turn["learn"]:
                    continue
                frame = int(turn["frame_index"])
                local = frame - int(timeline["source_start"])
                if not 0 <= local < len(states):
                    raise RuntimeError((condition, video_id, frame))
                state = states[local]
                if state != turn["state"]:
                    raise RuntimeError(f"{condition} state mismatch {video_id}:{frame}: {state} vs {turn['state']}")
                action = "Hold" if state == "Standby" else state
                key = video_id, frame
                positions[key] = {"split": "train", "raw_index": raw, "video_id": video_id,
                                  "frame_index": frame, "gold_action": action,
                                  "gold_action_id": int(Action[action]), "stratum": stratum(states, local),
                                  "mask_source": condition}
                presentations[action] += 1
        condition_pool[condition] = list(positions.values())
        preflight[condition] = {"episodes_before_cap": len(specs), "learn_presentations": dict(presentations),
                                "unique_decisions": dict(Counter(r["gold_action"] for r in positions.values())),
                                "unique_videos": len({key[0] for key in positions}), "builder_report": report}
        print(json.dumps({"condition": condition, **{k: preflight[condition][k] for k in ("episodes_before_cap", "learn_presentations", "unique_decisions", "unique_videos")}}), flush=True)
        del specs
    run3, report3 = select(condition_pool["base"], {"Silence": .755, "Hold": .032, "Response": .213}, label="run3")
    run4, report4 = select(condition_pool["w1"], {"Silence": .817, "Hold": .027, "Response": .156}, label="run4")
    n_replace = round(sum(r["gold_action"] == "Silence" for r in run4) * .1)
    existing = {(r["video_id"], r["frame_index"]) for r in run4}
    free_pool = [r for r in condition_pool["w1"] if r["gold_action"] == "Silence"
                 and r["frame_index"] in long_free[r["video_id"]] and (r["video_id"], r["frame_index"]) not in existing]
    # Response-free stretches are available in the full training timeline, not only W1 learned turns.
    for video_id, eligible in long_free.items():
        raw, timeline = timeline_by_id[video_id]
        states = timeline["states"]
        for frame in eligible:
            key = video_id, frame
            if key in existing:
                continue
            local = frame - timeline["source_start"]
            free_pool.append({"split": "train", "raw_index": raw, "video_id": video_id,
                              "frame_index": frame, "gold_action": "Silence", "gold_action_id": int(Action.Silence),
                              "stratum": stratum(states, local), "mask_source": "long_response_free"})
    free_pool = list({(r["video_id"], r["frame_index"]): r for r in free_pool}.values())
    free_pool.sort(key=lambda r: (bool(any(s == "Response" for s in timeline_by_id[r["video_id"]][1]["states"])),
                                  digest(f"run5:{r['video_id']}:{r['frame_index']}")))
    per_video = Counter(r["video_id"] for r in run4)
    removable = sorted((r for r in run4 if r["gold_action"] == "Silence"),
                       key=lambda r: digest(f"run5:remove:{r['video_id']}:{r['frame_index']}"))
    replacements = []
    removed = []
    for candidate in free_pool:
        if len(replacements) == n_replace:
            break
        if per_video[candidate["video_id"]] >= 14:
            donor = next((r for r in removable if r["video_id"] == candidate["video_id"] and r not in removed), None)
        else:
            donor = next((r for r in removable if r not in removed), None)
        if donor is None:
            continue
        removed.append(donor)
        replacements.append(candidate)
        per_video[donor["video_id"]] -= 1
        per_video[candidate["video_id"]] += 1
    if len(replacements) != n_replace:
        raise RuntimeError(f"run5 replacement infeasible: {len(replacements)}/{n_replace}")
    removed_keys = {(r["video_id"], r["frame_index"]) for r in removed}
    run5 = [r for r in run4 if (r["video_id"], r["frame_index"]) not in removed_keys] + replacements
    if len(run5) != 24300 or len({(r['video_id'], r['frame_index']) for r in run5}) != 24300:
        raise RuntimeError("run5 uniqueness/budget failure")
    for name, rows in (("run3", run3), ("run4", run4), ("run5", run5)):
        path = args.output_dir / f"{name}.jsonl"
        write(path, rows)
        preflight[name] = {"sha256": sha(path), "decisions": len(rows), "states": dict(Counter(r["gold_action"] for r in rows)),
                           "strata": dict(Counter(f"{r['gold_action']}/{r['stratum']}" for r in rows)),
                           "videos": len({r["video_id"] for r in rows}), "repeated_positions": len(rows) - len({(r['video_id'], r['frame_index']) for r in rows})}
    preflight["run3"]["selection"] = report3
    preflight["run4"]["selection"] = report4
    preflight["run5"]["long_response_free_eligible"] = len({(r['video_id'], r['frame_index']) for r in free_pool})
    preflight["run5"]["replaced_silence"] = n_replace
    preflight["video_splits"] = {"train": len(train_videos), "dev": len(dev_videos), "overlap": len(train_videos & dev_videos)}
    (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: preflight[k] for k in ("run3", "run4", "run5")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
