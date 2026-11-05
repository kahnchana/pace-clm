#!/usr/bin/env python3
"""Replace easy J1 negatives with distinct, history-grounded negatives."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return (row["video_id"], row["question"], row["frame_index"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("outputs/jevany/manifests/train.jsonl"))
    p.add_argument("--output", type=Path, default=Path("outputs/jevany-j1s/manifests/train.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--post-response-silence", type=int, default=10000)
    a = p.parse_args()
    rng = random.Random(a.seed)
    original = []
    final_episode = {}
    source_positions = set()
    for line in a.source.open():
        row = json.loads(line)
        original.append(row)
        final_episode[row["episode_id"]] = row
        source_positions.add(key(row))
    if not original:
        raise ValueError("source manifest is empty")
    original_labels = Counter(row["label"] for row in original)
    max_frame = defaultdict(int)
    for row in final_episode.values():
        max_frame[row["video_id"]] = max(max_frame[row["video_id"]], row["turns"][-1]["frame_index"])

    hold_candidates, silence_candidates = [], []
    seen_new = set()
    for episode_id, row in final_episode.items():
        turns = row["turns"]
        last_response = None
        for index, turn in enumerate(turns):
            if turn["state"] == "Response":
                last_response = index
            if turn["learn"] or turn["state"] not in ("Standby", "Silence"):
                continue
            new = dict(row, turn_index=index, frame_index=turn["frame_index"],
                       turns=turns[:index + 1], label="Hold" if turn["state"] == "Standby" else "Silence")
            position = key(new)
            if position in source_positions or position in seen_new:
                continue
            if last_response is None:
                continue
            # The source event painter marks only the interior of the same
            # caption event as Standby. A contiguous Response→Standby run is
            # therefore a mechanically grounded continuation.
            if new["label"] == "Hold":
                if not all(t["state"] == "Standby" for t in turns[last_response + 1:index + 1]):
                    continue
                hold_candidates.append(new)
            else:
                silence_candidates.append(new)
            seen_new.add(position)

    def priority(row):
        question = row["question"].lower()
        repeated = any(word in question for word in ("whenever", "every time", "each time", "every instance"))
        return (repeated, max_frame[row["video_id"]], rng.random())

    # Cap each video to spread exposure across videos and positions.
    def choose(rows, count, cap):
        chosen, per_video = [], Counter()
        for row in sorted(rows, key=priority, reverse=True):
            if per_video[row["video_id"]] >= cap:
                continue
            chosen.append(row)
            per_video[row["video_id"]] += 1
            if len(chosen) == count:
                break
        return chosen

    new_hold = choose(hold_candidates, len(hold_candidates), 8)
    new_silence = choose(silence_candidates, a.post_response_silence, 12)
    if len(new_silence) < a.post_response_silence:
        raise ValueError("insufficient distinct post-response Silence positions")

    # Memory-only Hold contexts lack an in-window response to tie to the
    # current event. Exclude those ambiguous presentations, never relabel them.
    ambiguous_hold = set()
    for row in original:
        if row["label"] == "Hold" and not any(
            t["state"] == "Response" for t in row["turns"][:-1]
        ):
            ambiguous_hold.add((row["episode_id"], row["turn_index"]))
    removals_needed = len(new_hold) + len(new_silence) - len(ambiguous_hold)
    easy = [row for row in original if row["label"] == "Silence" and
            row["memory"] is None and not any(t["state"] == "Response" for t in row["turns"][:-1])]
    # Remove the shortest and earliest easy negatives first; use seed for ties.
    random_ties = {id(row): rng.random() for row in easy}
    easy.sort(key=lambda row: (max_frame[row["video_id"]], row["frame_index"], random_ties[id(row)]))
    if len(easy) < removals_needed:
        raise ValueError("insufficient easy negatives")
    removed_silence = {(r["episode_id"], r["turn_index"]) for r in easy[:removals_needed]}
    removed_positions = ambiguous_hold | removed_silence
    keep = [r for r in original if (r["episode_id"], r["turn_index"]) not in removed_positions]
    result = keep + new_hold + new_silence
    result.sort(key=lambda r: (r["episode_id"], r["turn_index"]))
    labels = Counter(r["label"] for r in result)
    assert len(result) == len(original) and labels["Response"] == original_labels["Response"]
    assert all(r["turns"][-1]["frame_index"] == r["frame_index"] for r in result)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as f:
        for row in result:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "seed": a.seed, "source_sha256": digest(a.source), "manifest_sha256": digest(a.output),
        "presentations": len(result), "source_labels": dict(original_labels), "labels": dict(labels),
        "new_hold": len(new_hold), "new_post_response_silence": len(new_silence),
        "excluded_ambiguous_memory_only_hold": len(ambiguous_hold),
        "removed_easy_silence": len(removed_silence),
        "unique_source_positions": len(source_positions),
        "unique_result_positions": len({key(r) for r in result}),
        "videos": len({r["video_id"] for r in result}),
        "max_presentations_per_video": max(Counter(r["video_id"] for r in result).values()),
        "hold_eligibility": "contiguous in-window Response then Standby from event_interior painter",
    }
    a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
