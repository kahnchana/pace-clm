#!/usr/bin/env python3
"""Freeze the exact Stream-Gemma stride-10 cohort and CLM decision selection."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT / "src"), str(STREAM_GEMMA / "src")]

from stream_gemma.data import _conversation_and_range, _parse_record

from pace_clm.experiment import read_jsonl, state_timeline
from pace_clm.labels import Action, to_action

SOURCE = STREAM_GEMMA / "datasets/markii/train.jsonl"
METADATA = STREAM_GEMMA / "datasets/markii/metadata.json"
PACE200 = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno/subset/subset_200_manifest.jsonl"
POOL = ROOT / "manifests/pace_train_stream_gemma_10pct.jsonl"
DECISIONS = ROOT / "manifests/pace_train_10pct_decisions.jsonl"
SUMMARY = ROOT / "manifests/pace_train_10pct_summary.json"


def digest(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    metadata = json.loads(METADATA.read_text())
    held_out = {row["video_id"] for row in read_jsonl(PACE200)}
    pool: list[dict] = []
    valid_index = 0
    parsed = skipped = 0
    with SOURCE.open() as f:
        for raw_index, line in enumerate(f):
            parsed += 1
            try:
                relative, question, text_stream = _parse_record(json.loads(line))
                if not question or not text_stream:
                    raise ValueError("empty question or stream")
                video_id = str(Path(relative).with_suffix(""))
                meta = metadata[video_id]
                built = _conversation_and_range(
                    question=question,
                    text_stream=text_stream,
                    metadata=meta,
                    source_fps=1,
                    keep_trailing_silence=True,
                )
                if built is None:
                    raise ValueError("empty conversation")
            except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
                skipped += 1
                continue
            if valid_index % 10 == 0:
                pool.append({"raw_index": raw_index, "video_id": video_id, "valid_index": valid_index})
            valid_index += 1
    if (parsed, valid_index, len(pool)) != (176021, 169939, 16994):
        raise RuntimeError(f"source/stride drift: {(parsed, valid_index, len(pool))}")
    overlap = sorted({row["video_id"] for row in pool} & held_out)
    if overlap:
        pool = [row for row in pool if row["video_id"] not in held_out]
    write_jsonl(POOL, pool)

    # Stratify on class and temporal location. Hash ordering makes each selection
    # independent of annotation iteration order and gives stable nested curves.
    candidates: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    records = read_jsonl(SOURCE)
    timeline_failures = 0
    for pool_row in pool:
        try:
            timeline = state_timeline(records[pool_row["raw_index"]], metadata)
        except (KeyError, TypeError, ValueError, StopIteration):
            timeline_failures += 1
            continue
        states = timeline["states"]
        response_positions = [i for i, value in enumerate(states) if value == "Response"]
        hold_run_start = 0
        split = "dev" if digest("split:" + pool_row["video_id"]) % 10 == 0 else "train"
        for i, state in enumerate(states):
            action = to_action(state).name
            if action == "Silence":
                distance = min((abs(i - j) for j in response_positions), default=9999)
                stratum = "boundary" if distance <= 4 else "other"
            elif action == "Hold":
                if i == 0 or states[i - 1] != "Standby":
                    hold_run_start = i
                offset = i - hold_run_start
                stratum = "early" if offset < 3 else "middle" if offset < 8 else "late"
            else:
                stratum = "response"
            row = {
                "split": split,
                "raw_index": pool_row["raw_index"],
                "video_id": pool_row["video_id"],
                "frame_index": int(timeline["source_start"]) + i,
                "gold_action": action,
                "gold_action_id": int(Action[action]),
                "stratum": stratum,
            }
            candidates[(split, action, stratum)].append(row)

    selected: list[dict] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    per_video: Counter[str] = Counter()
    per_video_action: Counter[tuple[str, str]] = Counter()
    targets = {
        (split, action): (900 if split == "dev" else 8100)
        for split in ("train", "dev")
        for action in ("Response", "Silence", "Hold")
    }
    for (split, action), target in targets.items():
        strata = [key for key in candidates if key[:2] == (split, action)]
        preferred = {
            "Silence": {"boundary": target // 2, "other": target - target // 2},
            "Hold": {"early": target // 3, "middle": target // 3, "late": target - 2 * (target // 3)},
            "Response": {"response": target},
        }[action]
        for key in sorted(strata):
            rows = sorted(candidates[key], key=lambda row: digest(
                f"choice:{row['video_id']}:{row['frame_index']}:{action}"
            ))
            for row in rows:
                if counts[key] >= preferred[key[2]]:
                    break
                video_id = row["video_id"]
                if per_video[video_id] >= 14 or per_video_action[(video_id, action)] >= (10 if action == "Hold" else 3):
                    continue
                selected.append(row)
                counts[key] += 1
                per_video[video_id] += 1
                per_video_action[(video_id, action)] += 1
        current = sum(counts[key] for key in strata)
        if current < target:
            chosen = {(r["video_id"], r["frame_index"]) for r in selected}
            remaining = sorted(
                (row for key in strata for row in candidates[key]),
                key=lambda row: digest(f"fallback:{row['video_id']}:{row['frame_index']}:{action}"),
            )
            for row in remaining:
                if current >= target:
                    break
                video_id = row["video_id"]
                if ((video_id, row["frame_index"]) in chosen or per_video[video_id] >= 14
                        or per_video_action[(video_id, action)] >= (10 if action == "Hold" else 3)):
                    continue
                selected.append(row)
                chosen.add((video_id, row["frame_index"]))
                counts[(split, action, row["stratum"])] += 1
                per_video[video_id] += 1
                per_video_action[(video_id, action)] += 1
                current += 1
        if current != target:
            available = sum(len(candidates[key]) for key in strata)
            videos = len({row["video_id"] for key in strata for row in candidates[key]})
            raise RuntimeError(
                f"insufficient {split}/{action}: {current}/{target}; "
                f"candidates={available}, videos={videos}, "
                f"strata={dict((key[2], len(candidates[key])) for key in strata)}"
            )
    selected.sort(key=lambda r: (r["video_id"], r["frame_index"], r["gold_action_id"]))
    write_jsonl(DECISIONS, selected)
    train_ids = {r["video_id"] for r in selected if r["split"] == "train"}
    dev_ids = {r["video_id"] for r in selected if r["split"] == "dev"}
    if train_ids & dev_ids or (train_ids | dev_ids) & held_out:
        raise RuntimeError("video leakage")
    summary = {
        "source_sha256": file_hash(SOURCE),
        "metadata_sha256": file_hash(METADATA),
        "pace200_sha256": file_hash(PACE200),
        "source_records": parsed,
        "valid_records": valid_index,
        "skipped_records": skipped,
        "stride": 10,
        "pool_before_pace200_exclusion": 16994,
        "pace200_overlap_excluded": overlap,
        "pool_size": len(pool),
        "pool_sha256": file_hash(POOL),
        "timeline_failures": timeline_failures,
        "decision_count": len(selected),
        "decision_sha256": file_hash(DECISIONS),
        "distribution": {f"{split}/{action}/{stratum}": count for (split, action, stratum), count in sorted(counts.items())},
        "candidate_distribution": {
            f"{split}/{action}/{stratum}": len(rows)
            for (split, action, stratum), rows in sorted(candidates.items())
        },
        "train_videos": len(train_ids),
        "dev_videos": len(dev_ids),
        "selection_seed": "sha256 choice/fallback; split hash modulo 10",
        "max_decisions_per_video": max(per_video.values()),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
