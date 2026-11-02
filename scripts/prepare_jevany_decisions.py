#!/usr/bin/env python3
"""Export the promoted PACE episode decisions without changing their exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
STREAM = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(STREAM / "src"))

from stream_gemma.data import build_episode_specs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_args() -> SimpleNamespace:
    return SimpleNamespace(
        annotation_path=str(STREAM / "datasets/markii/train.jsonl"),
        metadata_path=str(STREAM / "datasets/markii/metadata.json"),
        video_root=str(STREAM / "datasets/markii/videos"),
        max_records=0, source_frame_fps=1, frame_fps=1,
        video_sample_stride=10, video_sample_fraction=0.0,
        keep_trailing_silence=True, episode_protocol="v25_last_response",
        standby_policy="event_interior", standby_lead_seconds=4.0,
        drop_first_fire_at_zero=True, standby_override_hold=True,
        zero_fire_silence_quota=2, keep_first_of_run=True,
        continuation_keep_silence=4, continuation_keep_standby=2,
        silence_sampling="uniform", silence_boundary_frames=-1,
        max_num_frames=32, keep_response_free_chunks=False,
        post_response_negative_min=4, post_response_negative_max=16,
        target_episode_count=26080, data_seed=42, seed=42,
        rolling_current_state=False, memory_k=1, event_merge_similarity=0.0,
        self_negatives=False, student_forcing_probability=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/jevany/manifests/train.jsonl")
    args = parser.parse_args()
    specs, report = build_episode_specs(make_args())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for episode_id, spec in enumerate(specs):
            turns = [dict(turn) for turn in spec.turns]
            for index, turn in enumerate(turns):
                if not turn["learn"]:
                    continue
                label = "Hold" if turn["state"] == "Standby" else turn["state"]
                counts[label] += 1
                row = {
                    "episode_id": episode_id, "turn_index": index,
                    "video_id": spec.video_id, "video_path": spec.video_path,
                    "question": spec.question, "label": label,
                    "frame_index": int(turn["frame_index"]),
                    "window_start": int(spec.chunk_start),
                    "memory": None if spec.memory is None else {
                        "frame_index": spec.memory.frame_index,
                        "time_seconds": spec.memory.time_seconds,
                        "content": spec.memory.content,
                    },
                    "turns": turns[: index + 1], "split": "train",
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    expected = {"Response": 41220, "Silence": 215502, "Hold": 7038}
    if dict(counts) != expected or len(specs) != 26080:
        raise RuntimeError(f"PACE exposure drift: {len(specs)} episodes, {dict(counts)} decisions")
    summary = {
        "episodes": len(specs), "decisions": sum(counts.values()),
        "labels": dict(counts), "manifest_sha256": sha256(args.output),
        "annotation_sha256": sha256(Path(make_args().annotation_path)),
        "metadata_sha256": sha256(Path(make_args().metadata_path)),
        "source_report": report,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_report"}, sort_keys=True))


if __name__ == "__main__":
    main()
