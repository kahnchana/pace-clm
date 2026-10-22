#!/usr/bin/env python3
"""Freeze one shared training-video rollout for E01 and E05 preparation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
OUT = ROOT / "manifests/exp10"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen training rollout artifact differs: {path}")
    else:
        path.write_text(content)


def caption_runs(stream: list, physical_duration: float) -> list[dict]:
    previous = None
    runs = []
    for row in stream or []:
        end = float(row[1]) if len(row) > 1 else 0.0
        caption = row[2] if len(row) > 2 and isinstance(row[2], str) else ""
        if end > physical_duration + 1e-9:
            break
        if not caption:
            previous = None
            continue
        if caption != previous:
            runs.append({"annotation_end_time": end, "gold_text": caption,
                         "response_mark": min(float(math.ceil(end)), physical_duration)})
        previous = caption
    return runs


def main() -> None:
    selected_path = ROOT / "outputs/pace-clm-mixtures/manifests/run5.jsonl"
    selected = [json.loads(line) for line in selected_path.read_text().splitlines() if line.strip()]
    by_video = {}
    max_frame_by_video = {}
    for row in selected:
        video_id = row["video_id"]
        raw = int(row["raw_index"])
        if video_id in by_video and by_video[video_id] != raw:
            raise RuntimeError("multiple training annotation indices per video")
        by_video[video_id] = raw
        max_frame_by_video[video_id] = max(max_frame_by_video.get(video_id, -1), int(row["frame_index"]))
    by_raw = {raw: video for video, raw in by_video.items()}
    if len(by_raw) != len(by_video):
        raise RuntimeError("multiple videos share an annotation record")
    metadata_path = SG / "datasets/markii/metadata.json"
    annotations_path = SG / "datasets/markii/train.jsonl"
    metadata = json.loads(metadata_path.read_text())
    records = {}
    with annotations_path.open() as f:
        for raw, line in enumerate(f):
            if raw in by_raw:
                records[by_raw[raw]] = json.loads(line)
    if set(records) != set(by_video):
        raise RuntimeError("missing training annotation")
    training_rows = []
    gold_records = []
    from sys import path as sys_path
    sys_path.insert(0, str(SG / "src"))
    from stream_gemma.data import _parse_record

    for video_id in sorted(by_video):
        relative, question, stream = _parse_record(records[video_id])
        if str(Path(relative).with_suffix("")) != video_id:
            raise RuntimeError(f"training video annotation mismatch: {video_id}")
        meta = metadata[video_id]
        count = int(meta["num_frames"])
        duration = float(meta["duration"])
        if count < 1 or duration <= 0:
            raise RuntimeError(f"invalid video metadata: {video_id}")
        times = [float(i) for i in range(count)]
        if max_frame_by_video[video_id] >= count:
            raise RuntimeError(f"selected position exceeds physical frames: {video_id}")
        video_path = meta.get("path") or str(SG / "datasets/markii/videos" / f"{video_id}.mp4")
        if not Path(video_path).is_file():
            raise FileNotFoundError(video_path)
        training_rows.append({
            "question_id": video_id, "video_id": video_id, "question": question,
            "resolved_video_path": video_path, "inference_source_indices": list(range(count)),
            "inference_timestamps": times, "inference_frame_count": count,
            "raw_index": by_video[video_id], "physical_duration": duration,
        })
        runs = caption_runs(stream, duration)
        windows = []
        for i, run in enumerate(runs):
            start = float(run["response_mark"])
            end = min(start + 4.0,
                      float(runs[i + 1]["response_mark"]) if i + 1 < len(runs) else math.inf,
                      duration)
            windows.append({
                "window_index": i, "response_mark": start, "window_start": start,
                "window_end": end, "end_inclusive": False,
                "length_seconds": max(0.0, end-start), "gold_text": run["gold_text"],
                "annotation_end_time": run["annotation_end_time"],
                "decision_timestamps": [t for t in times if start <= t < end],
            })
        gold_records.append({"question_id": video_id, "video_id": video_id,
                             "question": question, "physical_video_end": duration,
                             "windows": windows})
    old_valid = set()
    for name in ("selection", "confirmation"):
        old_valid |= {json.loads(line)["video_id"] for line in (OUT / f"{name}.jsonl").read_text().splitlines()}
    if old_valid & set(by_video):
        raise RuntimeError("training/evaluation video overlap")
    manifest_path = OUT / "training_rollout.jsonl"
    gold_path = OUT / "training_rollout.gold.json"
    write_frozen(manifest_path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in training_rows))
    write_frozen(gold_path, json.dumps({"protocol_version": "markii-paucf-200-zss7-yesno-v1",
                                        "window_seconds": 4.0, "boundary_semantics": "[start,end)",
                                        "records": gold_records}, indent=2, sort_keys=True) + "\n")
    summary = {"videos": len(training_rows), "positions": len(selected),
               "frames": sum(r["inference_frame_count"] for r in training_rows),
               "gold_windows": sum(len(r["windows"]) for r in gold_records),
               "source_run5_sha256": sha(selected_path), "source_annotations_sha256": sha(annotations_path),
               "source_metadata_sha256": sha(metadata_path),
               "manifest_sha256": sha(manifest_path), "gold_sha256": sha(gold_path)}
    write_frozen(OUT / "training_rollout.freeze.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
