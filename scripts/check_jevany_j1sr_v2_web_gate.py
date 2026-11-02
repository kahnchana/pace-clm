#!/usr/bin/env python3
"""Decide whether the nearly silent validation pattern recurs on WEB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jevany-j1sr-v2"


def main() -> None:
    score = json.loads((OUT / "evaluation/benchmarks/web/pauc_f_summary-timing-v1.json").read_text())
    videos = int(score["num_videos"])
    if videos != 500:
        raise ValueError(f"incomplete WEB score: {videos}/500")
    responses_per_video = score["num_fires"] / videos
    pause_ego = responses_per_video < 0.1
    result = {"benchmark": "web", "videos": videos,
              "precision": score["fire_precision"],
              "pauc_recall": score["pauc_recall_component"],
              "pauc_f1": score["pauc_f"],
              "responses": score["num_fires"],
              "responses_per_video": responses_per_video,
              "pause_ego": pause_ego,
              "rule": "Pause EGO if WEB responses/video < 0.1"}
    dest = OUT / "reports/web_gate.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# J1-S-R v2 WEB timing result", "",
             f"Precision: {result['precision']:.2f}%",
             f"PAUC recall: {result['pauc_recall']:.2f}%",
             f"PAUC-F1: {result['pauc_f1']:.2f}%",
             f"Responses/video: {responses_per_video:.3f} ({result['responses']}/{videos})",
             "", f"Pause EGO: **{'yes' if pause_ego else 'no'}**", "",
             "Timing-only diagnostic; answer quality is held at maximum for in-window fires."]
    (OUT / "reports/web_timing.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if pause_ego:
        sys.exit(3)


if __name__ == "__main__":
    main()
