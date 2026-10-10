#!/usr/bin/env python3
"""Run unchanged Stream-Gemma timing-only PAUC-F on decision artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(STREAM_GEMMA))

from zs_infer.markii_paucf_200 import _score_records, read_json, read_jsonl


def prediction_records(
    predictions: list[int], metadata: list[dict], manifest_rows: list[dict]
) -> list[dict]:
    fires: dict[str, list[dict]] = {}
    for predicted, row in zip(predictions, metadata, strict=True):
        if int(predicted) == 0:  # pace_clm.labels.Action.Response
            fires.setdefault(row["video_id"], []).append(
                {"time": float(row["timestamp"]), "content": "[timing-only response]"}
            )
    return [
        {
            "video_id": row["video_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "model_response_list": fires.get(row["video_id"], []),
        }
        for row in manifest_rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=Path("outputs/results/pace200_v1"))
    parser.add_argument("--token-dir", type=Path, default=Path("outputs/results/pace_token_v1"))
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--gold", type=Path)
    args = parser.parse_args()
    source_root = STREAM_GEMMA / "outputs/evals/markii-paucf-200-gemma4-base-zss7-yesno"
    manifest = read_jsonl(args.validation_manifest or source_root / "subset/subset_200_manifest.jsonl")
    gold = read_json(args.gold or source_root / "gold/paucf_4s_gold.json")["records"]
    out = args.result_dir / "timing_paucf"
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for checkpoint in sorted(args.result_dir.glob("*.seed42.pt")):
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        records = prediction_records(
            artifact["predictions"].tolist(), artifact["metadata"], manifest
        )
        summary, per_video = _score_records(records, gold, None)
        method = checkpoint.name.split(".seed", 1)[0]
        summaries[method] = summary
        (out / f"{method}.json").write_text(
            json.dumps({**summary, "per_video": per_video}, indent=2, sort_keys=True) + "\n"
        )
    token_rows = []
    for path in sorted(args.token_dir.glob("pace_token.rank*.jsonl")):
        token_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    if token_rows:
        token_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))
        records = prediction_records(
            [row["predicted_action_id"] for row in token_rows], token_rows, manifest
        )
        summary, per_video = _score_records(records, gold, None)
        summaries["pace_token_generation"] = summary
        (out / "pace_token_generation.json").write_text(
            json.dumps({**summary, "per_video": per_video}, indent=2, sort_keys=True) + "\n"
        )
    (out / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
