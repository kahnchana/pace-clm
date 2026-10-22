#!/usr/bin/env python3
"""Judge training-only E09/E10 candidates through the authorized GPT-4.1 portal."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.scoring import DEFAULT_GATEWAY, web_sync

OUT = ROOT / "outputs/pace-clm-exp10/cache/utility"


def load_gateway_key() -> None:
    for path in (ROOT / ".env", ROOT.parent / ".env", Path.home() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "X_API_KEY":
                os.environ["X_API_KEY"] = value.strip().strip('"').strip("'")
                break
        if os.environ.get("X_API_KEY"):
            break
    if not os.environ.get("X_API_KEY"):
        raise RuntimeError("X_API_KEY missing from .env or environment")
    os.environ.pop("OAI_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["OPENAI_BASE_URL"] = DEFAULT_GATEWAY


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=16)
    a = p.parse_args()
    judge = OUT / "judge"
    provenance = json.loads((judge / "provenance.json").read_text())
    request_path = judge / "pauc_batch_input.jsonl"
    output_path = judge / "pauc_batch_output.jsonl"
    expected = {r["custom_id"] for r in read_jsonl(request_path)}
    if len(expected) != provenance["judge_requests"]:
        raise RuntimeError("utility request count mismatch")
    if not output_path.exists():
        load_gateway_key()
        web_sync({"batch_input": request_path, "batch_output": output_path},
                 concurrency=a.concurrency, max_attempts=5, timeout=120.0)
    responses = {}
    for row in read_jsonl(output_path):
        custom_id = row["custom_id"]
        choices = ((row.get("response") or {}).get("body") or {}).get("choices") or []
        answer = str(((choices[0].get("message") or {}).get("content") if choices else "") or "").strip()
        if custom_id in expected and answer and answer[0] in "123":
            responses[custom_id] = int(answer[0])
    if set(responses) != expected:
        raise RuntimeError(f"utility judge incomplete: {len(responses)}/{len(expected)}")
    mappings = provenance["mappings"]
    targets = torch.tensor([responses[f"utility:{r['request_key']}"] / 3.0 if r["request_key"] else 0.0
                            for r in mappings], dtype=torch.float32)
    mask = torch.tensor([r["request_key"] is not None for r in mappings], dtype=torch.bool)
    features = torch.load(OUT / "candidate_features.pt", map_location="cpu", weights_only=False)
    keys = [(r["video_id"], int(r["frame_index"])) for r in mappings]
    if keys != features["keys"] or len(targets) != 4000:
        raise RuntimeError("utility labels/features misaligned")
    result_path = OUT / "utility_targets.pt"
    if result_path.exists():
        raise FileExistsError(result_path)
    torch.save({"keys": keys, "features": features["features"], "targets": targets,
                "inside_gold_window": mask, "provenance": provenance}, result_path)
    summary = {"candidates": len(targets), "judge_requests": len(expected),
               "inside_gold_window": int(mask.sum()), "outside_gold_window": int((~mask).sum()),
               "mean_inside_target": float(targets[mask].mean()),
               "rubric_sha256": provenance["rubric_sha256"],
               "request_payload_sha256": provenance["request_payload_sha256"],
               "model": provenance["judge_model"]}
    summary_path = judge / "utility_summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
