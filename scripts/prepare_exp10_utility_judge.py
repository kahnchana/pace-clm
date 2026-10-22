#!/usr/bin/env python3
"""Merge E09/E10 answer candidates and build the unchanged GPT-4.1 rubric requests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SG = ROOT.parent / "stream-gemma"
sys.path.insert(0, str(SG))
from zs_infer.markii_paucf_200 import JUDGE_INSTRUCTION, JUDGE_MODEL

OUT = ROOT / "outputs/pace-clm-exp10/cache/utility"
MANIFEST = ROOT / "manifests/exp10/e09_utility_candidates.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frozen(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"frozen utility judge artifact differs: {path}")
    else:
        path.write_text(content)


def main() -> None:
    manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len(manifest) != 4000:
        raise RuntimeError("utility candidate count changed")
    keys = [(r["video_id"], int(r["frame_index"])) for r in manifest]
    wanted = set(keys)
    rows = {}
    shard_hashes = {}
    for rank in range(8):
        path = OUT / f"utility.rank{rank:02d}.pt"
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item["source_sha256"] != sha(MANIFEST) or item["world_size"] != 8 or item["max_videos_per_rank"]:
            raise RuntimeError("utility cache provenance mismatch")
        shard_hashes[str(rank)] = sha(path)
        for row in item["rows"]:
            key = (row["video_id"], int(row["frame_index"]))
            if key in rows:
                raise RuntimeError("duplicate utility cached candidate")
            rows[key] = row
    if set(rows) != wanted:
        raise RuntimeError(f"utility cache incomplete: {len(rows)}/{len(wanted)}")
    requests = {}
    mappings = []
    features = []
    for candidate, key in zip(manifest, keys, strict=True):
        row = rows[key]
        if row["category"] != candidate["category"] or not row["answer"].strip():
            raise RuntimeError("utility candidate metadata or answer mismatch")
        features.append(row["feature"].float())
        request_key = None
        if candidate["gold_window_index"] is not None:
            prompt = (f"Question: {candidate['question']}\n"
                      f"Ground Truth Answer: {candidate['gold_text']}\n"
                      f"Predicted Answer: {row['answer']}")
            request_key = hashlib.sha256(prompt.encode()).hexdigest()
            requests.setdefault(request_key, {
                "custom_id": f"utility:{request_key}", "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": JUDGE_MODEL, "messages": [
                    {"role": "system", "content": JUDGE_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ]},
            })
        mappings.append({"video_id": key[0], "frame_index": key[1],
                         "category": candidate["category"], "request_key": request_key,
                         "answer_sha256": hashlib.sha256(row["answer"].encode()).hexdigest()})
    request_path = OUT / "judge/pauc_batch_input.jsonl"
    write_frozen(request_path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in requests.values()))
    metadata = {"candidates": len(keys), "inside_window": sum(r["request_key"] is not None for r in mappings),
                "judge_requests": len(requests), "judge_model": JUDGE_MODEL,
                "rubric_sha256": hashlib.sha256(JUDGE_INSTRUCTION.encode()).hexdigest(),
                "request_payload_sha256": sha(request_path), "candidate_manifest_sha256": sha(MANIFEST),
                "cache_shard_sha256": shard_hashes, "mappings": mappings}
    write_frozen(OUT / "judge/provenance.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    feature_path = OUT / "candidate_features.pt"
    if feature_path.exists():
        raise FileExistsError(feature_path)
    torch.save({"keys": keys, "features": torch.stack(features).half(),
                "candidate_manifest_sha256": sha(MANIFEST),
                "request_payload_sha256": sha(request_path)}, feature_path)
    print(json.dumps({k: v for k, v in metadata.items() if k not in ("mappings", "cache_shard_sha256")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
