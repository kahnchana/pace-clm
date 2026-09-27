#!/usr/bin/env python3
"""Fail closed unless the complete PACE-200 pilot artifact set is coherent."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pace_clm.artifacts import sha256_file


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_hash(path: Path, expected: str) -> None:
    require(path.is_file(), f"missing artifact: {path}")
    require(sha256_file(path) == expected, f"hash mismatch: {path}")


def main() -> None:
    smoke = read_json(ROOT / "outputs/smoke/real_model.json")
    require(smoke["hidden_size"] == 2048, "real-model hidden width changed")
    require(smoke["state_shape"] == [1, 2048], "invalid state extraction shape")
    require(smoke["action_shape"] == [3, 2048], "invalid action extraction shape")
    require(not smoke["lm_head_called"], "feature extraction invoked the LM head")
    require(not smoke["backbone_trainable_parameters"], "backbone was not frozen")
    require(smoke["cached_uncached_max_abs"] == 0, "action cache parity failed")

    manifest_ref = read_json(ROOT / "manifests/pace_manifests_v1.json")
    train_manifest = ROOT / manifest_ref["train_manifest"]
    validation_manifest = Path(manifest_ref["validation_manifest"])
    verify_hash(train_manifest, manifest_ref["train_sha256"])
    verify_hash(validation_manifest, manifest_ref["validation_sha256"])
    train_rows = read_jsonl(train_manifest)
    validation_rows = read_jsonl(validation_manifest)
    require(len(train_rows) == 768, "training manifest must contain 768 examples")
    require(len(validation_rows) == 200, "validation manifest must contain 200 videos")
    require(
        Counter(row["gold_action"] for row in train_rows)
        == {"Response": 256, "Silence": 256, "Hold": 256},
        "training manifest is not balanced",
    )
    require(
        not (
            {row["video_id"] for row in train_rows} & {row["video_id"] for row in validation_rows}
        ),
        "train/validation video overlap",
    )

    expected_features = {
        "train.pt": (768, {0: 256, 1: 256, 2: 256}),
        "pace200.pt": (10658, {0: 670, 1: 9937, 2: 51}),
    }
    for name, (examples, distribution) in expected_features.items():
        artifact = torch.load(
            ROOT / "outputs/features/v1" / name,
            map_location="cpu",
            weights_only=False,
        )
        require(tuple(artifact["features"].shape) == (examples, 2048), f"bad {name} shape")
        require(tuple(artifact["action_features"].shape) == (3, 2048), f"bad {name} actions")
        require(Counter(artifact["targets"].tolist()) == distribution, f"bad {name} labels")
        require(len(artifact["metadata"]) == examples, f"bad {name} metadata")

    result_root = ROOT / "outputs/results/pace200_v1"
    results = read_json(result_root / "results.json")
    require(results["tiny_overfit"]["accuracy"] == 1.0, "tiny-set overfit accuracy failed")
    require(results["tiny_overfit"]["examples"] == 24, "tiny-set gate used the wrong size")
    require(results["tiny_overfit"]["final_loss"] < 1e-4, "tiny-set loss did not converge")
    required_methods = {
        "raw_cosine": {42},
        "linear": {42, 43, 44},
        "mlp": {42, 43, 44},
        "clm_128": {42, 43, 44},
        "clm_256": {42, 43, 44},
        "clm_512": {42, 43, 44},
        "clm_shared_512": {42, 43, 44},
        "clm_short_512": {42, 43, 44},
    }
    observed_methods: dict[str, set[int]] = {}
    for run in results["runs"]:
        observed_methods.setdefault(run["method"], set()).add(int(run["seed"]))
        require("confusion_matrix" in run["classification"], "missing confusion matrix")
    require(observed_methods == required_methods, "method/seed matrix is incomplete")

    expected_scored = set(required_methods) | {"pace_token_generation"}
    timing = read_json(result_root / "timing_paucf/summary.json")
    require(set(timing) == expected_scored, "timing PAUC-F method matrix is incomplete")
    require(
        all(int(summary["num_sampled_videos"]) == 200 for summary in timing.values()),
        "timing scorer did not cover 200 videos",
    )
    prediction_files = sorted((result_root / "predictions").glob("*.jsonl"))
    require(len(prediction_files) == 9, "missing merged per-example prediction files")
    for path in prediction_files:
        require(sum(1 for line in path.open() if line.strip()) == 10658, f"bad row count: {path}")

    generation_root = result_root / "generated/clm_256_seed42_smoke8"
    selection = read_json(generation_root / "selection.json")
    generated_rows = read_jsonl(generation_root / "responses.rank00.jsonl")
    generated = [response for row in generated_rows for response in row["model_response_list"]]
    require(selection["selected_fires"] == 8, "generation smoke selection is incomplete")
    require(len(generated_rows) == 200, "generation smoke must preserve the 200-video schema")
    require(len(generated) == 8, "generation smoke did not emit eight responses")
    require(all(response["content"].strip() for response in generated), "empty generated response")

    provenance = read_json(result_root / "run_provenance.json")
    for field in (
        "checkpoints",
        "per_example_predictions",
        "token_prediction_shards",
        "scoring_artifacts",
    ):
        for relative, expected_hash in provenance[field].items():
            verify_hash(ROOT / relative, expected_hash)
    for field in ("selection", "responses"):
        item = provenance["generation_quality_sample"][field]
        verify_hash(Path(item["path"]), item["sha256"])
    for field in ("train", "pace200"):
        item = provenance["feature_artifacts"][field]
        verify_hash(Path(item["path"]), item["sha256"])
    verify_hash(
        Path(provenance["result_artifact"]["path"]), provenance["result_artifact"]["sha256"]
    )

    concise_note = (ROOT / "agent_notes/experiments/pace200_clm_pilot.md").read_text()
    require(
        sum(line.startswith("|---") for line in concise_note.splitlines()) == 2,
        "concise experiment note must contain exactly two tables",
    )
    print(
        json.dumps(
            {
                "status": "verified",
                "validation_videos": 200,
                "validation_decisions": 10658,
                "trained_checkpoints": len(provenance["checkpoints"]),
                "scored_methods": len(timing),
                "generation_smoke_responses": len(generated),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
