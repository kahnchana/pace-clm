#!/usr/bin/env python3
"""Write the resolved provenance record for the PACE-200 pilot."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STREAM_GEMMA = (ROOT / "../stream-gemma").resolve()
sys.path.insert(0, str(ROOT / "src"))

from pace_clm.artifacts import git_state, sha256_file
from pace_clm.config import load_mvp_config


def main() -> None:
    config_path = ROOT / "configs/qwen_mvp.yaml"
    actions_path = ROOT / "configs/actions_v1.yaml"
    manifest_ref = ROOT / "manifests/pace_manifests_v1.json"
    train_features = ROOT / "outputs/features/v1/train.pt"
    val_features = ROOT / "outputs/features/v1/pace200.pt"
    results = ROOT / "outputs/results/pace200_v1/results.json"
    result_root = results.parent
    token_root = ROOT / "outputs/results/pace_token_v1"
    smoke_root = result_root / "generated/clm_256_seed42_smoke8"
    smoke_selection = json.loads((smoke_root / "selection.json").read_text())
    smoke_rows = [
        json.loads(line)
        for line in (smoke_root / "responses.rank00.jsonl").read_text().splitlines()
        if line
    ]
    smoke_responses = [
        response for row in smoke_rows for response in row.get("model_response_list", [])
    ]

    def hashed_files(paths) -> dict[str, str]:
        return {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(paths)
            if path.is_file()
        }

    def package_version(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    payload = {
        "version": "pace-clm-pilot-v1",
        "pace_clm_git": git_state(ROOT),
        "stream_gemma_git": git_state(STREAM_GEMMA),
        "resolved_config": load_mvp_config(config_path),
        "config_sha256": sha256_file(config_path),
        "actions": yaml.safe_load(actions_path.read_text()),
        "actions_sha256": sha256_file(actions_path),
        "manifests": json.loads(manifest_ref.read_text()),
        "manifest_reference_sha256": sha256_file(manifest_ref),
        "feature_artifacts": {
            "train": {"path": str(train_features), "sha256": sha256_file(train_features)},
            "pace200": {"path": str(val_features), "sha256": sha256_file(val_features)},
        },
        "result_artifact": {"path": str(results), "sha256": sha256_file(results)},
        "checkpoints": hashed_files(result_root.glob("*.pt")),
        "per_example_predictions": hashed_files((result_root / "predictions").glob("*.jsonl")),
        "token_prediction_shards": hashed_files(token_root.glob("*.jsonl")),
        "scoring_artifacts": hashed_files(
            [
                result_root / "pace_token_results.json",
                result_root / "timing_paucf/summary.json",
                result_root / "RESULTS.md",
            ]
        ),
        "generation_quality_sample": {
            "selection": {
                "path": str(smoke_root / "selection.json"),
                "sha256": sha256_file(smoke_root / "selection.json"),
            },
            "responses": {
                "path": str(smoke_root / "responses.rank00.jsonl"),
                "sha256": sha256_file(smoke_root / "responses.rank00.jsonl"),
            },
            "selected_fires": smoke_selection["selected_fires"],
            "nonempty_responses": sum(
                bool(response.get("content", "").strip()) for response in smoke_responses
            ),
            "selection_mode": smoke_selection["selection_mode"],
        },
        "feature_contract": {
            "pooling": "final valid attended non-vision language token",
            "hidden_size": 2048,
            "history": "common gold prior-action history",
            "causal_window_frames": 32,
            "lm_head_used_for_features": False,
        },
        "training_seeds": [42, 43, 44],
        "software": {
            name: package_version(name) for name in ("torch", "transformers", "peft", "numpy")
        },
        "judge_status": {
            "semantic_pauc_f": "not run: timing-only evaluation by design",
            "timing_only_pauc_f": "complete through unchanged Stream-Gemma scorer",
        },
    }
    output = ROOT / "outputs/results/pace200_v1/run_provenance.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
