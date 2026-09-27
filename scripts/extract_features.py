#!/usr/bin/env python3
"""Validate extraction configuration before a real-model run is launched."""

from __future__ import annotations

import argparse
from pathlib import Path

from pace_clm.config import load_mvp_config
from pace_clm.stream_gemma_adapter import validate_promoted_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen_mvp.yaml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_mvp_config(args.config)
    project_root = args.config.resolve().parent.parent
    root = (project_root / config["source"]["stream_gemma_root"]).resolve()
    observed = validate_promoted_contract(root)
    print({"config": config["version"], "source": observed})
    if not args.validate_only:
        raise SystemExit(
            "real-model extraction requires an explicit dataset manifest; "
            "use --validate-only for the repository-foundation check"
        )


if __name__ == "__main__":
    main()
