#!/usr/bin/env python3
"""Load a saved controller without touching the promoted generator adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STREAM = (ROOT / "../stream-gemma").resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(STREAM / "src")]

from stream_gemma.inference.qwen_stream import QwenRuntime
from pace_clm.jevany_streaming import PointerHead, controller_logits
from scripts.train_jevany_streaming import EpisodeFrames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    original = STREAM / "outputs/lora-qwen-w1-long-neg/adapter_model.safetensors"
    before = sha256(original)
    latest = json.loads((args.run / "latest.json").read_text())
    checkpoint = Path(latest["checkpoint"])
    runtime = QwenRuntime.load(
        checkpoint=str(checkpoint / "controller_adapter"),
        model_name="Qwen/Qwen3.5-35B-A3B",
        model_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
        frame_num_tokens=144, generation_prefix_mode="direct",
    )
    head = PointerHead(int(runtime.model.config.text_config.hidden_size)).to(runtime.device)
    head.load_state_dict(torch.load(checkpoint / "pointer_head.pt", map_location="cpu",
                                    weights_only=True))
    head.eval()
    row = json.loads((args.run / "pilot_manifest.jsonl").read_text().splitlines()[0])
    inputs, vision, options, decision = EpisodeFrames(runtime, response_blind=False).prepare(row)
    with torch.no_grad():
        logits = controller_logits(runtime, head, inputs, vision, options, decision,
                                   train_decoder=False)
    after = sha256(original)
    if not torch.isfinite(logits).all() or before != after:
        raise ValueError("checkpoint reload or generator isolation failed")
    report = {"checkpoint": str(checkpoint), "label": row["label"],
              "logits": logits.detach().float().cpu().tolist(),
              "original_generator_adapter_sha256": after,
              "generator_adapter_unchanged": before == after}
    (args.run / "reload_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
