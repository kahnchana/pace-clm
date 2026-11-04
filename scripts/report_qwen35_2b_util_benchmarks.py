#!/usr/bin/env python3
"""Summarize phase timings and throughput from the sequential 2B benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jevany-qwen35-2b-util"


def summarize(path: Path) -> dict:
    ranks = json.loads(path.read_text())
    first = ranks[0]
    wall = max(rank["measured_seconds"] for rank in ranks)
    tokens = sum(rank["token_count"] for rank in ranks)
    padding = sum(rank["padding_tokens"] for rank in ranks)
    phases = sorted({name for rank in ranks for name in rank["phase_seconds"]})
    return {
        "backend": first["backend"], "microbatch": first["microbatch"],
        "accumulation": first["accumulation"],
        "context": path.stem.split("_")[-1],
        "ranks": len(ranks), "measured_updates": first["measured_updates"],
        "wall_seconds": wall,
        "presentations_per_second": sum(rank["measured_presentations"] for rank in ranks) / wall,
        "padding_fraction": padding / (tokens + padding),
        "max_peak_allocated_gib": max(rank["peak_allocated_gib"] for rank in ranks),
        "max_peak_reserved_gib": max(rank["peak_reserved_gib"] for rank in ranks),
        "slowest_rank_phase_seconds": {
            name: max(rank["phase_seconds"].get(name, 0.0) for rank in ranks)
            for name in phases
        },
    }


def main():
    results = [summarize(path) for path in sorted((OUT / "reports").glob("*_b*_*.json"))]
    (OUT / "reports/summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Qwen3.5-2B training utilization benchmarks", "",
             "Each configuration uses eight ranks, global batch 32, the same ordered manifest window, and warmup excluded from timing.", "",
             "| Backend | Microbatch / accumulation | Context | Presentations/s | Token padding | Peak allocated / reserved GiB |",
             "|---|---:|---|---:|---:|---:|"]
    for item in results:
        lines.append(f"| {item['backend']} | {item['microbatch']} / {item['accumulation']} | "
                     f"{item['context']} | {item['presentations_per_second']:.2f} | "
                     f"{item['padding_fraction']:.1%} | "
                     f"{item['max_peak_allocated_gib']:.2f} / {item['max_peak_reserved_gib']:.2f} |")
    lines += ["", "## Slowest-rank phase time, seconds", "",
              "These phases are measured with CUDA synchronization and can perturb throughput. "
              "They help locate bottlenecks; use the presentation rate for configuration selection.", "",
              "| Backend | Batch | Context | Decode + resize | Image prepare | Vision | Input assembly | Forward | Backward | Gradient sync | Optimizer |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = ("decode_resize", "image_prepare", "vision_encode", "input_assembly",
             "forward", "backward", "gradient_sync", "optimizer")
    for item in results:
        phase = item["slowest_rank_phase_seconds"]
        values = " | ".join(f"{phase.get(name, 0.0):.1f}" for name in names)
        lines.append(f"| {item['backend']} | {item['microbatch']} | {item['context']} | {values} |")
    lines += ["", "See individual JSON files and logs for all ranks and failures.", ""]
    (OUT / "reports/summary.md").write_text("\n".join(lines))
    print(json.dumps({"runs": len(results), "report": str(OUT / "reports/summary.md")}))


if __name__ == "__main__":
    main()
