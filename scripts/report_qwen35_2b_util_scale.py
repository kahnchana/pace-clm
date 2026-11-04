#!/usr/bin/env python3
"""Report throughput-only batch scaling and sampled GPU activity."""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/jevany-qwen35-2b-util"


def gpu_samples(batch: int, start: str, end: str) -> dict:
    path = OUT / f"logs/nccl_b{batch}_gpu_samples.csv"
    low = datetime.fromisoformat(start).replace(tzinfo=None)
    high = datetime.fromisoformat(end).replace(tzinfo=None)
    utilization = []
    memory = []
    with path.open() as handle:
        for row in csv.reader(handle):
            at = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            if low <= at <= high:
                utilization.append(int(row[2]))
                memory.append(int(row[3]))
    if not utilization:
        raise ValueError(f"no GPU samples in measured window for batch {batch}")
    return {
        "samples": len(utilization), "mean_gpu_util_pct": statistics.mean(utilization),
        "gpu_util_le_5_fraction": sum(value <= 5 for value in utilization) / len(utilization),
        "max_nvidia_memory_mib": max(memory),
    }


def main():
    results = []
    for batch in (1, 2, 4, 8, 16, 32):
        path = OUT / f"reports/nccl_b{batch}_throughput.json"
        if not path.exists():
            continue
        ranks = json.loads(path.read_text())
        first = ranks[0]
        wall = max(rank["measured_seconds"] for rank in ranks)
        actual = sum(rank["token_count"] for rank in ranks)
        padded = sum(rank["padding_tokens"] for rank in ranks)
        stress = OUT / f"reports/nccl_b{batch}_full32.json"
        stress_ranks = json.loads(stress.read_text()) if stress.exists() else []
        result = {
            "microbatch": batch,
            "global_batch": first.get("global_batch", 32),
            "measured_presentations": sum(rank["measured_presentations"] for rank in ranks),
            "measured_seconds": wall,
            "presentations_per_second": sum(rank["measured_presentations"] for rank in ranks) / wall,
            "padding_fraction": padded / (actual + padded),
            "peak_allocated_gib": max(rank["peak_allocated_gib"] for rank in ranks),
            "peak_reserved_gib": max(rank["peak_reserved_gib"] for rank in ranks),
            "full32_peak_allocated_gib": max((rank["peak_allocated_gib"] for rank in stress_ranks), default=None),
            "full32_peak_reserved_gib": max((rank["peak_reserved_gib"] for rank in stress_ranks), default=None),
        }
        result.update(gpu_samples(batch, first["measured_start_utc"], first["measured_end_utc"]))
        results.append(result)
    (OUT / "reports/scale_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Qwen3.5-2B eight-H200 batch scaling", "",
             "All runs used this node (`ip-10-3-120-167`), eight GPUs, packed NCCL reduction, and a warmed Triton cache. Five warmup updates were excluded from throughput and GPU-utilization readings.", "",
             "| Physical / global batch | Presentations/s | Mean GPU util. | Readings ≤5% | Token padding | Peak allocated / reserved GiB | Full 32-frame peak allocated / reserved GiB |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for item in results:
        stress = (f"{item['full32_peak_allocated_gib']:.2f} / {item['full32_peak_reserved_gib']:.2f}"
                  if item["full32_peak_allocated_gib"] is not None else "not run")
        lines.append(f"| {item['microbatch']} / {item['global_batch']} | "
                     f"{item['presentations_per_second']:.2f} | {item['mean_gpu_util_pct']:.1f}% | "
                     f"{item['gpu_util_le_5_fraction']:.1%} | {item['padding_fraction']:.1%} | "
                     f"{item['peak_allocated_gib']:.2f} / {item['peak_reserved_gib']:.2f} | {stress} |")
    lines += ["", "Batches 1–4 use the same 640-row source window. Larger batches use a longer contiguous source window to provide 20 updates; context mix can differ across those windows. Full 32-frame checks use separate selected rows and measure capacity, not representative throughput.", "",
              "Source: per-rank benchmark JSON and one-second `nvidia-smi` samples under `outputs/jevany-qwen35-2b-util/`.", ""]
    (OUT / "reports/scale_summary.md").write_text("\n".join(lines))
    print(json.dumps({"configurations": len(results), "report": str(OUT / "reports/scale_summary.md")}))


if __name__ == "__main__":
    main()
