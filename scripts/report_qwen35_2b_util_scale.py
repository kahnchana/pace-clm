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


def gpu_samples(sample_name: str, start: str, end: str) -> dict:
    path = OUT / f"logs/{sample_name}_gpu_samples.csv"
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
        raise ValueError(f"no GPU samples in measured window for {sample_name}")
    return {
        "samples": len(utilization), "mean_gpu_util_pct": statistics.mean(utilization),
        "gpu_util_le_5_fraction": sum(value <= 5 for value in utilization) / len(utilization),
        "max_nvidia_memory_mib": max(memory),
    }


def main():
    results = []
    configurations = [(f"nccl_b{batch}_throughput", f"nccl_b{batch}", "ordered")
                      for batch in (1, 2, 4, 8, 16, 32)]
    configurations += [
        ("nccl_b32_length_sorted", "nccl_b32_length_sorted", "length sorted"),
        ("nccl_b48_length_sorted_expandable", "nccl_b48_length_sorted_expandable", "length sorted"),
        ("nccl_b48_length_sorted_cache1024", "nccl_b48_length_sorted_cache1024", "length sorted + frame cache"),
        ("nccl_b48_length_sorted_vision16", "nccl_b48_length_sorted_vision16", "length sorted + batched vision"),
        ("nccl_b48_length_sorted_predecode4", "nccl_b48_length_sorted_predecode4", "length sorted + parallel decode"),
        ("nccl_b48_length_sorted_prefetch4_cache1024", "nccl_b48_length_sorted_prefetch4_cache1024", "length sorted + cache + prefetch"),
        ("nccl_b52_length_sorted_cache1024", "nccl_b52_length_sorted_cache1024", "length sorted + frame cache"),
    ]
    for report_name, sample_name, ordering in configurations:
        path = OUT / f"reports/{report_name}.json"
        if not path.exists():
            continue
        ranks = json.loads(path.read_text())
        first = ranks[0]
        batch = first["microbatch"]
        wall = max(rank["measured_seconds"] for rank in ranks)
        actual = sum(rank["token_count"] for rank in ranks)
        padded = sum(rank["padding_tokens"] for rank in ranks)
        stress = OUT / f"reports/nccl_b{batch}_full32.json"
        if batch == 52:
            stress = OUT / "reports/nccl_b52_full32_stress.json"
        stress_ranks = json.loads(stress.read_text()) if stress.exists() else []
        result = {
            "microbatch": batch,
            "ordering": ordering,
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
        result.update(gpu_samples(sample_name, first["measured_start_utc"], first["measured_end_utc"]))
        results.append(result)
    (OUT / "reports/scale_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Qwen3.5-2B eight-H200 batch scaling", "",
             "All runs used this node (`ip-10-3-120-167`), eight GPUs, packed NCCL reduction, and a warmed Triton cache. Five warmup updates were excluded from throughput and GPU-utilization readings.", "",
             "| Physical / global batch | Row order | Presentations/s | Mean GPU util. | Readings ≤5% | Token padding | Peak allocated / reserved GiB | Full 32-frame peak allocated / reserved GiB |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for item in results:
        stress = (f"{item['full32_peak_allocated_gib']:.2f} / {item['full32_peak_reserved_gib']:.2f}"
                  if item["full32_peak_allocated_gib"] is not None else "not run")
        lines.append(f"| {item['microbatch']} / {item['global_batch']} | {item['ordering']} | "
                     f"{item['presentations_per_second']:.2f} | {item['mean_gpu_util_pct']:.1f}% | "
                     f"{item['gpu_util_le_5_fraction']:.1%} | {item['padding_fraction']:.1%} | "
                     f"{item['peak_allocated_gib']:.2f} / {item['peak_reserved_gib']:.2f} | {stress} |")
    lines += ["", "Batches 1–4 use the same 640-row source window. Larger batches use a longer contiguous source window to provide 20 updates; context mix can differ across those windows. The length-sorted 32-row run uses the same measured row multiset as ordered 32, with the first five warmup updates preserved. The 48-row run uses a longer 7,680-row window and `expandable_segments:True` to avoid allocator fragmentation. Full 32-frame checks use separate selected rows and measure capacity, not representative throughput.", "",
              "Source: per-rank benchmark JSON and one-second `nvidia-smi` samples under `outputs/jevany-qwen35-2b-util/`.", ""]
    (OUT / "reports/scale_summary.md").write_text("\n".join(lines))
    print(json.dumps({"configurations": len(results), "report": str(OUT / "reports/scale_summary.md")}))


if __name__ == "__main__":
    main()
