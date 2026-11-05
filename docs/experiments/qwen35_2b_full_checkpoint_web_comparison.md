# Qwen3.5-2B full-data early-checkpoint WEB comparison

Date: 2026-11-05. Training remained stopped. Checkpoints 10 and 25 were
evaluated concurrently on **only** `ip-10-3-120-167`, using GPUs 0–3 and
4–7 respectively. Each run covered all 500 unique WEB videos and passed the
unchanged GPT-4.1 `pauc-f-v1` judge's completeness check. Their protocol
manifests match the [step-505 evaluation](qwen35_2b_full_step505_web.md) in
every field except the controller adapter and pointer-head hashes. The
earlier final 2B J1-S result used the same WEB protocol.

| Checkpoint | Training presentations | PAUC-F | Fire precision | PAUC recall | Fires | Useful | Wasted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-data step 10 | 8,672 | **34.89** | 45.18% | 28.42 | 301 | 136 | 165 |
| Full-data step 25 | 22,464 | **32.27** | 31.91% | 32.64 | 1,006 | 321 | 685 |
| Full-data step 505 | 426,464 | **47.12** | 59.56% | 38.97 | 863 | 514 | 349 |
| Earlier final 2B J1-S | 263,760 | **54.67** | 70.74% | 44.54 | 991 | 701 | 290 |

Step 25 scored **2.62 points below step 10**. It fired 705 more times, but
its fire precision fell by 13.27 points and wasted fires rose from 165 to
685. Step 505 recovered to **14.85 points above step 25**, yet remained
**7.55 points below** the earlier final 2B J1-S checkpoint. These early
training checkpoints have different exposure and optimizer progress; the
non-monotonic scores do not isolate a single cause. The full-data run was
stopped with step 505 as its latest saved checkpoint and has no final-epoch
WEB result.

Artifacts:

| Step | Merged predictions | Judged score | Protocol |
|---|---|---|---|
| 10 | [predictions](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step10/predictions.jsonl) | [score](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step10/pauc_f_summary-gpt-4.1-v1.json) | [manifest](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step10/manifest.json) |
| 25 | [predictions](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step25/predictions.jsonl) | [score](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step25/pauc_f_summary-gpt-4.1-v1.json) | [manifest](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step25/manifest.json) |
| 505 | [predictions](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/predictions.jsonl) | [score](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/pauc_f_summary-gpt-4.1-v1.json) | [manifest](../../outputs/jevany-qwen35-2b-full/evaluation/benchmarks/web_step505/manifest.json) |

The parallel runner is
[`run_qwen35_2b_full_web_checkpoints.sh`](../../scripts/run_qwen35_2b_full_web_checkpoints.sh).
