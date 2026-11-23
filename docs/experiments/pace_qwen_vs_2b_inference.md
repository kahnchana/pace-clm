# PACE Qwen MoE versus PACE-2B: inference time and PAUC-F

Compiled 2026-11-16 from existing records only. No new inference, judging, or evaluation was run.

Both systems use Qwen3.5-35B-A3B MoE answer generation. The promoted PACE Qwen
model uses that MoE for streaming decisions as well; the latest Jevany PACE-2B
uses a Qwen3.5-2B controller and the frozen promoted MoE answer generator.
PACE-2B results below are from the four-source balanced continuation at step 160.

| Setup | WEB avg. time/example | EGO avg. time/example | WEB PAUC-F (ω=0.5) | EGO PAUC-F (ω=0.5) |
| --- | ---: | ---: | ---: | ---: |
| Promoted PACE Qwen MoE (`qwen-w1-long-neg`) | ~39.6 s (estimated) | ~828 s (estimated) | 57.48 | 46.05 |
| PACE-2B controller + MoE generator, step 160 | 5.78 s (recorded timers) | 87.2 s (recorded timers) | 58.43 | 44.22 |
| Approximate speedup | 6.8× | 9.5× | — | — |
| Approximate time reduction | 85.4% | 89.5% | — | — |

Scores are percentages, GPT-4.1 judged, on full WEB (500 examples) and EGO
(326 examples). PACE-2B selects the raw learned action argmax with response
margin zero. The score changes relative to the promoted MoE are +0.95 points
on WEB and -1.83 points on EGO.

## Timing method and limits

These ratios are indicative estimates, not controlled speedup measurements.
The MoE figures estimate average worker time from whole-job wall durations,
assuming eight parallel workers:

- WEB: `2,472 seconds × 8 / 500 = 39.552 seconds/example`.
- EGO: `33,750 seconds × 8 / 326 = 828.221 seconds/example`.

Whole-job durations include loading, preparation, worker imbalance, and other
job overhead. Multiplying wall time by eight does not recover actual measured
per-example latency. The historical MoE jobs also used an older implementation;
the retained inference-speed note documents redundant frame preprocessing in
that generation path. The ratios therefore cannot isolate the benefit of
switching the controller model.

PACE-2B averages sum each example's `jevany_decision_seconds` and
`jevany_generation_seconds`, then divide by the number of examples:

| Task | Controller seconds/example | Generation seconds/example | Combined seconds/example |
| --- | ---: | ---: | ---: |
| WEB | 1.713 | 4.071 | 5.784 |
| EGO | 81.531 | 5.627 | 87.158 |

These timers exclude model loading and video preparation outside the timed
calls. They measure accumulated work over a complete example, not latency to
one emitted answer. No comparable four-task WEB/EGO/TV/VAD timing average was
found in the inspected records.

## Training exposure

Full WEB/EGO scores include examples seen during the new RL series. They are
not heldout-only generalization results. The step-160 report separates them:

| Task | Seen in RL: examples | Seen PAUC-F | Held out of RL: examples | Heldout PAUC-F |
| --- | ---: | ---: | ---: | ---: |
| WEB | 400 | 58.51 | 100 | 58.03 |
| EGO | 145 | 45.72 | 181 | 43.18 |

## Existing sources

- [PACE-2B step-160 predictions and timers](../../outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/full-step160/rollout/predictions.jsonl).
- [PACE-2B step-160 scores and cohort counts](../../outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160/evaluation/full-step160/result.json).
- [Timer implementation](../../scripts/eval_jevany_streaming.py) and [ω=0.5 scoring](../../scripts/jevany_mixed_reward.py).
- [Promoted MoE scores](/fsx/home/kranasinghe/repo/stream-gemma/agent_notes/docs/experiments/promoted-qwen-c4-10p.md).
- [Historical MoE job durations, section 6.1](/fsx/home/kranasinghe/repo/stream-gemma/agent_notes/docs/experiments/qwen_w1_long_neg.md): WEB job 1739, 00:41:12; EGO job 1740, 09:22:30; each eight H200 GPUs. The original WEB/EGO prediction directories cited there are absent from this checkout, so this report relies on the retained provenance note.
- [Historical MoE preprocessing analysis](/fsx/home/kranasinghe/repo/stream-gemma/agent_notes/docs/inference_speed.md).
