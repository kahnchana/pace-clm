# JevAny-style streaming controller on promoted PACE Qwen

## Summary for readers

We trained a new three-action streaming controller on top of our promoted PACE
Qwen model. At each video frame it chooses **Silence**, **Hold**, or **Response**
with a JevAny-style pointer head. The controller updates the existing Qwen LoRA
weights; when it chooses Response, the **original promoted PACE adapter**
generates the answer. We used one seed (42), one pass over the promoted 10%
MarkII training exposure, and eight H200 GPUs. Tiny J0/J1 pilots checked that
the implementation could learn; only J1 received a full training run.

The full GPT-4.1 judged results were **55.41 PAUC-F on WEB** (500 videos) and
**40.48 on EGO** (326 videos). Our earlier best, promoted PACE Qwen `long-neg`,
scored **57.47 / 46.05** under the same benchmark, judge, `pauc-f-v1` scorer,
and ω=0.5. J1 is therefore **2.06 points behind on WEB and 5.57 behind on
EGO**. On EGO it fired 3,020 times versus 1,634 for promoted PACE; 2,030 of
J1's fires were judged wasted. The controller found more useful events but lost
more to excess firing, especially in long EGO videos.

**Alignment with our best PACE Qwen training variant:** this experiment starts
from the exact promoted wave-1 cell 4 `long-neg` adapter and reconstructs its
10% video cohort, episode sampling, learned-decision masks, trailing silence,
and 4–16-frame post-response negatives. It also keeps the promoted inference
window, memory, prefix, and answer generator. It is **not a repeat of the
promoted training recipe**: PACE trained token actions *and answers* with its
SFT objective; J1 trains a contextual pointer and the LoRA on **action
cross-entropy only**, using different prompts, update units, learning rates,
and optimizer steps. The result tests that controller change on the best PACE
initialization; it does not show that the `long-neg` data recipe itself failed.

## Outcome at a glance

| Full split | Promoted PACE token controller | Frozen run-5 MLP | This J1 controller | J1 vs promoted PACE |
| --- | ---: | ---: | ---: | ---: |
| WEB judged PAUC-F | 57.47 | 53.50 | 55.41 | −2.06 |
| EGO judged PAUC-F | 46.05 | 43.95 | 40.48 | −5.57 |

| J1 split | Fire precision | Fires | Useful fires | Wasted fires | Judged recall component |
| --- | ---: | ---: | ---: | ---: | ---: |
| WEB | 65.48% | 1,240 | 812 | 428 | 48.03 |
| EGO | 32.78% | 3,020 | 990 | 2,030 | 52.90 |

The two reused MarkII validation cohorts scored 76.18 (selection) and 72.55
(final) judged PAUC-F. They are **not untouched confirmation sets**. These
MarkII scores use `markii-pauc-f-v1` and a 4-second gold window, so compare
them within that protocol, not numerically against WEB/EGO. The benchmark
controller used `response_margin=0.0`; no margin was chosen on WEB or EGO.

## Scope and interpretation for agents

This is the completed **J1** cell from the proposed [experiment
plan](../plans/jevany_streaming_experiment_plan.md). We deliberately simplified that
plan to one seed, a tiny matched J0/J1 overfit pilot, one full J1 run, and the
full WEB/EGO evaluations. J0 did **not** get a matched full run. B0 (promoted
PACE) and B1 (frozen run-5 MLP) are **historical full-split references**, not
new paired rollouts. The promoted PACE raw WEB/EGO predictions are unavailable
locally, so a paired video bootstrap against B0 was not computed. No TV/VAD,
ESTP, OVO, or StreamingBench result was generated for J1 here. Do not attribute
the benchmark difference solely to decoder adaptation: the pointer readout
also differs from the PACE token controller, and a full J0 control is absent.

At training time, each supervised decision uses the PACE episode's **gold prior
history/memory**. Validation and benchmark rollouts use the model's own prior
emissions and last-response memory. This teacher-forced training versus
self-generated inference distinction matters when reproducing the run.

## Immutable sources and environment

| Item | Exact source |
| --- | --- |
| PACE-CLM source at preflight | commit `5ee1c167edfbd2b2d54c1e0efc9ceb3ce9ff2161`; dirty status and patch **hash** in [preflight.json](../../outputs/jevany/source/preflight.json) |
| Stream-Gemma source at preflight | commit `34a771e52e11fa3d339d216762db79485fd684a8` |
| JevAny reference | commit `b273aea682e5b748a2605808e52ccafb7a378924`, Apache-2.0; local [checkout](../../third_party/JevAny) |
| Base model | `Qwen/Qwen3.5-35B-A3B`, revision `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| Initial and answer-generation adapter | [promoted `lora-qwen-w1-long-neg`](../../../stream-gemma/outputs/lora-qwen-w1-long-neg/INFERENCE.md), adapter SHA-256 `eb11e600de59edf48ef2d71bb632ae5944c0474441f264bb3ce6ceae8bb71da9` |
| Experiment config | [jevany_streaming.yaml](../../configs/jevany_streaming.yaml) |
| Python/dependencies | `../stream-gemma/.venv/bin/python`; [freeze](../../outputs/jevany/source/python-freeze.txt), with isolated `fla-core==0.3.2` and `flash-linear-attention==0.3.2` under `outputs/jevany/deps032` |
| Hardware | one node, eight NVIDIA H200 GPUs, 143,771 MiB each; original Slurm allocation 36781 |

The pinned promoted adapter's [run
manifest](../../../stream-gemma/outputs/lora-qwen-w1-long-neg/run_manifest.json)
and [training
note](../../../stream-gemma/agent_notes/docs/experiments/qwen_w1_long_neg.md)
are the authority for what the earlier best PACE run actually did. The
[replacement-run record](../../outputs/jevany/source/replacement-run.json)
pins the **successful** training script SHA-256
`d7ff9880cfa42ac93162bb1972349d25482782cd4673c7cc9842d3db85c01ba2`.
An earlier attempt used a different script and stopped before a checkpoint;
its first 50 updates are excluded from the reported run. The source worktree
was dirty at preflight, so the commit alone is insufficient provenance; use
the saved patch hash, linked implementation files, script hash, runtime
freeze, and artifacts. No standalone copy of the complete dirty patch was
saved; byte-for-byte reconstruction from the commit alone is unavailable.

## Data construction and causal inputs

The [manifest builder](../../scripts/prepare_jevany_decisions.py) calls
Stream-Gemma's `build_episode_specs` with the promoted `v25_last_response`
episode settings: MarkII training source, stride-10 video selection, seed 42,
26,080 episodes, 1 FPS, at most 32 frames, `event_interior` standby policy,
trailing silence, four to sixteen post-response negative frames, and the
promoted continuation and zero-fire sampling settings. Its exact arguments
are in `make_args()`; the [promoted run
manifest](../../../stream-gemma/outputs/lora-qwen-w1-long-neg/run_manifest.json)
provides the comparator. `Standby` is mapped to `Hold` without dropping a
learned turn.

The frozen [training manifest](../../outputs/jevany/manifests/train.jsonl)
contains **26,080 episodes and 263,760 learned decision presentations**:
215,502 Silence, 7,038 Hold, and 41,220 Response. Its SHA-256 is
`5451fab4bc38f49719044d5c348d9541c7539a4a49d258760889225c63db8176`;
the [manifest summary](../../outputs/jevany/manifests/train.summary.json)
records source hashes and the episode builder report. These are decision
presentations extracted from the same episode exposure, **not 263,760 unique
videos** and not the balanced 24,300-position frozen-head mixture.

The [controller bridge](../../src/pace_clm/jevany_streaming.py) preserves the
PACE system prompt, query, timestamped causal frame turns, prior assistant
emissions, and last-response memory. It appends the fixed
[action descriptions](../../configs/actions_v1.yaml) and a final decision
marker to the current turn. A 256-wide JevAny q/k pointer compares the marker
state with the three contextual option states in one forward pass. The frozen
vision tower supplies 144 tokens per frame at 384 px. The real 32-frame smoke
admitted 5,859 tokens; no shorter window was substituted to make training fit.

## Training recipe and checkpoints

The [training script](../../scripts/train_jevany_streaming.py) loads the
promoted PACE adapter, trains its LoRA tensors plus the new pointer, and leaves
the visual encoder frozen. It uses action cross-entropy, AdamW, gradient
clipping at 1.0, pointer LR `1e-4`, LoRA LR `2e-6`, four decisions of gradient
accumulation per GPU, eight replicated GPU workers, and one pass through the
manifest. Gradients are all-reduced across workers. The original PACE adapter
is retained separately for response generation; J1's adapted controller LoRA
is never used to write answer text.

The successful launch used [run_jevany_full_current_node.sh](../../scripts/run_jevany_full_current_node.sh)
inside the existing eight-GPU allocation with the Triton cache in
`/dev/shm/pace-jevany-triton`. The [Slurm launcher](../../scripts/sbatch_jevany_streaming.sh)
encodes the same training arguments for a fresh allocation, but the recorded
run used the **current-node launcher**, not a new `sbatch`. The launcher
resumes if `latest.json` exists. The successful run completed **263,760 global
presentations, 8,243 optimizer steps, seed 42**, with a 70.32 GiB peak
allocated on rank 0 and 91,742 seconds of training time after model load.
The [rank-0 summary](../../outputs/jevany/full-j1-seed42/summary.rank0.json),
[latest pointer](../../outputs/jevany/full-j1-seed42/latest.json), and final
[step-008243 checkpoint](../../outputs/jevany/full-j1-seed42/step-008243)
are the authoritative completed artifacts. Checkpoints include the controller
LoRA, pointer, optimizer state, and per-rank RNG states. Training loss decreased
but is not a held-out quality score.

Before the full run, a [real-model smoke](../../scripts/smoke_jevany_streaming.py)
verified all three labels, intended LoRA gradients, full visual-token
admission, and checkpoint reload. The matched nine-example, three-pass J0 and
J1 pilots are under `outputs/jevany/pilot-{j0,j1}-seed42/`; their loss
reductions only show that the plumbing can overfit. The
[reload report](../../outputs/jevany/pilot-j1-seed42/reload_report.json)
confirms the answer-generator adapter remained byte-identical.

## Evaluation contract and artifacts

The [post-training orchestrator](../../scripts/run_jevany_posttrain.sh) loads
the final checkpoint and runs complete WEB (500 videos) and EGO (326 videos)
on eight GPUs, then two reused 200-video MarkII validation cohorts. The
[rollout driver](../../scripts/eval_jevany_streaming.py) switches between the
trained controller adapter for decisions and the original PACE adapter for
forced answer generation. It uses 1 FPS sampled decisions, a causal 32-frame
window, `last-response` memory with v2 instruction and aged-out selection,
direct prefix, self-generated history, a 96-token answer cap, and zero response
margin. The [WEB](../../outputs/jevany/evaluation/benchmarks/web/manifest.json)
and [EGO](../../outputs/jevany/evaluation/benchmarks/ego/manifest.json)
manifests pin source and model hashes and question IDs. Per-rank predictions
are merged only after [merge_jevany_benchmark.py](../../scripts/merge_jevany_benchmark.py)
checks exact full-split coverage.

The [existing scorer wrapper](../../scripts/score_exp10_web_ego.py) invokes
Stream-Gemma's `zs_infer.scoring` with GPT-4.1 and `zs_infer.pauc_f` with
`omega=0.5`, scorer `pauc-f-v1`. The full
[WEB](../../outputs/jevany/evaluation/benchmarks/web/pauc_f_summary-gpt-4.1-v1.json)
and [EGO](../../outputs/jevany/evaluation/benchmarks/ego/pauc_f_summary-gpt-4.1-v1.json)
summaries and [combined report](../../outputs/jevany/reports/full-results.json)
are the score sources. The
[selection](../../outputs/jevany/evaluation/validation/selection/official/results/exp10_judged_summary.json)
and [final](../../outputs/jevany/evaluation/validation/final/official/results/exp10_judged_summary.json)
validation summaries use the separate MarkII 4-second-window scorer. The
benchmark rollouts were launched before those validation jobs, with all model
and margin choices already fixed; benchmark scores were not used for selection.

## Replication sequence

Use an isolated checkout/output tree for a new replication. The recorded
launchers and orchestrator have fixed absolute roots and output paths; edit
those paths for the new checkout. They also resume from an existing
`latest.json`; running them in this completed output tree will **resume or
reuse** artifacts rather than create an independent run. The commands below
describe the original path layout, after paths have been isolated as needed.
Ensure the pinned model, promoted adapter, MarkII data, WEB/EGO videos, and
dependency environment above are present. Then, from the PACE-CLM root:

```bash
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/outputs/jevany/deps032:$PWD/src:$PWD/scripts:$PWD/../stream-gemma/src:$PWD/../stream-gemma"
export TRITON_CACHE_DIR=/dev/shm/pace-jevany-triton
PY=../stream-gemma/.venv/bin/python
mkdir -p outputs/jevany/logs

"$PY" scripts/prepare_jevany_decisions.py \
  --output outputs/jevany/manifests/train.jsonl
sha256sum outputs/jevany/manifests/train.jsonl \
  ../stream-gemma/outputs/lora-qwen-w1-long-neg/adapter_model.safetensors \
  scripts/train_jevany_streaming.py
"$PY" scripts/smoke_jevany_streaming.py --long-window \
  --manifest outputs/jevany/manifests/train.jsonl
"$PY" scripts/train_jevany_streaming.py --mode J0 --stage pilot \
  --seed 42 --manifest outputs/jevany/manifests/train.jsonl \
  --output outputs/jevany/pilot-j0-seed42
"$PY" scripts/train_jevany_streaming.py --mode J1 --stage pilot \
  --seed 42 --manifest outputs/jevany/manifests/train.jsonl \
  --output outputs/jevany/pilot-j1-seed42
"$PY" scripts/check_jevany_reload.py \
  --run outputs/jevany/pilot-j1-seed42

# On one reserved eight-H200 node, using the recorded J1 recipe:
bash scripts/run_jevany_full_current_node.sh \
  > outputs/jevany/logs/full-current-node.log 2>&1

# After the full step-008243 checkpoint is complete, run full inference and scoring:
bash scripts/run_jevany_posttrain.sh \
  > outputs/jevany/logs/posttrain.log 2>&1
"$PY" scripts/summarize_jevany_results.py
```

For a new allocation, use the [Slurm launcher](../../scripts/sbatch_jevany_streaming.sh)
instead of the current-node launcher after changing its fixed paths and
verifying account/partition. The smoke's 32-frame case requires the isolated
FLA 0.3.2 path; the plain PyTorch gated-delta fallback OOMed on the same
H200. For the tiny matched pilots, invoke
`train_jevany_streaming.py --mode J0 --stage pilot` and `--mode J1 --stage
pilot` separately with seed 42 and distinct output directories; see the
[training script](../../scripts/train_jevany_streaming.py) for all flags.

## Conclusion and limits

The best PACE Qwen **data and initialization were carried forward**, but
changing the learned controller to a JevAny-style pointer plus decision-only
LoRA adaptation did not improve the promoted model on full WEB or EGO. The
EGO failure is mainly a precision/firing-volume problem: J1 produced more
value-adding responses (990 versus PACE's reported 754) while nearly doubling
physical fires (3,020 versus 1,634). This is one seed with no matched full J0
run and no new B0 rollout, so it supports a result for **this implemented J1
recipe**, not a general claim against pointer controllers or decoder
adaptation. The original plan's broader benchmark matrix remains unrun.

The later [generator-feedback note](controller_generator_feedback.md) records
the WEB base-Qwen answer rerun and the resulting closed-loop changes in J1's
fire schedule.

The subsequent [J1-S-R closed-loop mining cell](jevany_response_blind_next_three.md)
failed because its trained controller became nearly silent on frozen validation
and full WEB. EGO was paused after WEB confirmed the collapse; that note
records the full result and training-order analysis.
