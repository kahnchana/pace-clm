# Step-32 to step-128 mixed ESTP-IT continuation

Authorized 2026-11-13 and launched at 20:24 UTC in allocation steps 38177.43
(training) / 38226.34 (evaluation), tmux socket `pace-estp128`. Seven-learner
step-32 exact restore passed at 20:27 UTC with zero additional optimizer steps.
The first continuation update and its correctness gate are in progress.

The user explicitly requested completion through optimizer
step 128 regardless of evaluation scores. Correctness checks remain mandatory.
This is 96 additional updates, not 128 additional updates.

Parent: `outputs/jevany-2b-rl/mixed-web-ego-episode-balanced-v2-to32/step-000032`.
New run: `outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128`.
Parent checkpoints, evaluations, schedule and source freezes are preserved.

## Training allocation and recipe

Use only allocation 38177 / ip-10-3-120-167: seven controller/learner GPUs and
one frozen answer-generator GPU. Restore model, optimizer, all seven learner
RNG states, policy version, sampler cursor and supervised-anchor calibration.
Keep the original SFT reference, frozen generator, learning rates, optimizer,
four trajectories/episode, constant-32 normalization and native PAUC-F rewards.
No inference-time score offsets, thresholds, cooldowns or calibration. Evaluation
uses the raw learned argmax and response margin zero.

Each update has four original, two WEB and two egocentric-family episodes.
EGO and ESTP-IT share the egocentric family's 0.25 objective weight; original
and WEB retain weights 0.5 and 0.25. Mixing in IT changes the egocentric objective
and defines a new training phase.

| Source | Additional episodes | Additional episode share | Total through step 128 |
|---|---:|---:|---:|
| Original | 384 | 50% | 512 |
| WEB | 192 | 25% | 256 |
| EGO | 81 | 10.546875% | 145 |
| ESTP-IT | 111 | 14.453125% | 111 |
| Total | 768 | 100% | 1,024 |

All episode selections are without replacement. Each of the 111 IT questions
comes from a different source video. Distribute 81 batches with one EGO and one
IT episode and 15 batches with two IT episodes throughout the continuation with
a fixed seed-42 schedule. Keep the first 32 updates byte-for-byte equivalent.
The frozen schedule adds 381,336 decisions, reaching 519,796 cumulative decisions
and 15,989 replay credits. Compute the complete decision budget and replay credits before launching; extend
the replay bank by cycling its existing frozen rows without changing its prefix
or coefficient.

## ESTP data and timing

Use the existing local ESTP-IT `ESTP_IT/estp.json` single-question annotations.
Preserve the entire ESTP-Bench SQ release for evaluation. Split IT by source
video using SHA256 of `estp-it-split-42:` plus the source-video ID, after applying
the existing protected benchmark/validation aliases (including clip IDs):
932 training / 117 development / 117 test videos. This is a project-defined
split, not a claimed official ESTP-IT split. Check media bounds and file hashes.

Training candidates require an explicit standing question, complete clip
bounds, no dynamic user turns, nonempty answers, native bounded positive-width
answer intervals, and a reachable decision in every interval on a fixed 1-FPS
grid. Never move the grid using gold annotations or widen answer windows.
Use the first physical 2-FPS frame at/after clip start, then stride by two.
Maintain physical indices separately from clip-relative timestamps; gold
windows are translated by the same clip origin. Do not truncate at the last
gold answer. Reject malformed records instead of treating them as silence.
Record rejection counts and selected task types in `data/audit.json`.

The 932-video training split contains 516 videos with eligible SQ episodes under
these criteria. This continuation uses 111 such videos. The filtered sample is
primarily explicit visual tasks; it does not establish contextual/multi-question
ESTP training. Task diversity and synthetic annotation limitations must remain
visible in reporting. Development/test sampling omits malformed/dynamic-query
records but does not filter otherwise valid answer windows for reachability.
Fixed probes use 128 episodes across 60 dev and 69 test source videos.

Preserve all 1,212 ESTP-Bench cases, including native zero-width and out-of-clip
answer annotations. Fifty-seven cases across 25 videos have annotated clip ends
up to 1.5 seconds beyond physical media; retain every case and its official gold,
and end inference at the last available physical frame. Score them with the existing unmodified official ESTP-F1
reference and its absolute-time schema conversion, not coerced PAUC windows.
Training PAUC-F on converted IT annotations is an explicitly documented proxy;
ESTP-F1 is reported separately with the GPT-4.1 answer judge on the authorized
gateway. This does not change learned action scores at evaluation.

## Correctness, evaluation and recovery

Before continuing beyond update 33: verify all step-32 checkpoint hashes;
restore all seven learners without optimizer updates; collect/learn update 33;
verify its learner restore and gradient/memory checks; and replay its original
collection batches independently on the evaluation node using step-32 weights.
Keep runtime pinned to PyTorch 2.5.1+cu121 / CUDA 12.1 / cuDNN 90100, clearing the
inherited library-path override. Preserve existing parity tolerances.

Allocation 38226 / ip-10-3-70-177 provides eight asynchronous evaluation workers.
Evaluate a step-32 baseline on existing cohorts plus fixed IT dev, then updates
48/64/80/96/112 on those same cohorts. Existing WEB/EGO sampled/unsampled and
heldout identities remain explicit. IT test and full ESTP-Bench are endpoint
only. Existing benchmark reuse is not an untouched-test claim.

After the trainer exits at step 128 and releases all GPUs, run the final suite
on allocation 38177. Preserve earlier asynchronous results on allocation 38226.
The final suite adds the fixed IT test probe and all 1,212 benchmark SQ cases.
Evaluation failures retry the same immutable jobs and preserve completed shards
and judgments. Quality results never request a stop, skip updates or select a
replacement endpoint checkpoint. Save each committed update; archive failed
uncommitted attempts before exact checkpoint recovery. Persistent failures remain
visible in status/recovery logs rather than being labeled successful training.

Entry points:

```bash
bash scripts/run_jevany_estp128.sh prepare
# Through srun inside the existing authorized allocations, with --cpu-bind=none:
bash scripts/run_jevany_estp128.sh train
bash scripts/run_jevany_estp128.sh eval
```

The preparation step freezes sources, manifests, hashes, schedule, replay bank,
checkpoint lineage and runtime contract. A durable supervisor manages both
correctness gating and checkpoint recovery. Training completion and final
judging must be verified before reporting the experiment complete.
