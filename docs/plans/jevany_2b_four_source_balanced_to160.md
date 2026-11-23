# Four-source balanced continuation, steps 129–160

**Completed:** training reached step 160 on 2026-11-16 at 09:01 UTC, and the
128/144/160 development evaluations completed by 11:14:58 UTC. See the
[consolidated setup and results](../experiments/jevany_2b_rl_current_setup.md).
The separately recovered previous endpoint suite is still pending its final
official scoring; its status does not change this continuation's completion.

Authorized 2026-11-15. Continue the committed step-128 model, optimizer, all
seven learner RNG states, sampler cursor and supervised-anchor calibration.
Parent run: `outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128`.
New run: `outputs/jevany-2b-rl/mixed-web-ego-estp-balanced-v4-to160`.

Each of 32 optimizer updates uses two original, two WEB, two native EGO and
two ESTP-IT complete episodes, with four causal trajectories per episode.
Each source receives objective weight 0.25. Keep the fixed denominator of
global trajectories times 32: longer episodes still contribute more decisions.
Keep RLOO, learning rates, original frozen SFT reference, generator, reward,
KL/entropy regularization and calibrated supervised replay unchanged.
Inference always uses raw learned argmax and response margin zero.

The phase samples 64 episodes per source without within-phase repeats. Use
unseen original and WEB episodes and 64 new ESTP-IT training source videos.
All 145 native EGO training episodes were already used by step 128; reuse 64
of them, as explicitly accepted by the user. Do not consume the 181 held-out
EGO episodes. Preserve the existing IT source-video splits and filtering.
Extend supervised replay by cycling its frozen bank, preserving its prefix,
decision-based credit rate and coefficient. Freeze manifests, hashes and source
snapshots before collection; distinguish phase exposure from cumulative exposure.

Train on allocation 38177 / ip-10-3-120-167 using seven learners and one frozen
generator. Restore all learners without updates, collect update 129, check
gradients/memory and exact restore, then verify its original observation batches
against step-128 weights on the second node. Correctness failures require recovery
from committed checkpoints; quality diagnostics do not stop the 32-step budget.

Evaluate on allocation 38226 / ip-10-3-70-177 at steps 128/144/160 using identical
development cohorts, GPT-4.1 judging, complete raw-action traces, and paired
bootstrap intervals. Keep WEB/EGO training/heldout and phase-sampled/unsampled
labels explicit. IT test and ESTP-Bench are excluded from these development
comparisons and remain in the original run's endpoint suite. No new untouched
test claim applies to reused development cohorts.

Local temporary files and compiler caches use the user-provided
`scripts/with_jevany_nvme_cache.sh` on each node. Models, checkpoints, logs,
manifests, judge caches, results and immutable snapshots remain on FSx.

After step 160 releases the training GPUs, recover the original run's unfinished
final evaluation on allocation 38177 with NVMe caches, preserving existing jobs,
shards and judgments. The new continuation's evaluations stay on allocation 38226.
Report training, new evaluation and prior endpoint recovery separately; do not
mark the combined workflow complete while any required evaluation is pending.
