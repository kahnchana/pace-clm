# Four-source balanced RLOO continuation, steps 129–160

Continued model, optimizer, seven learner RNG states, sampler cursor and supervised replay from step 128.
Each update: 2 original + 2 WEB + 2 native EGO + 2 ESTP-IT episodes; four trajectories per episode.
Each source has objective weight 0.25; decision loss retains constant-32 normalization.
Raw learned argmax, response margin zero; no inference-time score modifications.

Successful update time: 13.45 node-hours / 107.57 allocated GPU-hours; loading, gates, retries and evaluation excluded.
64 native EGO episodes reused from parent training. Other phase episodes are fresh; heldouts remain excluded.
Evaluation cohorts are reused development cohorts; full WEB/EGO include training examples.
IT dev uses the PAUC-F proxy. IT test and ESTP-Bench remain in the original parent endpoint suite.

| Cohort | Step 128 | Step 144 | Step 160 | Change 128→160 |
|---|---:|---:|---:|---:|
| web_train | 58.126 | 58.287 | 58.512 | +0.386 |
| web_heldout | 57.325 | 57.708 | 58.032 | +0.706 |
| ego_train | 44.870 | 44.994 | 45.715 | +0.846 |
| ego_heldout | 43.160 | 43.519 | 43.176 | +0.015 |
| ours_heldout | 69.695 | 70.164 | 70.197 | +0.503 |
| ours_train | 74.648 | 74.715 | 75.017 | +0.369 |
| perfect20 | 97.292 | 97.292 | 97.292 | +0.000 |
| web_all | 57.974 | 58.182 | 58.426 | +0.452 |
| web_sampled | 57.971 | 58.125 | 58.565 | +0.594 |
| web_unsampled | 58.703 | 58.884 | 58.304 | -0.399 |
| ego_all | 43.850 | 44.114 | 44.223 | +0.373 |
| ego_sampled | 44.870 | 44.994 | 45.715 | +0.846 |
| estp_it_dev | 10.220 | 10.413 | 10.730 | +0.510 |
| web_phase_sampled | 60.601 | 59.976 | 60.718 | +0.117 |
| web_phase_unsampled | 57.713 | 58.003 | 58.138 | +0.425 |
| ego_phase_sampled | 47.124 | 46.456 | 47.604 | +0.481 |
| ego_phase_unsampled | 42.994 | 43.792 | 44.109 | +1.114 |

Paired bootstrap intervals, useful/wasted fires, precision and recall are in each evaluation result.json.
This one-arm continuation changes both episode mix and configured source weights; it does not isolate them or compare trajectory counts.
