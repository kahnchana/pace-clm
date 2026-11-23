# Mixed WEB/EGO/ESTP-IT episode RLOO through step 128

Training completed all 128 optimizer steps, including 96 updates continued from step 32.
Evaluation scores did not control stopping. Inference used raw learned argmax with response margin zero.

## Exposure

Additional episode counts: `{'ours': 384, 'web': 192, 'ego': 81, 'estp_it': 111}`.
Additional decisions: 381,336.
Successful continuation-update time: 36.75 hours / 293.98 GPU-hours; excludes loading, retries, idle time and evaluation.

## Final PAUC-F

| Cohort | PAUC-F | Useful | Wasted |
|---|---:|---:|---:|
| web_train | 58.128 | 694 | 288 |
| web_heldout | 57.343 | 169 | 64 |
| ego_train | 44.870 | 216 | 174 |
| ego_heldout | 43.157 | 300 | 313 |
| ours_heldout | 69.695 | 314 | 141 |
| ours_train | 74.648 | 441 | 153 |
| perfect20 | 97.292 | 77 | 3 |
| web_all | 57.979 | 863 | 352 |
| web_sampled | 57.424 | 442 | 197 |
| web_unsampled | 59.412 | 252 | 91 |
| ego_all | 43.849 | 516 | 487 |
| ego_sampled | 44.870 | 216 | 174 |
| estp_it_dev | 10.220 | 33 | 492 |
| estp_it_test | 11.352 | 45 | 590 |

## Official ESTP-F1

ESTP-Bench and the fixed IT test probe use the official scorer and its separate answer-quality rubric.
Full endpoint artifacts: [mixed-web-ego-estp-episode-balanced-v3-to128](/fsx/home/kranasinghe/repo/pace-clm/outputs/jevany-2b-rl/mixed-web-ego-estp-episode-balanced-v3-to128).

Existing full WEB/EGO cohorts include training identities. Existing heldouts are reused historical evaluations.
IT splits exclude source-video overlap within this phase; eligibility filtering and selected task types are recorded in data/audit.json.
Training uses converted IT PAUC-F rewards; official ESTP-F1 is a separate endpoint metric.

- estp_bench_official_estp_f1: ESTP-F1 x100 = 31.196070151629762.
- estp_it_test_official_estp_f1: ESTP-F1 x100 = 17.528553980711216.
