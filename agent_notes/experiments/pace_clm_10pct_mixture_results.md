# PACE-CLM 10% data-mixture results

The six-condition comparison used the fresh 200-video mixture-selection set. Values are mean ± sample SD across seeds 42/43/44; PAUC-F and F1 values are percentages.

| Condition | Head | PAUC-F | Macro F1 | Silence F1 | Hold F1 | Response F1 | NLL | ECE | Timing F1@4s | Fires/video | Repeats/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | mlp | 80.59 ± 1.26 | 80.63 ± 1.20 | 98.39 ± 0.11 | 67.67 ± 2.48 | 75.84 ± 1.14 | 0.11 ± 0.01 | 0.01 ± 0.00 | 76.78 ± 1.35 | 3.29 ± 0.04 | 0.01 ± 0.00 |
| control | clm_short_256 | 79.75 ± 0.81 | 79.90 ± 0.59 | 98.28 ± 0.07 | 66.21 ± 1.49 | 75.20 ± 0.77 | 0.12 ± 0.01 | 0.01 ± 0.00 | 75.95 ± 0.70 | 3.33 ± 0.07 | 0.01 ± 0.00 |
| run1 | mlp | 81.57 ± 1.41 | 85.93 ± 0.74 | 98.54 ± 0.09 | 81.23 ± 1.09 | 78.01 ± 1.08 | 0.09 ± 0.01 | 0.01 ± 0.00 | 79.57 ± 1.49 | 3.68 ± 0.15 | 0.03 ± 0.00 |
| run1 | clm_short_256 | 81.49 ± 0.30 | 85.87 ± 0.26 | 98.56 ± 0.01 | 81.09 ± 0.60 | 77.94 ± 0.37 | 0.09 ± 0.00 | 0.01 ± 0.00 | 79.37 ± 0.39 | 3.68 ± 0.04 | 0.02 ± 0.00 |
| run2 | mlp | 82.03 ± 1.11 | 86.91 ± 0.53 | 98.50 ± 0.10 | 83.48 ± 1.96 | 78.75 ± 0.74 | 0.09 ± 0.01 | 0.01 ± 0.00 | 80.57 ± 0.89 | 3.93 ± 0.52 | 0.05 ± 0.04 |
| run2 | clm_short_256 | 82.42 ± 1.32 | 87.20 ± 0.11 | 98.55 ± 0.13 | 84.11 ± 0.56 | 78.95 ± 0.57 | 0.08 ± 0.01 | 0.01 ± 0.00 | 80.63 ± 0.67 | 3.74 ± 0.44 | 0.03 ± 0.00 |
| run3 | mlp | 81.77 ± 0.59 | 86.98 ± 0.18 | 98.55 ± 0.03 | 84.60 ± 0.90 | 77.81 ± 0.36 | 0.08 ± 0.00 | 0.00 ± 0.00 | 79.72 ± 0.83 | 3.50 ± 0.04 | 0.03 ± 0.01 |
| run3 | clm_short_256 | 81.61 ± 0.34 | 86.24 ± 0.28 | 98.55 ± 0.06 | 82.64 ± 0.84 | 77.52 ± 0.05 | 0.08 ± 0.00 | 0.00 ± 0.00 | 79.11 ± 0.23 | 3.32 ± 0.18 | 0.03 ± 0.01 |
| run4 | mlp | 81.86 ± 1.14 | 87.02 ± 0.53 | 98.55 ± 0.08 | 84.30 ± 0.77 | 78.22 ± 1.03 | 0.09 ± 0.01 | 0.01 ± 0.00 | 80.07 ± 1.13 | 3.57 ± 0.08 | 0.04 ± 0.00 |
| run4 | clm_short_256 | 80.55 ± 0.78 | 86.45 ± 0.60 | 98.42 ± 0.12 | 83.35 ± 1.97 | 77.58 ± 0.63 | 0.09 ± 0.01 | 0.00 ± 0.00 | 79.19 ± 0.52 | 4.05 ± 0.24 | 0.04 ± 0.02 |
| run5 | mlp | 83.25 ± 0.49 | 87.84 ± 0.88 | 98.62 ± 0.03 | 85.29 ± 2.02 | 79.60 ± 0.67 | 0.08 ± 0.00 | 0.01 ± 0.00 | 81.48 ± 0.86 | 3.63 ± 0.27 | 0.04 ± 0.01 |
| run5 | clm_short_256 | 82.71 ± 0.71 | 87.07 ± 1.32 | 98.61 ± 0.07 | 83.25 ± 3.11 | 79.34 ± 0.80 | 0.09 ± 0.01 | 0.01 ± 0.00 | 80.98 ± 0.93 | 3.70 ± 0.17 | 0.04 ± 0.01 |

Selected mixture: **run5**. Mean PAUC-F beat the balanced control: **True**.

## Manifest counts

| Condition | Decisions | Silence | Hold | Response | Unique videos | Repeated positions |
|---|---:|---:|---:|---:|---:|---:|
| run3 | 24300 | 18346 | 778 | 5176 | 7252 | 0 |
| run4 | 24300 | 19853 | 656 | 3791 | 7401 | 0 |
| run5 | 24300 | 19853 | 656 | 3791 | 7431 | 0 |

Weighted-loss effective sample size: run 1 **11695**, run 2 **9579** out of 24,300 selected decisions. Run 5 replaced **1,985** Silence decisions with long response-free Silence; all three mask manifests have zero repeated positions.

## PACE-like exposure on the untouched final set

The selected and balanced regimes used the same selected positions, 263,760 decision presentations, and 3,260 optimizer updates per model/seed.

| Condition | Head | PAUC-F | Macro F1 | Silence F1 | Hold F1 | Response F1 | NLL | ECE | Timing F1@4s | Fires/video | Repeats/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| selected | mlp | 77.09 ± 1.35 | 85.28 ± 0.64 | 98.16 ± 0.14 | 84.71 ± 0.69 | 72.96 ± 1.12 | 0.13 ± 0.02 | 0.01 ± 0.01 | 74.57 ± 1.20 | 3.53 ± 0.30 | 0.02 ± 0.01 |
| selected | clm_short_256 | 77.82 ± 1.46 | 85.62 ± 0.27 | 98.26 ± 0.15 | 85.27 ± 1.12 | 73.32 ± 1.33 | 0.11 ± 0.02 | 0.01 ± 0.01 | 75.32 ± 1.83 | 3.45 ± 0.23 | 0.06 ± 0.05 |
| balanced | mlp | 77.09 ± 1.35 | 85.28 ± 0.64 | 98.16 ± 0.14 | 84.71 ± 0.69 | 72.96 ± 1.12 | 0.13 ± 0.02 | 0.01 ± 0.01 | 74.57 ± 1.20 | 3.53 ± 0.30 | 0.02 ± 0.01 |
| balanced | clm_short_256 | 77.82 ± 1.46 | 85.62 ± 0.27 | 98.26 ± 0.15 | 85.27 ± 1.12 | 73.32 ± 1.33 | 0.11 ± 0.02 | 0.01 ± 0.01 | 75.32 ± 1.83 | 3.45 ± 0.23 | 0.06 ± 0.05 |

Because run 5 uses unweighted loss, selected and balanced-loss scale-up runs are the same recipe on the same positions and give identical scores.

## Frozen 24.3k references on the same final set

These checkpoints were trained before the final set was examined and were scored only to put the scale-up result on the same held-out videos.

| Condition | Head | PAUC-F | Macro F1 | Silence F1 | Hold F1 | Response F1 | NLL | ECE | Timing F1@4s | Fires/video | Repeats/video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | mlp | 79.59 ± 0.51 | 79.90 ± 0.89 | 98.24 ± 0.07 | 67.24 ± 2.10 | 74.22 ± 0.60 | 0.12 ± 0.01 | 0.01 ± 0.00 | 75.14 ± 0.36 | 2.90 ± 0.06 | 0.00 ± 0.00 |
| control | clm_short_256 | 78.90 ± 1.15 | 79.03 ± 1.26 | 98.11 ± 0.13 | 65.45 ± 2.65 | 73.52 ± 1.21 | 0.13 ± 0.01 | 0.01 ± 0.00 | 74.44 ± 0.94 | 2.97 ± 0.08 | 0.00 ± 0.00 |
| run5 | mlp | 80.11 ± 0.59 | 86.37 ± 0.37 | 98.44 ± 0.07 | 85.06 ± 1.12 | 75.60 ± 0.32 | 0.09 ± 0.00 | 0.01 ± 0.00 | 77.02 ± 0.47 | 3.20 ± 0.28 | 0.02 ± 0.01 |
| run5 | clm_short_256 | 79.26 ± 1.26 | 85.38 ± 1.83 | 98.34 ± 0.15 | 82.79 ± 4.02 | 75.02 ± 1.33 | 0.10 ± 0.01 | 0.01 ± 0.00 | 76.33 ± 1.34 | 3.32 ± 0.19 | 0.01 ± 0.01 |

Across both heads and three seeds, 24.3k run 5 scored 79.69 PAUC-F versus 79.24 for the original balanced control; scale-up scored 77.45. Extra exposure changed PAUC-F by -2.24 points on this final set.

## Selection tradeoffs and extraction note

On the selection set, the selected mixture changed fires/video from 3.31 to 3.66, mean repeat fires per 200 videos from 1.33 to 8.00, and Hold recall by -10.33 percentage points. It therefore did not meet the plan's no-extra-fires/repeats and per-state-recall preferences.

The raw training-union extraction differed materially from the prior cache at 9 of 3,185 overlapping positions because OpenCV seek placement changed with denser frame requests. The raw cache is preserved, and the working union cache reuses the prior frozen vector at every overlapping position; see feature_overlap.json.

Detailed per-seed metrics, PAUC-F components, per-video scores, and timing/offset breakdowns are in the JSON artifacts beside this report.
