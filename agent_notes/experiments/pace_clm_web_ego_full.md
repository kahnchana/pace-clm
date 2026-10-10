# Frozen run 5 MLP full WEB/EGO evaluation

Status: Full WEB and EGO generation, timing-only scoring, and GPT-4.1 judged scoring complete.

## Frozen setup

- Action checkpoint: `outputs/pace-clm-mixtures/checkpoints/run5.mlp.seed42.pt` (`sha256=307952585e4784a5aa571b88ae6866d4679e9173f313fa5e9b677da1cf5b0130`). The response margin is the checkpoint's dev-selected `1.183475375175476`.
- Backbone and response generator: promoted `outputs/lora-qwen-w1-long-neg` Qwen adapter on revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- Streaming protocol: 1 FPS, 32-frame causal window, direct Qwen prefix, last-response memory with instruction v2 and aged-out memory selection. Decisions use the frozen MLP on the final language-token backbone state. At Response decisions, the original Qwen generator fills response text; its output enters subsequent causal history. Greedy generation cap: 96 tokens.
- Full splits: WEB 500 videos at source 2 FPS, EGO 326 videos at source 6 FPS. No frame cap or split subsampling.
- Driver: `scripts/eval_web_ego_full.py`; merge validator: `scripts/merge_web_ego_full.py`. Outputs and frozen manifests: `outputs/pace-clm-web-ego-full/{web,ego}`.

## WEB local result

The eight-GPU rollout completed 500/500 videos with 925 physical fires (1.85/video), 61 silent videos, zero malformed emissions, and zero empty responses. The official scorer with all answer scores forced to maximum yielded timing-only PAUC-F (`pauc-f-v1`, `omega=0.5`) **63.57664990749798**, timing PAUC recall component **56.006252738970815**, and fire precision **73.51351351351352**. This is an upper bound, not a judged PAUC-F result and cannot be directly compared with the promoted Qwen judged WEB 57.47497434486915.

Artifacts: `outputs/pace-clm-web-ego-full/web/predictions.jsonl`, `outputs/pace-clm-web-ego-full/web/pauc_f_summary-timing-v1.json`, and `outputs/pace-clm-web-ego-full/web/pauc_batch_input-gpt-4.1.jsonl`. The GPT-4.1 judge input contains 839 requests, all completed.

## EGO local result

The eight-GPU rollout completed 326/326 videos with 1,310 physical fires (4.0184/video), zero malformed emissions, and zero empty responses. The official scorer with answer scores forced to maximum yielded timing-only PAUC-F (`pauc-f-v1`, `omega=0.5`) **48.79059622061979**, timing PAUC recall component **49.83805605889914**, and fire precision **47.786259541984734**. This is an upper bound, not a judged PAUC-F result and cannot be directly compared with the promoted Qwen judged EGO 46.05065873842904.

Artifacts: `outputs/pace-clm-web-ego-full/ego/predictions.jsonl`, `outputs/pace-clm-web-ego-full/ego/pauc_f_summary-timing-v1.json`, and `outputs/pace-clm-web-ego-full/ego/pauc_batch_input-gpt-4.1.jsonl`. The GPT-4.1 judge input contains 716 requests, all completed.

Both prediction files match the frozen manifest order and have unique IDs. Prediction SHA-256: WEB `8de5602e201922f0786d34ec0df5dd4b04c9b7b977e5f3e9c5dd84203b3e1fe6`; EGO `8438b56be088a9d5620f305c162310279d86e91800768ea762d5404b6f5a1d83`.

## GPT-4.1 judged result (2026-10-10)

After explicit user authorization, the configured Salesforce Research gateway
returned 839/839 WEB and 716/716 EGO judge responses, with unique request IDs
and no reported request errors. The requests contained question, reference, and
generated response text, but no video frames. The unchanged official PAUC
processor and `pauc-f-v1` scorer used `gpt-4.1` and `omega=0.5`.

| Split | Candidate judged PAUC-F | Promoted PACE Qwen judged PAUC-F | Difference | Candidate judged recall component | Candidate judged fire precision | Candidate value-adding / wasted fires |
|---|---:|---:|---:|---:|---:|---:|
| WEB | 53.4984 | 57.4750 | -3.9766 | 43.0282 | 70.7027% | 654 / 271 |
| EGO | 43.9530 | 46.0507 | -2.0977 | 41.6185 | 46.5649% | 610 / 700 |

The frozen run 5 MLP fired less often than the promoted PACE token decision
(WEB 925 versus 1,132; EGO 1,310 versus 1,634), with nearly unchanged judged
fire precision. It covered fewer value-adding turns, reducing judged recall and
PAUC-F on both splits. The candidate does not beat the promoted PACE baseline
on the judged headline metric.

Judged artifacts are `pauc_batch_output-gpt-4.1.jsonl`,
`pauc_llm_scores-gpt-4.1.json`, `pauc_summary-gpt-4.1.json`, and
`pauc_f_summary-gpt-4.1-v1.json` in each split directory. The baseline WEB/EGO
raw outputs are absent from the local Stream-Gemma output tree, so its reported
point estimates can be compared at the same benchmark, judge, scorer version,
and omega, but a paired video bootstrap or baseline rescoring is unavailable.
