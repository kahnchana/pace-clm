# Controller evaluation policy

For the current controller experiments, the user requires decisions to come
directly from the learned action scores. Do not add hand-chosen action-logit
offsets, response margins, temperature calibration, cooldowns, or heuristic
response suppression. Learn desired decision behavior through training.

Use response margin 0 for CLI compatibility; the evaluator must take the raw
learned argmax without modifying scores. Nonzero margins are rejected by
`pace_clm.evaluation_policy`. Prior margin-adjusted experiments are retired for
model selection and scaling decisions; retain their logs as audit records.

Run training and GPU evaluation for the current RL experiment series only on
`ip-10-3-120-167` (allocation 38177), the original node explicitly requested by
the user in this session, using its eight GPUs. GPT judging uses the user-authorized
gateway. The current full-WEB quality floor is 54.67 PAUC-F.

The user reiterated on 2026-11-09 that improvements must come from training.
Do not add inference-time score mixing, token/action logit changes, response
thresholds, cooldowns, or other runtime behavioral heuristics. Supervised replay
and regularization may affect learned weights during training only.
