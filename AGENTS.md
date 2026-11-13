# Controller evaluation policy

For the current controller experiments, the user requires decisions to come
directly from the learned action scores. Do not add hand-chosen action-logit
offsets, response margins, temperature calibration, cooldowns, or heuristic
response suppression. Learn desired decision behavior through training.

Use response margin 0 for CLI compatibility; the evaluator must take the raw
learned argmax without modifying scores. Nonzero margins are rejected by
`pace_clm.evaluation_policy`. Prior margin-adjusted experiments are retired for
model selection and scaling decisions; retain their logs as audit records.

Run training for the current RL experiment series only on `ip-10-3-120-167`
(allocation 38177), using its eight GPUs. On 2026-11-11 the user additionally
authorized the assigned eight-GPU node `ip-10-3-70-177` (allocation 38226) for
asynchronous standalone evaluation. Keep learner correctness checks on the
training node; do not train on the evaluation node or allocate other nodes.
On 2026-11-12 the user requested the mixed-run final evaluation on the original
training node, allocation 38177, after training releases its eight GPUs. Keep
earlier evaluation results from allocation 38226; route only the final evaluation
to the training node and wait for training to finish before acquiring its GPUs.
GPT judging uses the user-authorized gateway. The historical full-WEB quality
floor is 54.67 PAUC-F; mixed WEB/EGO training results must distinguish seen
examples from examples held out of the new RL run.

The user reiterated on 2026-11-09 that improvements must come from training.
Do not add inference-time score mixing, token/action logit changes, response
thresholds, cooldowns, or other runtime behavioral heuristics. Supervised replay
and regularization may affect learned weights during training only.
