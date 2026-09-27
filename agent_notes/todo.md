# PACE-CLM MVP: implementation handoff

## Objective

Build a clean proof of concept that adds CLM-style contrastive state/action projection heads to the existing PACE-Qwen VLM. The frozen PACE VLM should act as a multimodal feature extractor and retain its existing response-generation capability.

The initial action space is:

1. `Response` — sufficient new evidence is available; invoke the existing PACE generation path to produce the natural-language response.
2. `Silence` — no currently relevant event or evidence warrants a response.
3. `Hold` — relevant evidence/event is present, but another response is not warranted because it would be redundant.

This is an MVP and an evaluation of the method, not a rewrite of Stream-Gemma.

## Source repository and new repository

- Existing reference repository: `/fsx/home/kranasinghe/repo/stream-gemma`
- Create a separate repository at: `/fsx/home/kranasinghe/repo/pace-clm`
- Treat `stream-gemma` as the source of truth for:
  - PACE-Qwen model/checkpoint loading
  - causal streaming input construction
  - PACE state labels
  - WEB/EGO adapters and scoring
  - PAUC-F implementation
- Do not copy or simplify evaluation code in a way that changes benchmark behavior. Prefer a small adapter/import boundary or clearly attributed, tested code copied from Stream-Gemma.
- The CLM repository is Apache-2.0: <https://github.com/Contrastive-LM/CLM>. Preserve attribution for adapted components.

## Initial scope

Implement PACE-Qwen first. Do not initially support every PACE backbone.

Reference model currently used by Stream-Gemma:

```text
Qwen/Qwen3.5-35B-A3B
```

Use the existing promoted PACE-Qwen checkpoint/configuration used by the current evaluation harness. Do not silently substitute base Qwen or a different PACE checkpoint.

Freeze the entire VLM and its existing PACE adapter. Only the following components should be trainable:

- state projection head
- action projection head
- scalar logit temperature/scale

## Proposed architecture

For each streaming state at time `t`, construct exactly the same causal multimodal input used by the PACE evaluation harness:

- system/task instruction
- user query
- causal rolling video window ending at `t`
- configured interaction/last-response memory
- no future frames

Run a normal frozen VLM forward pass and extract the final language-position hidden state after the final transformer block:

```text
h_state: [batch, hidden_size]
```

Important:

- Read `hidden_size` from the loaded model configuration. Do not hardcode 4096.
- Select the final valid language token, not a padded position or vision placeholder.
- Confirm that the selected position can causally attend to the full constructed input.
- Do not generate a token or apply the vocabulary LM head when extracting the state feature.
- Run extraction under inference/no-grad mode with the frozen model in evaluation mode.

Encode each action using the same frozen Qwen language tower in text-only mode. Start with fixed descriptive action strings rather than only the three label words. Put these strings in a versioned config file and record their exact values in every run artifact. Initial wording can be:

```text
Response: There is sufficient new visual evidence to answer the user's request now.
Silence: There is no relevant visual evidence that warrants a response now.
Hold: Relevant evidence is present, but no new response should be made because it would repeat an ongoing or previous response.
```

Extract one final-token feature for each action:

```text
h_action: [3, hidden_size]
```

Use independent projection heads with the same architecture but different parameters:

```text
z_state  = L2Norm(state_head(L2Norm(h_state)))
z_action = L2Norm(action_head(L2Norm(h_action)))
logits   = exp(logit_scale) * z_state @ z_action.T
```

Initial projection-head configuration:

- input: model `hidden_size`
- hidden width: 1536
- depth: 3 linear layers total
- activation: GELU
- LayerNorm: enabled
- output dimension: 512
- no residual connection initially

Make all dimensions configurable. Cache the three frozen base action embeddings. The projected action embeddings must remain live during training so gradients reach `action_head`.

## Labels and data

Use the existing PACE state targets, without relabeling from predictions:

- `Silence`: no relevant event
- `Response`: event onset with sufficient evidence
- `Hold`: subsequent annotated frames from the same ongoing event, where another response would be redundant

Note that Stream-Gemma may internally use `Standby` in parts of the episode-building code. Resolve and document the exact mapping to the paper-facing `Hold` label. Add a test proving the mapping for all three states.

Projection-head training data must come from the PACE training split. Do not train on the 200-sample PACE validation set.

Start with a small, deterministic training subset sufficient to establish that the model can learn. The sampler must:

- avoid overwhelming the loss with `Silence`
- retain meaningful `Hold` examples
- preserve video-level split boundaries
- record the selected video/example IDs and random seed

Use class-balanced sampling or explicit class weights, and report the natural and sampled class distributions.

## Training objective

Implement the CLM-style contrastive objective over state/action pairs. At minimum:

- positive pair: state and its gold action
- negatives: the other two action embeddings
- learned positive logit scale initialized to a reasonable CLIP-style value
- cross-entropy over the three action logits

If implementing the full bidirectional CLM/InfoNCE loss, ensure repeated action labels within a batch are not incorrectly treated as negatives. A simple three-candidate state-to-action cross-entropy objective is the safer MVP. Clearly distinguish this from full bidirectional CLM in documentation.

## Inference behavior

At every streaming decision point:

1. Extract the frozen PACE-Qwen state feature.
2. Project it with `state_head`.
3. Score it against the three projected action embeddings.
4. Choose an action using the frozen selection rule.
5. If the action is `Response`, invoke the existing PACE response-generation path.
6. If the action is `Silence` or `Hold`, emit the corresponding non-response state and do not generate answer text.

For the MVP, a second VLM pass for response generation is acceptable. Measure and report the overhead. Do not attempt KV-cache/hidden-state reuse until correctness is established.

## Required baselines

Run all methods on identical examples with identical streaming inputs:

1. Existing PACE token-generation state decision.
2. CLM-style separate state/action projection heads.
3. Standard three-way linear classifier over the same frozen state feature.
4. Standard three-way MLP classifier with capacity comparable to the projection-head system.
5. Raw cosine similarity between unprojected state and action embeddings.

The classifier baselines are mandatory: they determine whether contrastive state/action alignment provides value beyond ordinary classification.

Useful ablations after the MVP works:

- separate versus shared projection heads
- short action label versus descriptive action text
- final-token pooling versus mean language-token pooling
- base Qwen versus PACE-Qwen features
- 128/256/512 projection dimension

Do not run the ablations on WEB/EGO during method development.

## Evaluation protocol

### Phase 1: correctness smoke tests

- Run a few hand-inspected PACE episodes.
- Confirm strictly causal frame access.
- Confirm final-token selection and tensor shapes.
- Confirm all VLM parameters have `requires_grad=False` and receive no gradients.
- Confirm both projection heads and the logit scale receive gradients.
- Confirm cached and uncached action embeddings are numerically identical.
- Overfit a tiny balanced set. Failure to overfit is a blocking implementation bug.

### Phase 2: PACE validation

Evaluate on the fixed 200-sample PACE validation subset. This is the only set used to choose:

- action descriptions
- projection architecture
- learning rate and regularization
- temperature initialization
- class weighting/sampling
- any confidence threshold or temporal smoothing

The 200 examples are evaluation/development examples only. Do not update model parameters on them.

Report:

- overall accuracy
- macro-F1
- per-state precision, recall, and F1
- 3x3 confusion matrix
- calibration/NLL if probabilities are used
- PAUC-F through the unchanged PACE harness
- fires per video
- repeated-response rate
- response latency/timeliness
- generation quality for selected `Response` events

### Phase 3: frozen WEB/EGO pilot

After the method and all hyperparameters are frozen, run exactly:

- 50 fixed WEB samples
- 50 fixed EGO samples

Create deterministic, versioned manifests for both sets before running. Prefer stratification by event/response availability and video duration if metadata permits. Use the same manifests for every baseline.

Treat this as a pilot, not final paper evidence. Do not inspect individual failures and then alter the method while continuing to report the same pilot as an untouched test. Any post-pilot change requires a new clearly identified development cycle and a fresh held-out test protocol.

Report PAUC-F plus its components and behavioral diagnostics:

- PAUC-F
- underlying PAUC/recall term
- firing precision
- fires per video
- duplicate/repeated fires
- out-of-window fires
- response latency
- state/action distribution

If the 50/50 pilot is promising, run the full WEB and EGO evaluations and use bootstrap confidence intervals or repeated fixed-seed samples for claims.

## Reproducibility and artifacts

Every run must save:

- git commit and dirty-state indicator for both repositories
- full resolved config
- model and adapter identifiers/checkpoint hashes
- tokenizer and processor identifiers
- exact action strings
- selected pooling position/method
- train/dev/test manifests
- random seeds
- class distributions
- checkpoint containing both heads and logit scale
- per-example logits, probabilities, gold state, selected state, timestamp, and video ID
- benchmark-compatible prediction files
- summary metrics and confusion matrices

## Suggested repository layout

```text
pace-clm/
  README.md
  pyproject.toml
  configs/
    qwen_mvp.yaml
    actions_v1.yaml
  src/pace_clm/
    backbone.py
    pooling.py
    heads.py
    labels.py
    datasets.py
    train.py
    infer.py
    artifacts.py
    stream_gemma_adapter.py
  scripts/
    extract_features.py
    train_heads.py
    eval_pace200.py
    eval_web_ego_pilot.py
  tests/
    test_label_mapping.py
    test_pooling_position.py
    test_frozen_backbone.py
    test_head_gradients.py
    test_causal_inputs.py
  manifests/
  docs/
    design.md
    experiment_log.md
```

## Execution order

1. Inspect the promoted PACE-Qwen loading and inference path in Stream-Gemma; document exact checkpoint/config choices.
2. Scaffold `pace-clm` with tests and configuration management.
3. Implement a state/action feature extractor and verify final-token selection.
4. Implement projection heads, loss, checkpointing, and action caching.
5. Build the PACE training adapter and balanced deterministic subset.
6. Pass tiny-set overfitting and gradient/freeze tests.
7. Train the MVP heads.
8. Implement all required baselines.
9. Evaluate on the fixed PACE-200 development set and freeze the design.
10. Create versioned WEB-50 and EGO-50 manifests.
11. Run the frozen pilot once, report results, and recommend whether full evaluation is justified.

## MVP completion criteria

The MVP is complete only when:

- PACE-Qwen is demonstrably frozen.
- State and action features come from the same PACE-Qwen language tower.
- The final valid language-token feature is selected correctly for multimodal and text-only inputs.
- Separate state/action heads train and the tiny-set overfit test passes.
- All five required methods run through one common evaluation interface.
- The PACE-200 results include state metrics and PAUC-F behavioral metrics.
- WEB-50/EGO-50 are not run until the configuration is frozen.
- All manifests, predictions, configs, and checkpoints are reproducible.

## Important non-goals for the first pass

- Do not fine-tune the VLM.
- Do not support multiple backbones before PACE-Qwen works.
- Do not optimize KV-cache reuse yet.
- Do not tune directly on WEB or EGO.
- Do not claim that a 50/50 pilot is statistically sufficient for a paper.
- Do not change PAUC-F or benchmark scoring behavior.
