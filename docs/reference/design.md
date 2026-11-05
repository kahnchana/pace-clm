# MVP design and source audit

## Source-of-truth decision

Inspection was performed against Stream-Gemma commit
`34a771e52e11fa3d339d216762db79485fd684a8`. The source worktree was dirty at
inspection time, so the pinned model and runtime contract was cross-checked
against the promoted artifact's immutable `INFERENCE.md`, `run_manifest.json`,
and `adapter_config.json`, not inferred only from the working tree.

The promoted PACE-Qwen checkpoint is the sole promoted Qwen wave-1 cell:

- Base: `Qwen/Qwen3.5-35B-A3B`
- Revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- Adapter: `stream-gemma/outputs/lora-qwen-w1-long-neg`
- Adapter SHA-256: `eb11e600de59edf48ef2d71bb632ae5944c0474441f264bb3ce6ceae8bb71da9`
- Final adapter step: 3260; LoRA r64, alpha128, dropout 0.05
- Runtime: 1 FPS, 32-frame causal rolling window, 384px, 144 tokens/frame
- Prefix: direct (the native empty-think suffix is removed)
- Memory: last-response, v2 instruction, k=1, aged-out selection
- Adapter scale: 1.0

Evidence:

- `stream-gemma/agent_notes/docs/promoted_runs.md`
- `stream-gemma/agent_notes/docs/experiments/promoted-qwen-c4-10p.md`
- `stream-gemma/outputs/lora-qwen-w1-long-neg/INFERENCE.md`
- `stream-gemma/outputs/lora-qwen-w1-long-neg/run_manifest.json`
- `stream-gemma/src/stream_gemma/inference/qwen_stream.py`
- `stream-gemma/src/stream_gemma/v24_memory.py`

## Integration boundary

`pace_clm.stream_gemma_adapter` validates the pinned artifact and imports
Stream-Gemma's timestamp and memory formatters. It mirrors only the causal
message topology needed to expose a feature-extraction decision point. Dataset
loading, WEB/EGO adapters, generation parsing, and PAUC-F remain in
Stream-Gemma and must be invoked there rather than copied.

The adapter never includes frames after the current decision time. It includes
at most 32 frames ending at the current frame, prior assistant emissions within
that window, and the configured last-response memory.

## Label semantics

Stream-Gemma emits `Silence`, `Standby`, and `Response`. For the paper-facing
CLM action space, `Standby` maps to `Hold`. Under the promoted
`event_interior` policy, `Standby` covers later annotated frames in an ongoing
event after its response onset. Those are relevant frames on which another
response would be redundant, matching `Hold`. This mapping is explicit and
exhaustively tested.

## Feature extraction contract

The extractor freezes the entire conditional-generation model (including its
PACE LoRA adapter), switches it to eval mode, and invokes the multimodal
backbone below the vocabulary LM head under `torch.inference_mode()`. It reads
`text_config.hidden_size`, then pools the final attended non-vision token.

For a decoder-only causal model, the selected language token is required to be
the final valid sequence position. Therefore it can attend to the complete
constructed prefix; a vision placeholder at the endpoint is rejected rather
than silently pooling the wrong token.

## Attribution

The causal message topology is adapted from Stream-Gemma. The independent
normalized state/action projections and learned temperature follow the
Apache-2.0 CLM project: https://github.com/Contrastive-LM/CLM

## JevAny streaming controller (2026-10-22)

`src/pace_clm/jevany_streaming.py` adds a candidate-conditioned pointer over
the promoted PACE decoder. The q/k dot-product readout follows JevAny commit
`b273aea682e5b748a2605808e52ccafb7a378924` (Apache-2.0). The local
reference checkout is under `third_party/JevAny`; no upstream patch was made.
J0 trains only this pointer. J1 trains the same pointer and the existing PACE
LoRA tensors. Neither run updates the frozen visual encoder.

Each decision preserves the native PACE system instruction, query, last-response
memory, timestamped one-frame user turns, and prior assistant emissions from
the sampled episode. Fixed action descriptions and a final decision marker are
appended to the current user turn. The pointer compares the final marker's
decoder state with states at the three option ends, all within the same causal
forward pass. The original PACE adapter remains the separate generator source;
the controller checkpoint is stored separately.

`scripts/prepare_jevany_decisions.py` invokes Stream-Gemma's promoted v25
episode builder with the long-negative settings. It exports all 26,080 episodes
and 263,760 supervised presentations, including trailing and post-response
negatives. The deterministic manifest hash is recorded in
`configs/jevany_streaming.yaml`. The existing balanced 27,000-decision head
manifest is not used for this controller study.

The 32-frame, 5,859-token real-model smoke requires the isolated
`fla-core==0.3.2` and `flash-linear-attention==0.3.2` runtime. Qwen's PyTorch
gated-delta fallback OOMs on a 144 GiB H200 for that case. The FLA kernel passed
forward and backward at 122.0 GiB peak allocated GPU memory. The shared
Stream-Gemma Python environment was not modified.
