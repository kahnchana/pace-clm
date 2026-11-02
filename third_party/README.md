# JevAny reference

The local `JevAny/` checkout is pinned to
`b273aea682e5b748a2605808e52ccafb7a378924` from
<https://github.com/SimpleJev/JevAny>. Its Apache-2.0 license is retained in
`JevAny/LICENSE`. The controller in `src/pace_clm/jevany_streaming.py` adapts
JevAny's `PointerHead` q/k dot-product readout to Stream-Gemma's native PACE
frame, timestamp, and response-history topology. No JevAny source patch was
applied.

The isolated FLA runtime uses `fla-core==0.3.2` and
`flash-linear-attention==0.3.2` in `outputs/jevany/deps032`; the shared
Stream-Gemma environment was not modified.
