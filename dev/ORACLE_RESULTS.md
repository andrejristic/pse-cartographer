# Runtime verification on real models (2026-10-12, i3-2330M, no AVX2)

Our numpy runtime vs transformers 5.19 / torch 2.14 (CPU), Q8_0 GGUFs.
`dev/oracle_forward.py <model.gguf>`; finite differences in float64 with the runtime itself.

| Model | Tokens vs HF | Logits max diff (scale) | Resid rel diff | VJP vs torch | VJP vs finite diff (fp64) | fp32 vs fp64 VJP |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (qwen2) | 6/6 equal | 2.3e-4 (17.7) | ≤1.2e-5 | ≤2.5e-5 | – | ≤1.2e-5 |
| Qwen3-0.6B (qwen3) | 6/6 equal | 3.1e-5 (17.1) | ≤2.4e-7 | ≤1.8e-5 | – | – |
| TinyLlama-1.1B (llama, SentencePiece) | equal except BOS* | 2.2e-5 (21.5) | ≤5.7e-8 | 0.19–0.37 ** | 7 digits, all positions | ≤2.2e-6 |
| Llama-3.2-1B (llama, rope_freqs) | 6/6 equal | 3.3e-2 (20.7) *** | ≤2.9e-5 | 2.5e-3–3.2e-2 ** | – | ≤3.6e-6 |

\* transformers' GGUF loader does not prepend `<s>` for this file; the GGUF says
`add_bos_token` and Llama-2-style models are trained with it, so we prepend it.
\** The forward pass is identical to transformers, and our VJP equals float64 finite
differences of that same function to 7 digits, so the gap is in how the torch-side
gradient is taken for transformers' Llama layers, not in our backward. Finite
differences are therefore used as the second, independent judge.
\*** transformers recomputes Llama-3 rope scaling from config; llama.cpp (and we) use
the `rope_freqs.weight` factors stored in the GGUF. Argmax agreement is 100%.

Speed (Qwen2.5-0.5B, 10 tokens): forward 0.7 s; VJP with 64 probes over all 24 layers 12.4 s.
