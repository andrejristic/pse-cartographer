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

# Phase 2 families (0.4.0)

Tokenizers vs each model's original `tokenizer.json` (`dev/oracle_tok.py`, 411 texts: GPL text,
digits, emoji, CJK, Cyrillic, odd whitespace): **411/411 identical** for StableLM 2, OLMo 2,
Granite 3.3, Phi-2, Phi-3.5, Granite 3.1 MoE (1B, 3B), OLMoE. Fixed on the way: StableLM 2 split
rule when the GGUF names none, Granite's "refact" label (its tokenizer.json is plain GPT-2), and
Phi-2's whitespace-run added tokens keeping their own ids. Gemma uses transformers' GGUF loader
as reference.

Forward vs transformers (`dev/oracle2.py`, two stages, 9 GB memory cap; bf16 where fp32 does
not fit twice in 16 GB):

| Model | Reference | Result |
|---|---|---|
| Gemma 3 270M | GGUF loader, fp32 | logits 6e-5 (35), VJP vs torch ≤6e-6 |
| StableLM 2 1.6B | GGUF loader, fp32 | logits 6e-5 (25), argmax 100% |
| OLMo 2 1B | weights mapped, fp32 | logits 4e-5 (18), argmax 100% |
| Granite 3.3 2B | weights mapped, bf16 | argmax 100%, resid ≤5% (bf16) |
| Phi-2 | weights mapped, bf16 | argmax 100%, last layer 11% (Phi-2's huge activations in bf16) |
| Gemma 2 2B | GGUF loader, bf16 | argmax 100%, resid ≤0.4% |
| Phi-3.5-mini | weights mapped + original longrope config, bf16 | positions 1–14: corr 0.9998, argmax 100%; position 0 (first token, attention sink) differs in bf16 |

transformers' GGUF loader ignores Phi-3.5's longrope factors (default RoPE), so the original
config was used. OLMo 2's RoPE type was decided by perplexity (neox 16 vs norm 191) and then
confirmed against transformers; Granite's likewise (norm 5.1 vs neox 97).

Every family also has a tiny fixture in the selftest with all its switches on; the hand-written
VJP matches float64 finite differences on each, and deliberate bugs in each new backward piece
(LayerNorm, GELU, softcap, post-norm, full-vector QK norm, MoE router) are caught.

# Mixture of experts (0.4.0)

| Model | Reference | Result |
|---|---|---|
| Granite 3.1 MoE 1B-A400M (32 experts, top-8) | weights mapped, fp32 | logits 3e-5 (28), argmax 100%, resid 1e-8; VJP vs FD (frozen routing) 3e-3 |
| Granite 3.1 MoE 3B-A800M (40 experts, top-8) | weights mapped, bf16 | argmax 100%, resid ≤1%; VJP vs FD 4e-3 |
| OLMoE 1B-7B (64 experts, top-8) | 28 GB in fp32: no transformers run | RoPE neox (perplexity 6.9 vs 303); VJP vs FD 3e-3; greedy "Paris." |

Top-k routing is a step function: a finite difference that crosses a routing switch measures the
step, not the gradient. The FD checks therefore freeze the experts chosen in the base pass
(`NumpyRuntime.force_routing`); the router weights stay smooth and are checked. The selftest
fixtures (tiny-granitemoe with a shared expert, tiny-olmoe without renormalisation) do the same
in float64, and deliberate bugs in the router backward are caught.

First real map: Granite MoE 1B, 8 prompts, 24 layers, 5.5 min on the i3-2330M; routing drift
0.66 on average, correlating with concept drift at 0.59.
