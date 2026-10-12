"""Build the tiny random-weight GGUF fixtures that ship with the selftest.

Each fixture is a real GGUF of one supported architecture, with every switch
that architecture uses turned on (QKV bias, QK-norm, head_dim != d/n_head,
rope_freqs, untied W_U, sliding window, SentencePiece vocab, a Q8_0 tensor...),
but a few hundred KB in size. The selftest checks the hand-written VJP against
finite differences on each of them, so a backward bug in any switch fails CI on
both Linux and Windows.

Dev-only (needs the `gguf` package from the dev venv):
    .dev/venv/bin/python dev/make_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter, TokenType
from gguf.quants import quantize

OUT = Path(__file__).resolve().parents[1] / "app" / "pse_a" / "tests" / "fixtures"
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def bpe_vocab():
    toks = bytes_to_unicode()
    merges = ["Ġ t", "h e", "Ġt he", "i n", "Ġ a", "e r", "o n", "Ġ s", "a n", "Ġ w"]
    toks += ["Ġt", "he", "Ġthe", "in", "Ġa", "er", "on", "Ġs", "an", "Ġw"]
    types = [TokenType.NORMAL] * len(toks)
    for sp in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        toks.append(sp)
        types.append(TokenType.CONTROL)
    return toks, types, merges


def spm_vocab():
    toks = ["<unk>", "<s>", "</s>"]
    types = [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL]
    scores = [0.0, 0.0, 0.0]
    for b in range(256):
        toks.append(f"<0x{b:02X}>")
        types.append(TokenType.BYTE)
        scores.append(0.0)
    pieces = ["▁"] + list("abcdefghijklmnopqrstuvwxyz") + ["▁t", "he", "▁the", "in", "▁a", "er"]
    for i, p in enumerate(pieces):
        toks.append(p)
        types.append(TokenType.NORMAL)
        scores.append(-float(i))
    return toks, types, scores


def build(name: str, arch: str, *, d=64, n_head=4, n_kv=2, head_dim=16, ff=96, layers=3,
          qkv_bias=False, qk_norm="", tied=True, rope_freqs=False, sliding=None,
          spm=False, q8=False, seed=0, norm="rms", post_norm=False, pre_norm=True,
          parallel=False, mlp="gated", fused_qkv=False, fused_gate_up=False, all_bias=False,
          rope_dim=None, extra=None, out_bias=False, moe=None, shexp=False):
    rng = np.random.default_rng(seed)
    w = GGUFWriter(str(OUT / f"{name}.gguf"), arch)
    if spm:
        toks, types, scores = spm_vocab()
        w.add_tokenizer_model("llama")
        w.add_token_list(toks)
        w.add_token_types(types)
        w.add_token_scores(scores)
        w.add_bos_token_id(1)
        w.add_eos_token_id(2)
        w.add_unk_token_id(0)
        w.add_add_bos_token(True)
        w.add_add_space_prefix(True)
    else:
        toks, types, merges = bpe_vocab()
        w.add_tokenizer_model("gpt2")
        w.add_tokenizer_pre("llama-bpe" if arch == "llama" else "qwen2")
        w.add_token_list(toks)
        w.add_token_types(types)
        w.add_token_merges(merges)
        w.add_eos_token_id(toks.index("<|im_end|>"))
        w.add_bos_token_id(toks.index("<|endoftext|>"))
        w.add_add_bos_token(arch == "llama")
        w.add_chat_template(CHATML)
    V = len(toks)
    w.add_block_count(layers)
    w.add_context_length(256)
    w.add_embedding_length(d)
    w.add_feed_forward_length(ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    if norm == "layer":
        w.add_layer_norm_eps(1e-5)
    else:
        w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(rope_dim or head_dim)
    if sliding:
        w.add_sliding_window(sliding)
    if moe:
        w.add_uint32(f"{arch}.expert_count", moe[0])
        w.add_uint32(f"{arch}.expert_used_count", moe[1])
    for k, v in (extra or {}).items():                  # arch-specific keys
        if isinstance(v, str):
            w.add_string(f"{arch}.{k}", v)
        elif isinstance(v, int):
            w.add_uint32(f"{arch}.{k}", v)
        else:
            w.add_float32(f"{arch}.{k}", float(v))

    def mat(rows, cols, scale=None):
        s = scale or 1.0 / np.sqrt(cols)
        return (rng.standard_normal((rows, cols)) * s).astype(np.float32)

    def vec(n, mean=0.0, sd=0.1):
        return (mean + sd * rng.standard_normal(n)).astype(np.float32)

    def put(nm, arr, quant=False):
        if quant:
            qd = quantize(arr, GGMLQuantizationType.Q8_0)
            w.add_tensor(nm, qd, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(nm, arr)

    def norm_t(prefix, n):
        put(prefix + ".weight", vec(n, 1.0))
        if norm == "layer":
            put(prefix + ".bias", vec(n))

    put("token_embd.weight", mat(V, d, 1.0))
    if not tied:
        put("output.weight", mat(V, d))
    if out_bias:
        put("output.bias", vec(V))
    norm_t("output_norm", d)
    if rope_freqs:
        put("rope_freqs.weight", np.linspace(1.0, 8.0, (rope_dim or head_dim) // 2).astype(np.float32))
    nq, nk = n_head * head_dim, n_kv * head_dim
    for L in range(layers):
        p = f"blk.{L}."
        if pre_norm:
            norm_t(p + "attn_norm", d)
            if not parallel:
                norm_t(p + "ffn_norm", d)
        if post_norm:
            norm_t(p + "post_attention_norm", d)
            norm_t(p + "post_ffw_norm", d)
        if fused_qkv:
            put(p + "attn_qkv.weight", mat(nq + 2 * nk, d))
            if qkv_bias or all_bias:
                put(p + "attn_qkv.bias", vec(nq + 2 * nk))
        else:
            put(p + "attn_q.weight", mat(nq, d))
            put(p + "attn_k.weight", mat(nk, d))
            put(p + "attn_v.weight", mat(nk, d))
            if qkv_bias or all_bias:
                for k, n in (("attn_q", nq), ("attn_k", nk), ("attn_v", nk)):
                    put(p + k + ".bias", vec(n))
        put(p + "attn_output.weight", mat(d, nq))
        if all_bias:
            put(p + "attn_output.bias", vec(d))
        if moe:
            E = moe[0]
            put(p + "ffn_gate_inp.weight", mat(E, d, 1.0))
            put(p + "ffn_gate_exps.weight", np.stack([mat(ff, d) for _ in range(E)]))
            put(p + "ffn_up_exps.weight", np.stack([mat(ff, d) for _ in range(E)]))
            put(p + "ffn_down_exps.weight", np.stack([mat(d, ff) for _ in range(E)]))
            if shexp:
                put(p + "ffn_gate_shexp.weight", mat(ff, d))
                put(p + "ffn_up_shexp.weight", mat(ff, d))
                put(p + "ffn_down_shexp.weight", mat(d, ff))
        elif fused_gate_up:
            put(p + "ffn_up.weight", mat(2 * ff, d), quant=q8)
        else:
            if mlp == "gated":
                put(p + "ffn_gate.weight", mat(ff, d), quant=q8)
            put(p + "ffn_up.weight", mat(ff, d), quant=q8)
            if all_bias:
                put(p + "ffn_up.bias", vec(ff))
        put(p + "ffn_down.weight", mat(d, ff), quant=q8)
        if all_bias:
            put(p + "ffn_down.bias", vec(d))
        if qk_norm == "head":
            put(p + "attn_q_norm.weight", vec(head_dim, 1.0))
            put(p + "attn_k_norm.weight", vec(head_dim, 1.0))
        elif qk_norm == "full":
            put(p + "attn_q_norm.weight", vec(nq, 1.0))
            put(p + "attn_k_norm.weight", vec(nk, 1.0))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{name}.gguf  {(OUT / f'{name}.gguf').stat().st_size // 1024} KB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    small = dict(d=32, n_head=4, n_kv=2, head_dim=8, ff=48, layers=3)
    # phase 1
    build("tiny-qwen2", "qwen2", qkv_bias=True, q8=True, seed=1)
    build("tiny-qwen3", "qwen3", qk_norm="head", head_dim=24, seed=2)       # head_dim != d/n_head
    build("tiny-llama3", "llama", tied=False, rope_freqs=True, seed=3)
    build("tiny-llama-spm", "llama", tied=False, spm=True, n_kv=4, seed=4)  # SentencePiece, MHA
    build("tiny-mistral", "llama", tied=False, sliding=3, spm=True, seed=5)  # sliding window
    # phase 2 (smaller: they carry more switches)
    build("tiny-gemma", "gemma", **small, spm=True, seed=13)                # Gemma 1: no post norms
    build("tiny-gemma2", "gemma2", **small, post_norm=True, sliding=3, spm=True, seed=6,
          extra={"attn_logit_softcapping": 3.0, "final_logit_softcapping": 5.0})
    build("tiny-gemma3", "gemma3", **{**small, "layers": 4}, post_norm=True, qk_norm="head",
          sliding=3, spm=True, tied=False, seed=7,
          extra={"attention.sliding_window_pattern": 2, "rope.scaling.type": "linear",
                 "rope.scaling.factor": 4.0})
    build("tiny-phi3", "phi3", **small, fused_qkv=True, fused_gate_up=True, mlp="plain",
          tied=False, spm=True, seed=8)
    build("tiny-phi2", "phi2", **{**small, "n_kv": 4}, norm="layer", parallel=True, mlp="plain",
          fused_qkv=True, all_bias=True, rope_dim=4, tied=False, out_bias=True, seed=9)
    build("tiny-stablelm", "stablelm", **small, norm="layer", qkv_bias=True, rope_dim=4,
          tied=False, seed=10)
    build("tiny-olmo2", "olmo2", **small, pre_norm=False, post_norm=True, qk_norm="full",
          tied=False, seed=11)
    build("tiny-granite", "granite", **small, seed=12,
          extra={"embedding_scale": 2.0, "residual_scale": 0.5, "logit_scale": 3.0,
                 "attention.scale": 0.2})
    # phase 3: mixture of experts
    build("tiny-granitemoe", "granitemoe", **small, moe=(6, 2), shexp=True, seed=14,
          extra={"embedding_scale": 2.0, "residual_scale": 0.5, "logit_scale": 3.0})
    build("tiny-olmoe", "olmoe", **small, moe=(8, 3), qk_norm="full", tied=False, seed=15)
    sys.exit(0)
