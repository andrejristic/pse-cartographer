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
          qkv_bias=False, qk_norm=False, tied=True, rope_freqs=False, sliding=None,
          spm=False, q8=False, seed=0):
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
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(head_dim)
    if sliding:
        w.add_sliding_window(sliding)

    def mat(rows, cols, scale=None):
        s = scale or 1.0 / np.sqrt(cols)
        return (rng.standard_normal((rows, cols)) * s).astype(np.float32)

    def put(nm, arr, quant=False):
        if quant:
            qd = quantize(arr, GGMLQuantizationType.Q8_0)
            w.add_tensor(nm, qd, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(nm, arr)

    put("token_embd.weight", mat(V, d, 1.0))
    if not tied:
        put("output.weight", mat(V, d))
    put("output_norm.weight", (1 + 0.1 * rng.standard_normal(d)).astype(np.float32))
    if rope_freqs:
        put("rope_freqs.weight", np.linspace(1.0, 8.0, head_dim // 2).astype(np.float32))
    for L in range(layers):
        p = f"blk.{L}."
        put(p + "attn_norm.weight", (1 + 0.1 * rng.standard_normal(d)).astype(np.float32))
        put(p + "ffn_norm.weight", (1 + 0.1 * rng.standard_normal(d)).astype(np.float32))
        put(p + "attn_q.weight", mat(n_head * head_dim, d))
        put(p + "attn_k.weight", mat(n_kv * head_dim, d))
        put(p + "attn_v.weight", mat(n_kv * head_dim, d))
        put(p + "attn_output.weight", mat(d, n_head * head_dim))
        put(p + "ffn_gate.weight", mat(ff, d), quant=q8)
        put(p + "ffn_up.weight", mat(ff, d), quant=q8)
        put(p + "ffn_down.weight", mat(d, ff), quant=q8)
        if qkv_bias:
            for k, n in (("attn_q", n_head), ("attn_k", n_kv), ("attn_v", n_kv)):
                put(p + k + ".bias", (0.1 * rng.standard_normal(n * head_dim)).astype(np.float32))
        if qk_norm:
            put(p + "attn_q_norm.weight", (1 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32))
            put(p + "attn_k_norm.weight", (1 + 0.1 * rng.standard_normal(head_dim)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{name}.gguf  {(OUT / f'{name}.gguf').stat().st_size // 1024} KB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build("tiny-qwen2", "qwen2", qkv_bias=True, q8=True, seed=1)
    build("tiny-qwen3", "qwen3", qk_norm=True, head_dim=24, seed=2)       # head_dim != d/n_head
    build("tiny-llama3", "llama", tied=False, rope_freqs=True, seed=3)
    build("tiny-llama-spm", "llama", tied=False, spm=True, n_kv=4, seed=4)  # SentencePiece, MHA
    build("tiny-mistral", "llama", tied=False, sliding=3, spm=True, seed=5)  # sliding window
    sys.exit(0)
