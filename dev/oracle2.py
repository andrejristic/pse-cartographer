"""Two-stage oracle for models too big to hold twice in RAM (and for architectures
transformers cannot load from GGUF).

  stage 1:  .dev/venv/bin/python dev/oracle2.py hf   <model.gguf> [--hf-dir DIR] [--bf16]
            transformers forward -> <model>.oracle.npz  (logits + hidden states)
  stage 2:  .dev/venv/bin/python dev/oracle2.py ours <model.gguf>
            our runtime forward, compared against the npz; plus a float32 finite-difference
            spot check of our VJP on the real model, and a greedy continuation as a sanity read.

For architectures transformers loads from GGUF (gemma2/3, phi3, stablelm, llama, qwen...)
stage 1 uses that loader. For the others (granite, olmo2, phi2) --hf-dir points at the
original model's config.json and the weights are moved from the GGUF into the transformers
model by name (`MAP` below), un-permuting llama.cpp's q/k permutation where it applies.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

PROMPT = "She saw the man with the telescope. The bank by the river was"


def unpermute(w, n_head):
    """Inverse of llama.cpp's permute(): GGUF q/k rows -> HF rows."""
    out, d = w.shape
    return w.reshape(n_head, out // n_head // 2, 2, d).swapaxes(1, 2).reshape(out, d)


def gguf_state(path, arch, cfg, permuted):
    from pse_a.core.source.gguf import open_gguf
    from pse_a.core.source import dequant as dq
    g = open_gguf(path)
    t = lambda n: dq.tensor(g, n).astype(np.float32)
    has = lambda n: n in g.tensors
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    sd = {"model.embed_tokens.weight": t("token_embd.weight")}
    pre = "model.layers.{L}."
    if arch == "phi2":
        sd["model.final_layernorm.weight"] = t("output_norm.weight")
        sd["model.final_layernorm.bias"] = t("output_norm.bias")
    else:
        sd["model.norm.weight"] = t("output_norm.weight")
    if has("output.weight"):
        sd["lm_head.weight"] = t("output.weight")
    if has("output.bias"):
        sd["lm_head.bias"] = t("output.bias")
    for L in range(cfg.num_hidden_layers):
        p, q = f"blk.{L}.", pre.format(L=L)
        if has(p + "attn_qkv.weight"):
            qkv = t(p + "attn_qkv.weight")
            hd = cfg.hidden_size // H
            Q, K, V = np.split(qkv, [H * hd, (H + KV) * hd])
            parts = {"q": Q, "k": K, "v": V}
            if has(p + "attn_qkv.bias"):
                bq, bk, bv = np.split(t(p + "attn_qkv.bias"), [H * hd, (H + KV) * hd])
                for n, b in (("q", bq), ("k", bk), ("v", bv)):
                    sd[q + f"self_attn.{n}_proj.bias"] = b
        else:
            parts = {n: t(p + f"attn_{n}.weight") for n in "qkv"}
            for n in "qkv":
                if has(p + f"attn_{n}.bias"):
                    sd[q + f"self_attn.{n}_proj.bias"] = t(p + f"attn_{n}.bias")
        if permuted:
            parts["q"], parts["k"] = unpermute(parts["q"], H), unpermute(parts["k"], KV)
        for n, w in parts.items():
            sd[q + f"self_attn.{n}_proj.weight"] = w
        if arch == "phi2":
            sd[q + "self_attn.dense.weight"] = t(p + "attn_output.weight")
            sd[q + "self_attn.dense.bias"] = t(p + "attn_output.bias")
            sd[q + "input_layernorm.weight"] = t(p + "attn_norm.weight")
            sd[q + "input_layernorm.bias"] = t(p + "attn_norm.bias")
            sd[q + "mlp.fc1.weight"] = t(p + "ffn_up.weight")
            sd[q + "mlp.fc1.bias"] = t(p + "ffn_up.bias")
            sd[q + "mlp.fc2.weight"] = t(p + "ffn_down.weight")
            sd[q + "mlp.fc2.bias"] = t(p + "ffn_down.bias")
            continue
        sd[q + "self_attn.o_proj.weight"] = t(p + "attn_output.weight")
        for a, b in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj")):
            sd[q + f"mlp.{b}.weight"] = t(p + f"{a}.weight")
        if arch == "olmo2":
            qn, kn = t(p + "attn_q_norm.weight"), t(p + "attn_k_norm.weight")
            if permuted:      # the norm runs over the full vector, BEFORE rope: permute it too
                qn = unpermute(qn[:, None], H)[:, 0]
                kn = unpermute(kn[:, None], KV)[:, 0]
            sd[q + "self_attn.q_norm.weight"] = qn
            sd[q + "self_attn.k_norm.weight"] = kn
            sd[q + "post_attention_layernorm.weight"] = t(p + "post_attention_norm.weight")
            sd[q + "post_feedforward_layernorm.weight"] = t(p + "post_ffw_norm.weight")
        else:
            sd[q + "input_layernorm.weight"] = t(p + "attn_norm.weight")
            sd[q + "post_attention_layernorm.weight"] = t(p + "ffn_norm.weight")
    return sd


def stage_hf(path, hf_dir, bf16, permuted):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from pse_a.core.runtime.numpy_rt import NumpyRuntime
    from pse_a.core.source.gguf import open_gguf
    torch.set_num_threads(os.cpu_count())
    arch = open_gguf(path).metadata["general.architecture"]
    dt = torch.bfloat16 if bf16 else torch.float32
    if hf_dir:
        cfg = AutoConfig.from_pretrained(hf_dir)
        model = AutoModelForCausalLM.from_config(cfg, dtype=dt)
        sd = {k: torch.from_numpy(v).to(dt) for k, v in gguf_state(path, arch, cfg, permuted).items()}
        if "lm_head.weight" not in sd and getattr(cfg, "tie_word_embeddings", False):
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "rotary" not in m]
        print("missing", missing[:6], "unexpected", unexpected[:6])
    else:
        model = AutoModelForCausalLM.from_pretrained(str(Path(path).parent), gguf_file=Path(path).name,
                                                     dtype=dt)
    model.eval()
    rt_tok = NumpyRuntime.__new__(NumpyRuntime)          # only for the tokenizer
    from pse_a.core.runtime.tokenizer import GGUFTokenizer
    ids = GGUFTokenizer(open_gguf(path).metadata).encode(PROMPT)
    with torch.no_grad():
        out = model(torch.tensor([ids]), output_hidden_states=True)
    n = len(out.hidden_states) - 1
    hs = {str(L): out.hidden_states[L][0].float().numpy() for L in (0, n // 2, n - 1)}
    np.savez(str(path) + ".oracle.npz", ids=np.array(ids), logits=out.logits[0].float().numpy(),
             **{f"h{k}": v for k, v in hs.items()})
    print("saved", str(path) + ".oracle.npz", "tokens", len(ids), "dtype", dt)


def stage_ours(path):
    from pse_a.core.runtime.numpy_rt import NumpyRuntime
    o = np.load(str(path) + ".oracle.npz")
    rt = NumpyRuntime(path)
    print("arch", rt.spec.arch, "resident", rt.caps.resident)
    ids = [int(i) for i in o["ids"]]
    x, acts = rt.run_resid(ids)
    ours = rt.unembed(rt.final_norm(x))
    ref = o["logits"]
    print(f"logits max abs diff {np.abs(ours - ref).max():.4g} (scale {np.abs(ref).max():.3g}), "
          f"argmax agree {(ours.argmax(-1) == ref.argmax(-1)).mean():.3f}")
    for k in o.files:
        if k.startswith("h"):
            L = int(k[1:])
            h = acts[f"resid_pre.{L}"]
            print(f"resid_pre.{L} rel diff {np.abs(h - o[k]).max() / np.abs(o[k]).max():.2e}")
    # FD spot check of the VJP on the real model (float32: expect ~1e-3)
    rng = np.random.default_rng(0)
    U = rng.standard_normal((1, rt.d)).astype(np.float32)
    L = rt.n_layers // 2
    g = rt.vjp_multi(ids, [L], U)[L][0]
    h0 = acts[f"resid_pre.{L}"].copy()
    D = rng.standard_normal(h0.shape).astype(np.float32)
    eps = 1e-2 * float(np.abs(h0).mean())
    f = lambda h: float(rt.final_norm(rt.run_resid(ids, start_layer=L, x=h)[0][-1]) @ U[0])
    fd = (f(h0 + eps * D) - f(h0 - eps * D)) / (2 * eps)
    an = float((g * D).sum())
    print(f"VJP FD spot check layer {L}: fd {fd:.5g} analytic {an:.5g} rel {abs(fd - an) / abs(fd):.2e}")
    gen = list(rt.generate(rt.encode("The capital of France is"), max_new=8, temperature=0,
                           stop_ids=rt.tok.stop_ids()))
    print("greedy:", repr(rt.decode(gen)))


if __name__ == "__main__":
    stage, path = sys.argv[1], sys.argv[2]
    hf_dir = sys.argv[sys.argv.index("--hf-dir") + 1] if "--hf-dir" in sys.argv else None
    if stage == "hf":
        stage_hf(path, hf_dir, "--bf16" in sys.argv, "--permuted" in sys.argv)
    else:
        stage_ours(path)
