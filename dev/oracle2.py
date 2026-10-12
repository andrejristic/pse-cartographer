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
    """Yields (hf_name, fp32 array) one tensor at a time (never the whole model in RAM)."""
    yield from _gguf_state(path, arch, cfg, permuted)


def _gguf_state(path, arch, cfg, permuted):
    from pse_a.core.source.gguf import open_gguf
    from pse_a.core.source import dequant as dq
    g = open_gguf(path)
    t = lambda n: dq.tensor(g, n).astype(np.float32)
    has = lambda n: n in g.tensors
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads
    sd = {}
    yield "model.embed_tokens.weight", t("token_embd.weight")
    pre = "model.layers.{L}."
    if arch == "phi2":
        yield "model.final_layernorm.weight", t("output_norm.weight")
        yield "model.final_layernorm.bias", t("output_norm.bias")
    else:
        yield "model.norm.weight", t("output_norm.weight")
    if has("output.weight"):
        yield "lm_head.weight", t("output.weight")
    if has("output.bias"):
        yield "lm_head.bias", t("output.bias")
    for L in range(cfg.num_hidden_layers):
        p, q = f"blk.{L}.", pre.format(L=L)
        if arch == "phi3":
            parts = {}
        elif has(p + "attn_qkv.weight"):
            qkv = t(p + "attn_qkv.weight")
            hd = cfg.hidden_size // H
            Q, K, V = np.split(qkv, [H * hd, (H + KV) * hd])
            parts = {"q": Q, "k": K, "v": V}
            if has(p + "attn_qkv.bias"):
                bq, bk, bv = np.split(t(p + "attn_qkv.bias"), [H * hd, (H + KV) * hd])
                for n, b in (("q", bq), ("k", bk), ("v", bv)):
                    yield q + f"self_attn.{n}_proj.bias", b
        else:
            parts = {n: t(p + f"attn_{n}.weight") for n in "qkv"}
            for n in "qkv":
                if has(p + f"attn_{n}.bias"):
                    yield q + f"self_attn.{n}_proj.bias", t(p + f"attn_{n}.bias")
        if permuted:
            parts["q"], parts["k"] = unpermute(parts["q"], H), unpermute(parts["k"], KV)
        for n, w in parts.items():
            yield q + f"self_attn.{n}_proj.weight", w
        if arch == "phi2":
            yield q + "self_attn.dense.weight", t(p + "attn_output.weight")
            yield q + "self_attn.dense.bias", t(p + "attn_output.bias")
            yield q + "input_layernorm.weight", t(p + "attn_norm.weight")
            yield q + "input_layernorm.bias", t(p + "attn_norm.bias")
            yield q + "mlp.fc1.weight", t(p + "ffn_up.weight")
            yield q + "mlp.fc1.bias", t(p + "ffn_up.bias")
            yield q + "mlp.fc2.weight", t(p + "ffn_down.weight")
            yield q + "mlp.fc2.bias", t(p + "ffn_down.bias")
            continue
        if arch == "phi3":                                    # HF keeps qkv and gate_up fused
            yield q + "self_attn.qkv_proj.weight", t(p + "attn_qkv.weight")
            yield q + "self_attn.o_proj.weight", t(p + "attn_output.weight")
            yield q + "mlp.gate_up_proj.weight", t(p + "ffn_up.weight")
            yield q + "mlp.down_proj.weight", t(p + "ffn_down.weight")
            yield q + "input_layernorm.weight", t(p + "attn_norm.weight")
            yield q + "post_attention_layernorm.weight", t(p + "ffn_norm.weight")
            continue
        yield q + "self_attn.o_proj.weight", t(p + "attn_output.weight")
        if has(p + "ffn_gate_inp.weight"):                     # mixture of experts
            m = "block_sparse_moe" if arch == "granitemoe" else "mlp"
            yield q + f"{m}.{'router' if arch == 'granitemoe' else 'gate'}.weight", t(p + "ffn_gate_inp.weight")
            yield q + f"{m}.experts.gate_up_proj", np.concatenate(
                [t(p + "ffn_gate_exps.weight"), t(p + "ffn_up_exps.weight")], axis=1)
            yield q + f"{m}.experts.down_proj", t(p + "ffn_down_exps.weight")
            if has(p + "ffn_up_shexp.weight"):
                yield q + "shared_mlp.input_linear.weight", np.concatenate(
                    [t(p + "ffn_gate_shexp.weight"), t(p + "ffn_up_shexp.weight")], axis=0)
                yield q + "shared_mlp.output_linear.weight", t(p + "ffn_down_shexp.weight")
        else:
            for a, b in (("ffn_gate", "gate_proj"), ("ffn_up", "up_proj"), ("ffn_down", "down_proj")):
                yield q + f"mlp.{b}.weight", t(p + f"{a}.weight")
        if arch == "olmoe":
            yield q + "self_attn.q_norm.weight", t(p + "attn_q_norm.weight")
            yield q + "self_attn.k_norm.weight", t(p + "attn_k_norm.weight")
        if arch == "olmo2":
            qn, kn = t(p + "attn_q_norm.weight"), t(p + "attn_k_norm.weight")
            if permuted:      # the norm runs over the full vector, BEFORE rope: permute it too
                qn = unpermute(qn[:, None], H)[:, 0]
                kn = unpermute(kn[:, None], KV)[:, 0]
            yield q + "self_attn.q_norm.weight", qn
            yield q + "self_attn.k_norm.weight", kn
            yield q + "post_attention_layernorm.weight", t(p + "post_attention_norm.weight")
            yield q + "post_feedforward_layernorm.weight", t(p + "post_ffw_norm.weight")
        else:
            yield q + "input_layernorm.weight", t(p + "attn_norm.weight")
            yield q + "post_attention_layernorm.weight", t(p + "ffn_norm.weight")


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
        from transformers.initialization import no_init_weights
        with no_init_weights():          # every weight is overwritten from the GGUF anyway
            model = AutoModelForCausalLM.from_config(cfg, dtype=dt)
        params = dict(model.named_parameters())
        seen, unexpected = set(), []
        with torch.no_grad():
            for k, v in gguf_state(path, arch, cfg, permuted):
                if k not in params:
                    unexpected.append(k)
                    continue
                params[k].copy_(torch.from_numpy(v))      # cast to dt in place, then free v
                seen.add(k)
                del v
        if getattr(cfg, "tie_word_embeddings", False):
            model.tie_weights()                         # lm_head shares the embedding
            seen.add("lm_head.weight")
        missing = [m for m in params if m not in seen]
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
    if rt.spec.n_expert:
        # top-k selection is a step function: a finite difference that crosses a routing
        # switch measures the step, not the gradient. Freeze the experts chosen in the base
        # pass (router WEIGHTS stay smooth) -- the analytic VJP must then match.
        rt.force_routing = rt.route(ids)
    f = lambda h: float(rt.final_norm(rt.run_resid(ids, start_layer=L, x=h)[0][-1]) @ U[0])
    fd = (f(h0 + eps * D) - f(h0 - eps * D)) / (2 * eps)
    rt.force_routing = None
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
