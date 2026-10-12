"""Oracle: our numpy runtime vs transformers (torch) on a real GGUF.
Run with the dev venv:  .dev/venv/bin/python dev/oracle_forward.py <model.gguf>
Checks: tokenization, logits, resid stream per layer, VJP rows vs torch autograd."""
import sys, time, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from pse_a.core.runtime.numpy_rt import NumpyRuntime

path = Path(sys.argv[1])
TEXTS = ["The bank by the river was", "She saw the man with the telescope.",
         "def add(a, b):\n    return a + b", "Ćevapi i burek — 2 + 2 = 4!  Double  spaces\n\nnewlines",
         "Hello world", "  leading spaces and émojis 🙂 中文 тест"]

t0 = time.time(); rt = NumpyRuntime(path); print(f"numpy rt load {time.time()-t0:.1f}s  arch={rt.spec.arch} resident={rt.caps.resident}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(os.cpu_count())
t0 = time.time()
tok = AutoTokenizer.from_pretrained(str(path.parent), gguf_file=path.name)
model = AutoModelForCausalLM.from_pretrained(str(path.parent), gguf_file=path.name, torch_dtype=torch.float32)
model.eval(); print(f"hf load {time.time()-t0:.1f}s")

bad = 0
for t in TEXTS:
    a = rt.encode(t); b = tok(t)["input_ids"]
    if a != b:
        bad += 1; print("TOKEN MISMATCH", repr(t), a[:20], b[:20])
    if rt.decode(rt.encode(t, bos=False)) != t:
        print("  roundtrip differs:", repr(rt.decode(rt.encode(t, bos=False))))
print("tokenization mismatches:", bad, "/", len(TEXTS))

ids = rt.encode(TEXTS[1])
with torch.no_grad():
    out = model(torch.tensor([ids]), output_hidden_states=True)
hf_logits = out.logits[0].numpy(); hs = [h[0].numpy() for h in out.hidden_states]
x, acts = rt.run_resid(ids)
ours = rt.final_norm(x) @ rt.W_U.T
print("logits max abs diff", float(np.abs(ours - hf_logits).max()), " scale", float(np.abs(hf_logits).max()),
      " argmax agree", float((ours.argmax(-1) == hf_logits.argmax(-1)).mean()))
for L in (0, rt.n_layers // 2, rt.n_layers - 1):
    d = np.abs(acts[f"resid_pre.{L}"] - hs[L]).max() / (np.abs(hs[L]).max())
    print(f"resid_pre.{L} rel diff {d:.2e}")

# VJP vs torch autograd: d(u . norm(h_final[pos])) / d h_L, h_L = hidden_states[L] (input to block L)
P = 6; rng = np.random.default_rng(0); U = rng.standard_normal((P, rt.d)).astype(np.float32)
layers = [0, rt.n_layers // 2, rt.n_layers - 2]
t0 = time.time(); g = rt.vjp_multi(ids, layers, U, position=-1, normed=True); print(f"numpy vjp {time.time()-t0:.1f}s")
for L in layers:
    store = {}
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h = h.detach().requires_grad_(True); store["h"] = h
        if args: return (h,) + tuple(args[1:]), kwargs
        kwargs["hidden_states"] = h; return args, kwargs
    hd = model.model.layers[L].register_forward_pre_hook(hook, with_kwargs=True)
    o = model.model(torch.tensor([ids])).last_hidden_state[0, -1]   # already final-normed
    hd.remove()
    grads = []
    for p in range(P):
        gr, = torch.autograd.grad(o @ torch.tensor(U[p]), store["h"], retain_graph=p < P - 1)
        grads.append(gr[0].numpy())
    ref = np.stack(grads)
    rel = np.abs(g[L] - ref).max() / np.abs(ref).max()
    print(f"VJP layer {L}: rel max diff {rel:.2e}")
