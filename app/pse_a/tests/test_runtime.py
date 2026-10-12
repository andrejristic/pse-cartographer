"""Runtime bez modela sa interneta: mini GGUF-ovi u tests/fixtures.

Svaki fixture je pravi GGUF jedne podrzane arhitekture sa svim njenim
prekidacima ukljucenim, ali sitan. Ovde se proverava ono sto se tiho kvari:
  - VJP (rucni backward) prema KONACNIM RAZLIKAMA, u float64, za svaki fixture
  - inkrementalno generisanje (KV kes) daje iste logite kao pun forward
  - tokenizer: encode -> decode vraca tekst
  - SAE gradijenti prema konacnim razlikama
Prolazi na Linux-u i Windows-u u CI-ju posle instalacije paketa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parent / "fixtures"


def fixtures() -> list[Path]:
    return sorted(FIX.glob("*.gguf"))


def _rt64(path):
    from ..core.runtime import numpy_rt
    old = numpy_rt.F
    numpy_rt.F = np.float64
    try:
        return numpy_rt.NumpyRuntime(path, resident=True)
    finally:
        numpy_rt.F = old


def check_vjp_fd(path, tol: float = 1e-6) -> list[str]:
    from ..core.runtime import numpy_rt
    rt = _rt64(path)
    old = numpy_rt.F
    numpy_rt.F = np.float64
    bad = []
    try:
        ids = rt.encode("the cat sat on the mat and the dog ran")
        rng = np.random.default_rng(0)
        U = rng.standard_normal((3, rt.d))
        layers = list(range(rt.n_layers))
        g = rt.vjp_multi(ids, layers, U, position=-1, normed=True)
        if rt.spec.n_expert:                  # FD must not cross a routing switch
            rt.force_routing = rt.route(ids)
        for L in layers:
            _, acts = rt.run_resid(ids, hook_points={f"resid_pre.{L}"})
            h0 = acts[f"resid_pre.{L}"].copy()
            D = rng.standard_normal(h0.shape)
            eps = 1e-5

            def f(h):
                x, _ = rt.run_resid(ids, start_layer=L, x=h)
                return rt.final_norm(x[-1]) @ U.T
            fd = (f(h0 + eps * D) - f(h0 - eps * D)) / (2 * eps)       # (P,)
            an = np.einsum("ptd,td->p", g[L], D)
            err = np.abs(fd - an).max() / max(np.abs(fd).max(), 1e-12)
            if not err < tol:
                bad.append(f"{path.name}: VJP layer {L} rel err {err:.2e}")
    finally:
        numpy_rt.F = old
    return bad


def check_generate(path) -> list[str]:
    from ..core.runtime.numpy_rt import NumpyRuntime
    rt = NumpyRuntime(path)
    ids = rt.encode("the cat")
    out = list(rt.generate(ids, max_new=6, temperature=0))
    full = ids + out
    lg = rt.logits(full[:-1])
    greedy = [int(i) for i in lg[len(ids) - 1:].argmax(-1)]
    if greedy != out:
        return [f"{path.name}: KV-cache generation {out} != full forward {greedy}"]
    return []


def check_tokenizer(path) -> list[str]:
    from ..core.runtime.numpy_rt import NumpyRuntime
    rt = NumpyRuntime(path)
    bad = []
    for t in ("the cat sat", "Hello, World! 123", "ćevapi — тест 🙂", "a  b\n\nc"):
        back = rt.decode(rt.encode(t, bos=False))
        if back != t:
            bad.append(f"{path.name}: roundtrip {t!r} -> {back!r}")
    if not rt.chat_ids("system", [("user", "hi")]):
        bad.append(f"{path.name}: empty chat prompt")
    return bad


def check_sae_grad() -> list[str]:
    """Rucni SAE gradijenti (bez AuxK i sa AuxK) prema konacnim razlikama."""
    from ..core.sae import sae as S
    rng = np.random.default_rng(1)
    d, m, k, B = 6, 12, 3, 5
    x = rng.standard_normal((B, d))
    P = {"W_enc": rng.standard_normal((d, m)), "b_enc": rng.standard_normal(m) * 0.1,
         "W_dec": rng.standard_normal((m, d)), "b_dec": rng.standard_normal(d) * 0.1}

    def loss(P):
        z, _, _ = S.encode_np(x, P["W_enc"], P["b_enc"], P["b_dec"], k)
        return float(((x - (z @ P["W_dec"] + P["b_dec"])) ** 2).sum(-1).mean())

    # analiticki gradijent iz istih formula kao u train_sae (bez AuxK)
    xc = x - P["b_dec"]
    z, pre, idx = S.encode_np(x, P["W_enc"], P["b_enc"], P["b_dec"], k)
    err = x - (z @ P["W_dec"] + P["b_dec"])
    G = (-2.0 / B) * err
    dz = G @ P["W_dec"].T
    dpre = np.zeros_like(pre)
    live = np.take_along_axis(pre, idx, axis=-1) > 0
    np.put_along_axis(dpre, idx, np.take_along_axis(dz, idx, axis=-1) * live, axis=-1)
    an = {"W_dec": z.T @ G, "W_enc": xc.T @ dpre, "b_enc": dpre.sum(0),
          "b_dec": G.sum(0) - (dpre @ P["W_enc"].T).sum(0)}
    bad = []
    for name in P:
        D = rng.standard_normal(P[name].shape)
        eps = 1e-6
        Pp = {k2: v.copy() for k2, v in P.items()}
        Pm = {k2: v.copy() for k2, v in P.items()}
        Pp[name] += eps * D
        Pm[name] -= eps * D
        fd = (loss(Pp) - loss(Pm)) / (2 * eps)
        a = float((an[name] * D).sum())
        if abs(fd - a) > 1e-5 * max(1.0, abs(fd)):
            bad.append(f"SAE grad {name}: fd {fd:.6g} vs analytic {a:.6g}")
    return bad


def check_sae_train() -> list[str]:
    """Kratko treniranje na sintetickim retkim podacima mora smanjiti FVU."""
    from ..core.sae.sae import SAEConfig, train_sae
    rng = np.random.default_rng(2)
    d, n_true = 16, 32
    D = rng.standard_normal((n_true, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    def sample(n):
        c = np.zeros((n, n_true))
        for i in range(n):
            j = rng.choice(n_true, 3, replace=False)
            c[i, j] = rng.uniform(0.5, 2.0, 3)
        return (c @ D).astype(np.float32)

    cfg = SAEConfig(d=d, expansion=4, k=4, k_aux=16, batch=128, steps=600, lr=2e-3,
                    dead_after=200, resample_every=300, seed=0)
    res = train_sae(sample, cfg, sample(2048))
    first = res.history[0]["fvu"] if res.history else 1.0
    if not (res.fvu < 0.5 and res.fvu < first):
        return [f"SAE did not learn: FVU {first:.3f} -> {res.fvu:.3f}"]
    return []


def run() -> list[str]:
    bad = []
    fx = fixtures()
    if not fx:
        return ["no fixtures found in tests/fixtures"]
    for p in fx:
        bad += check_vjp_fd(p)
        bad += check_generate(p)
        bad += check_tokenizer(p)
    bad += check_sae_grad()
    bad += check_sae_train()
    return bad


if __name__ == "__main__":
    bad = run()
    for b in bad:
        print("  " + b)
    raise SystemExit(1 if bad else 0)
