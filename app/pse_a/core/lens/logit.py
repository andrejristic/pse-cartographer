"""Logit lens -- [SPEC] Sec 5. Jeftina, iskrena aproksimacija.

"What would the model say if forced to speak at this layer": resid -> FINAL
NORM -> W_U. Final norm se primenjuje UVEK (bez njega lens meri pogresnu
skalu i srednji slojevi izgledaju gluplje nego sto jesu).

Ogranicenje koje [SPEC] trazi da se kaze gde god se lens prikazuje: forward-only,
bez gradijenata, pa POTCENJUJE neverbalizovano medjuracunanje. Taj jaz je
upravo ono sto L4 (J-lens) postoji da zatvori.
"""
from __future__ import annotations

import numpy as np


def waterfall(rt, tokens, k: int = 5) -> dict:
    """Mreza slojevi × pozicije; celija = top-k (token_id, verovatnoca).

    Pakuje se kompaktno (§7.2): token u16/u32 + verovatnoca u8, jer JSON sa
    stringovima za 24×512×5 celija je ~1.5 MB, a binarno ~100 KB.
    """
    acts = rt.forward_with_hooks(tokens)
    layers = sorted(int(n.split(".")[1]) for n in acts if n.startswith("resid_pre."))
    rows = []
    for L in layers + ["final"]:
        h = acts["resid_final"] if L == "final" else acts[f"resid_pre.{L}"]
        z = (rt.final_norm(h) @ rt.W_U.T).astype(np.float64)
        z -= z.max(-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(-1, keepdims=True)
        ti = np.argsort(-p, axis=-1)[:, :k]
        tv = np.take_along_axis(p, ti, axis=-1)
        ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0)).sum(-1)
        rows.append({"layer": L, "ids": ti.astype(np.uint32),
                     "probs": tv.astype(np.float32), "entropy": ent.astype(np.float32)})
    return {"tokens": list(tokens), "layers": [r["layer"] for r in rows], "rows": rows}


def pack(wf: dict) -> bytes:
    """Binarni artefakt: [n_layers, T, k] ids (u32) + probs (u8, kvantizovano)."""
    ids = np.stack([r["ids"] for r in wf["rows"]])
    pr = np.stack([r["probs"] for r in wf["rows"]])
    L, T, k = ids.shape
    head = np.array([L, T, k], dtype=np.uint32).tobytes()
    return head + ids.astype(np.uint32).tobytes() + np.round(pr * 255).astype(np.uint8).tobytes()
