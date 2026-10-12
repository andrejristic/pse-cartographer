"""
J-space basis: the one-time, expensive step (C5).

Estimates the averaged lens rows W_U·J_l for a layer via vector-Jacobian products over a
calibration prompt set, and records the vocabulary TOKEN behind each probe row so later
readouts can label concept-chips in plain words. The basis is cached per (model_hash,
layer) in the engine's SHARED_LIB and reused — this is what makes per-prompt readout cheap
later (C6): once the basis exists, a new prompt needs only a forward pass + a numpy
decomposition, no more autograd.

This module keeps its own probe selection (top-norm unembedding rows) so it can carry the
token indices/labels; it does not modify the shared engine.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pse_a.constants import VJP_PROBES


def select_probes(runtime, n_probes: int = VJP_PROBES):
    """Return (rows [n,d_model], token_ids [n], labels [n]) — the probe set (spec §7)."""
    W_U = runtime.W_U.weight.detach().float().cpu().numpy()   # [vocab, d_model]
    norms = np.linalg.norm(W_U, axis=1)
    idx = np.argsort(-norms)[:n_probes]
    labels = [runtime.token_str(int(i)) for i in idx]
    return W_U[idx], idx.astype(int), labels


def _cache_dir(shared_lib) -> Path:
    d = Path(shared_lib.root) / "jspace_basis"
    d.mkdir(parents=True, exist_ok=True)
    return d


class JSpaceBasis:
    """Averaged lens rows for one layer + the token label per row."""

    def __init__(self, layer, rows, token_ids, labels, n_prompts):
        self.layer = layer
        self.rows = np.asarray(rows, dtype=np.float64)        # [n_probes, d_model]
        self.token_ids = list(token_ids)
        self.labels = list(labels)
        self.n_prompts = n_prompts

    def save(self, shared_lib, model_hash):
        d = _cache_dir(shared_lib)
        np.save(d / f"{model_hash}_L{self.layer}.rows.npy", self.rows)
        (d / f"{model_hash}_L{self.layer}.meta.json").write_text(json.dumps({
            "layer": self.layer, "token_ids": self.token_ids,
            "labels": self.labels, "n_prompts": self.n_prompts}))

    @classmethod
    def load(cls, shared_lib, model_hash, layer):
        d = _cache_dir(shared_lib)
        rp = d / f"{model_hash}_L{layer}.rows.npy"
        mp = d / f"{model_hash}_L{layer}.meta.json"
        if not (rp.exists() and mp.exists()):
            return None
        meta = json.loads(mp.read_text())
        return cls(layer, np.load(rp), meta["token_ids"], meta["labels"],
                   meta["n_prompts"])


def estimate_basis(runtime, model_map, layer, calib_prompts, shared_lib, model_hash,
                   unit=None, use_cache=True, progress=None) -> JSpaceBasis:
    """
    Estimate (or load) the averaged lens rows for `layer`. Averaged over calib_prompts;
    cached per (model_hash, layer). `unit` (a PSE unit) carries the compute budget.
    """
    if use_cache:
        cached = JSpaceBasis.load(shared_lib, model_hash, layer)
        if cached is not None:
            return cached

    rows0, token_ids, labels = select_probes(runtime)
    final = runtime.n_layers - 1
    acc = np.zeros((rows0.shape[0], model_map.d_model), dtype=np.float64)
    used = 0
    for i, p in enumerate(calib_prompts):
        if unit is not None and unit.budget.exhausted():
            break
        acc += runtime.vjp(p, from_layer=final, to_layer=layer, cotangents=rows0)
        used += 1
        if unit is not None:
            unit.budget.step(rows0.shape[0])
        if progress:
            progress(i + 1, len(calib_prompts), layer)
    rows = acc / max(used, 1)
    if unit is not None and used > 0:
        unit.theta.note_learning(1)
        unit.outcome_ok = True

    basis = JSpaceBasis(layer, rows, token_ids, labels, used)
    basis.save(shared_lib, model_hash)
    return basis
