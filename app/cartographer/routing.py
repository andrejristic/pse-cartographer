"""
Expert routing for mixture-of-experts patients (0.4): the second map next to J-space.

A dense model sends every token through the same MLP. A MoE model has a router that, at every
layer, sends each token to a few experts out of many. That choice is observable, and it gives
Cartographer questions a dense model cannot answer:
  - routing drift:     how much the chosen experts change from prompt to prompt (next to the
                       J-space drift of the concepts held),
  - expert usage:      which experts carry a prompt, layer by layer,
  - expert ↔ concept:  which concepts the model holds when a given expert is chosen.

All pure functions over PromptReadout.routing (filled by CartographySession for MoE models).
"""
from __future__ import annotations

import numpy as np

from . import drift as drift_mod


def has_routing(readouts) -> bool:
    return bool(readouts) and all(r.routing for r in readouts)


def _last_set(r, L) -> set:
    return {e for e, _ in r.routing.get(L, {}).get("last", [])}


def routing_drift(readouts) -> list:
    """Per consecutive pair: Jaccard distance of the last token's experts, averaged over layers."""
    out = []
    for a, b in zip(readouts, readouts[1:]):
        layers = sorted(set(a.routing) & set(b.routing))
        ds = []
        for L in layers:
            sa, sb = _last_set(a, L), _last_set(b, L)
            ds.append(1.0 - len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0)
        out.append(float(np.mean(ds)) if ds else 0.0)
    return out


def usage_matrix(readouts, which=None) -> np.ndarray:
    """(layers, experts) share of routed token slots; `which` = prompt index or None for all."""
    rs = readouts if which is None else [readouts[which]]
    layers = sorted(rs[0].routing)
    M = np.array([np.sum([r.routing[L]["usage"] for r in rs], axis=0) for L in layers], dtype=float)
    return M / np.maximum(M.sum(1, keepdims=True), 1.0)


def expert_concepts(readouts, top: int = 6) -> dict:
    """{layer: {expert: [(concept, score), ...]}} -- concepts held at that layer, weighted by how
    strongly the router sent the prompt's last token to that expert."""
    acc: dict = {}
    for r in readouts:
        held = {lr.layer: lr.top_concepts[:12] for lr in r.per_layer}
        for L, rt in r.routing.items():
            for e, w in rt["last"]:
                for c, wc in held.get(L, []):
                    d = acc.setdefault(L, {}).setdefault(e, {})
                    d[c] = d.get(c, 0.0) + w * wc
    return {L: {e: sorted(cs.items(), key=lambda kv: -kv[1])[:top] for e, cs in es.items()}
            for L, es in acc.items()}


def summary(readouts) -> dict:
    if not has_routing(readouts):
        return {}
    rd = routing_drift(readouts)
    jd = drift_mod.drift(readouts)
    n_exp = len(readouts[0].routing[next(iter(readouts[0].routing))]["usage"])
    U = usage_matrix(readouts)
    used = (U > 0).sum(1)
    corr = float(np.corrcoef(rd, jd)[0, 1]) if len(rd) > 2 and np.std(rd) > 0 and np.std(jd) > 0 else None
    return {"n_experts": int(n_exp), "experts_per_token": len(readouts[0].routing[
                next(iter(readouts[0].routing))]["last"]),
            "mean_routing_drift": float(np.mean(rd)) if rd else 0.0,
            "max_routing_drift": float(np.max(rd)) if rd else 0.0,
            "experts_used_mean": float(used.mean()), "drift_correlation": corr}
