"""
J-space movement: drift, diff, clustering, ignition tracking (C7) — the heart.

All pure numpy over the PromptReadout list from a session. This is what lets the human
WATCH the J-space change as prompts are fed:
  - drift:      how much the "held in mind" concept set moves between consecutive prompts,
  - diff:       exactly which concepts turned on/off between two prompts (+ region deltas),
  - cluster:    group prompts whose J-space signatures are alike,
  - ignition:   track where the model abruptly commits, prompt by prompt.
"""
from __future__ import annotations

import numpy as np


def representative_layer(readout):
    """The layer to read concepts from: the ignition layer, else the max-variance layer."""
    if readout.ignition_layer is not None:
        for lr in readout.per_layer:
            if lr.layer == readout.ignition_layer:
                return lr
    return max(readout.per_layer, key=lambda lr: lr.var_frac, default=None)


def active_set(readout, weight_min: float = 0.0) -> set:
    """Set of concept labels the model is 'holding in mind' at its representative layer."""
    lr = representative_layer(readout)
    if lr is None:
        return set()
    return {label for label, w in lr.top_concepts if w > weight_min}


def _weight_vector(readout, vocab):
    lr = representative_layer(readout)
    v = np.zeros(len(vocab))
    if lr is None:
        return v
    idx = {lab: i for i, lab in enumerate(vocab)}
    for label, w in lr.top_concepts:
        if label in idx:
            v[idx[label]] = w
    return v


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def drift(readouts) -> list:
    """Jaccard distance between consecutive prompts' active concept sets (0=same, 1=disjoint)."""
    sets = [active_set(r) for r in readouts]
    return [_jaccard(sets[i - 1], sets[i]) for i in range(1, len(sets))]


def diff(readout_a, readout_b) -> dict:
    """What changed between two prompts."""
    a, b = active_set(readout_a), active_set(readout_b)
    va = {lr.layer: lr.var_frac for lr in readout_a.per_layer}
    vb = {lr.layer: lr.var_frac for lr in readout_b.per_layer}
    return {
        "turned_on": sorted(b - a),
        "turned_off": sorted(a - b),
        "shared": sorted(a & b),
        "var_frac_delta": {L: round(vb.get(L, 0) - va.get(L, 0), 4)
                           for L in sorted(set(va) | set(vb))},
        "ignition_shift": (readout_a.ignition_layer, readout_b.ignition_layer),
    }


def cluster(readouts, threshold: float = 0.6) -> list:
    """
    Greedy clustering of prompts by J-space signature (cosine over concept weights).
    Returns list of clusters; each cluster is a list of readout indices. No sklearn.
    """
    vocab = sorted({lab for r in readouts for lab, _ in
                    (representative_layer(r).top_concepts if representative_layer(r) else [])})
    vecs = [_weight_vector(r, vocab) for r in readouts]
    norms = [np.linalg.norm(v) or 1.0 for v in vecs]
    clusters, centroids = [], []
    for i, v in enumerate(vecs):
        best, best_sim = -1, -1.0
        for c, cen in enumerate(centroids):
            sim = float(v @ cen) / (norms[i] * (np.linalg.norm(cen) or 1.0))
            if sim > best_sim:
                best, best_sim = c, sim
        if best_sim >= threshold:
            clusters[best].append(i)
            centroids[best] = (centroids[best] * (len(clusters[best]) - 1) + v) / len(clusters[best])
        else:
            clusters.append([i]); centroids.append(v.copy())
    return clusters


def ignition_track(readouts) -> list:
    """Ignition layer per prompt (None where undetected) — watch commitment move."""
    return [r.ignition_layer for r in readouts]


def summary(readouts) -> dict:
    """Compact numbers the report (C9) and the drift view render."""
    d = drift(readouts)
    igs = [i for i in ignition_track(readouts) if i is not None]
    return {
        "n_prompts": len(readouts),
        "mean_drift": float(np.mean(d)) if d else 0.0,
        "max_drift": float(np.max(d)) if d else 0.0,
        "ignition_layers_seen": sorted(set(igs)),
        "ignition_detected_fraction": (len(igs) / len(readouts)) if readouts else 0.0,
        "n_clusters": len(cluster(readouts)),
    }
