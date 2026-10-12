"""
J-space basis: the one-time, expensive step (C5).

Estimates the averaged lens rows W_U·J_l for EVERY mapped layer via vector-Jacobian
products over a calibration prompt set, and records the vocabulary TOKEN behind each
probe row so later readouts can label concept-chips in plain words. One backward pass
per calibration prompt yields the rows for all layers at once (pse_a NumpyRuntime
vjp_multi). The basis is cached per (model_hash, probe set, calibration set, layer) under
the data root and reused — this is what makes per-prompt readout cheap later (C6): once
the basis exists, a new prompt needs only a forward pass + a numpy decomposition.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pse_a.core.lens.jlens import jlens_rows_multi


_CONCEPTS = Path(__import__("pse_a").__file__).resolve().parent / "data" / "concepts_en.txt"


def concept_words() -> list[str]:
    words = []
    for ln in _CONCEPTS.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#"):
            continue
        words += ln.split()
    return words


def select_probes(rt, n_probes: int = 128, texts=()):
    """(token_ids [n], labels [n]) — probe words the human can read as concepts.

    Probes are WORDS, not arbitrary rows: first the content words of the prompts being
    studied, then a built-in list of common concepts (data/concepts_en.txt). A word is
    used only if the patient's vocabulary holds it as ONE token (with its leading space),
    so the probe row really is that word. (Top-norm unembedding rows, the first idea,
    surface rare byte fragments that read as nothing.)
    """
    ids, labels, seen = [], [], set()
    cand = []
    for t in texts:
        cand += [w for w in __import__("re").findall(r"[A-Za-z]{3,}", t)]
    cand += concept_words()
    seen_words = set()
    for w in cand:
        if w.lower() in seen_words:
            continue
        for form in (" " + w, " " + w.lower(), w):
            tok = rt.encode(form, bos=False)
            if len(tok) == 1 and tok[0] not in seen:
                seen.add(tok[0])
                seen_words.add(w.lower())
                ids.append(int(tok[0]))
                labels.append(w.lower())
                break
        if len(ids) >= n_probes:
            break
    return ids, labels


def _key(model_hash, probe_ids, calib) -> str:
    h = hashlib.sha256(json.dumps([model_hash, list(map(int, probe_ids)), list(calib)]).encode())
    return h.hexdigest()[:16]


def _cache_dir(root) -> Path:
    d = Path(root) / "jspace_basis"
    d.mkdir(parents=True, exist_ok=True)
    return d


class JSpaceBasis:
    """Averaged lens rows for one layer + the token label per row."""

    def __init__(self, layer, rows, token_ids, labels, n_prompts):
        self.layer = int(layer)
        self.rows = np.asarray(rows, dtype=np.float64)        # [n_probes, d_model]
        self.token_ids = [int(t) for t in token_ids]
        self.labels = list(labels)
        self.n_prompts = n_prompts

    def save(self, root, key):
        d = _cache_dir(root)
        np.save(d / f"{key}_L{self.layer}.rows.npy", self.rows)
        (d / f"{key}_L{self.layer}.meta.json").write_text(json.dumps({
            "layer": self.layer, "token_ids": self.token_ids,
            "labels": self.labels, "n_prompts": self.n_prompts}))

    @classmethod
    def load(cls, root, key, layer):
        d = _cache_dir(root)
        rp, mp = d / f"{key}_L{layer}.rows.npy", d / f"{key}_L{layer}.meta.json"
        if not (rp.exists() and mp.exists()):
            return None
        meta = json.loads(mp.read_text())
        return cls(layer, np.load(rp), meta["token_ids"], meta["labels"], meta["n_prompts"])


def estimate_bases(rt, layers, calib_prompts, root, n_probes: int = 128, use_cache=True,
                   progress=None, extra_texts=()) -> tuple[dict, bool]:
    """{layer: JSpaceBasis} for all layers (one backward per prompt). Returns (bases, cached)."""
    probe_ids, labels = select_probes(rt, n_probes, texts=list(calib_prompts) + list(extra_texts))
    key = _key(rt.model_hash, probe_ids, calib_prompts)
    if use_cache:
        cached = {L: JSpaceBasis.load(root, key, L) for L in layers}
        if all(v is not None for v in cached.values()):
            return cached, True
    ids_list = [rt.encode(p) for p in calib_prompts]
    rows = jlens_rows_multi(rt, ids_list, list(layers), probe_ids,
                            progress=(lambda i, n: progress(i, n)) if progress else None)
    bases = {}
    for L in layers:
        b = JSpaceBasis(L, rows[L], probe_ids, labels, len(ids_list))
        b.save(root, key)
        bases[L] = b
    return bases, False
