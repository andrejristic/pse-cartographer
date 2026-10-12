"""
Cartography session: feed prompts, read the J-space, record the trajectory (C6).

Once the basis exists (C5), each prompt is CHEAP: one forward pass to get the residual at
each mapped layer, then a numpy decomposition into that layer's fixed basis (the engine's
sparse NON-NEGATIVE pursuit: J-space is a cone, not a subspace). Per prompt we record, per
layer: the J-space variance fraction, the decomposition entropy, how close the lens read is
to the model's actual output, and the top concept-chips (as vocabulary tokens); then the
depth-map regions and the ignition layer (the engine's depth_map).

The whole trajectory is written to a separate results file (`results_<model>_<run>.json`)
that the plain-language report (C9) points the human to for a deeper look.

`readout_from_acts` is pure numpy so the readout math is testable without a model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pse_a.core.lens.jlens import depth_map, jspace_decompose, lens_read


@dataclass
class LayerReadout:
    layer: int
    var_frac: float
    entropy: float
    motor_kl: float
    top_concepts: list          # [(label, coef), ...] highest first


@dataclass
class PromptReadout:
    prompt: str
    per_layer: list             # list[LayerReadout]
    ignition_layer: object      # int or None
    region_map: dict


def readout_from_acts(acts_by_layer: dict, final_probe_p: np.ndarray, bases: dict,
                      top_k: int = 25) -> PromptReadout:
    """acts_by_layer: {layer: residual at the read position [d_model]}.
    final_probe_p: the model's own output distribution over the probe tokens."""
    per_layer, depth_input = [], {}
    for layer in sorted(bases):
        basis = bases[layer]
        h = np.asarray(acts_by_layer[layer], dtype=np.float64)
        coef, var_frac, ent = jspace_decompose(h, basis.rows)
        q = lens_read(h, basis.rows)
        p = final_probe_p
        kl = float((p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12))).sum())
        order = np.argsort(-coef)
        chips = [(basis.labels[i], float(coef[i])) for i in order[:top_k] if coef[i] > 0]
        per_layer.append(LayerReadout(layer, float(var_frac), float(ent), kl, chips))
        depth_input[layer] = {"var_frac": float(var_frac), "entropy": float(ent), "motor_kl": kl}
    region = depth_map(depth_input)
    return PromptReadout("", per_layer, region.get("ignition"), region)


class CartographySession:
    def __init__(self, runtime, results_root, bases: dict, run_id: str = "run"):
        self.rt = runtime
        self.root = Path(results_root)
        self.model_hash = runtime.model_hash
        self.bases = bases                      # {layer: JSpaceBasis}
        self.run_id = run_id
        self.readouts: list = []
        any_b = next(iter(bases.values()))
        self._Wp = runtime.W_U[any_b.token_ids]

    def readout(self, prompt: str) -> PromptReadout:
        ids = self.rt.encode(prompt)
        x, acts = self.rt.run_resid(ids, hook_points={f"resid_pre.{L}" for L in self.bases})
        logit = (self.rt.final_norm(x[-1]) @ self._Wp.T).astype(np.float64)
        e = np.exp(logit - logit.max())
        r = readout_from_acts({L: acts[f"resid_pre.{L}"][-1] for L in self.bases},
                              e / e.sum(), self.bases)
        r.prompt = prompt
        self.readouts.append(r)
        return r

    def run_battery(self, prompts, unit=None, progress=None):
        for i, p in enumerate(prompts):
            if unit is not None and unit.budget.exhausted:
                break
            self.readout(p)
            if progress:
                progress(i + 1, len(prompts), p)
        self.write_results()
        return self.readouts

    # -- the separate results file the report points to (C9) ---------------- #
    def results_path(self) -> Path:
        d = self.root / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"results_{self.model_hash[:16]}_{self.run_id}.json"

    def payload(self) -> dict:
        return {
            "model_hash": self.model_hash, "run_id": self.run_id,
            "layers_mapped": sorted(self.bases),
            "probes": next(iter(self.bases.values())).labels,
            "readouts": [{
                "prompt": r.prompt,
                "ignition_layer": r.ignition_layer,
                "regions": {k: r.region_map.get(k) for k in
                            ("sensory", "workspace", "motor", "note", "stride")},
                "per_layer": [{"layer": lr.layer, "var_frac": lr.var_frac,
                               "entropy": lr.entropy, "motor_kl": lr.motor_kl,
                               "top_concepts": lr.top_concepts} for lr in r.per_layer],
            } for r in self.readouts],
        }

    def write_results(self) -> Path:
        p = self.results_path()
        p.write_text(json.dumps(self.payload(), indent=1))
        return p


def readouts_from_payload(payload: dict) -> list:
    """Rebuild PromptReadouts from a sealed cartography artifact (for the GUI)."""
    out = []
    for r in payload["readouts"]:
        pl = [LayerReadout(x["layer"], x["var_frac"], x.get("entropy", 0.0),
                           x.get("motor_kl", 0.0), [tuple(c) for c in x["top_concepts"]])
              for x in r["per_layer"]]
        out.append(PromptReadout(r["prompt"], pl, r["ignition_layer"], r.get("regions", {})))
    return out
