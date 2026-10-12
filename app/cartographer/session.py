"""
Cartography session: feed prompts, read the J-space, record the trajectory (C6).

Once the basis exists (C5), each prompt is CHEAP: one forward pass to get the residual at
each mapped layer, then a numpy decomposition into that layer's fixed basis. Per prompt we
record, per layer: the J-space variance fraction and the top concept-chips (as vocabulary
tokens), plus the depth-map regions and the ignition layer.

The whole trajectory is written to a separate results file (`results_<run>.json`) that the
plain-language report (C9) points the human to for a deeper look.

`_readout_from_acts` is pure numpy so the readout math is testable without torch.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pse_a.core.l4_jlens import depth_map, jspace_decompose


@dataclass
class LayerReadout:
    layer: int
    var_frac: float
    top_concepts: list          # [(label, coef), ...] highest first


@dataclass
class PromptReadout:
    prompt: str
    per_layer: list             # list[LayerReadout]
    ignition_layer: object      # int or None
    region_map: dict


def _readout_from_acts(acts_by_layer: dict, bases: dict, top_k: int = 25) -> PromptReadout:
    """
    Pure-numpy readout. acts_by_layer: {layer: residual [seq, d_model]}.
    bases: {layer: JSpaceBasis}. Returns a PromptReadout (prompt filled by caller).
    """
    per_layer = []
    depth_input = []
    for layer in sorted(bases):
        basis = bases[layer]
        h = np.asarray(acts_by_layer[layer], dtype=np.float64).mean(axis=0)   # mean pos
        coef, var_frac = jspace_decompose(h, basis.rows)
        order = np.argsort(-coef)
        chips = [(basis.labels[i], float(coef[i])) for i in order[:top_k]
                 if coef[i] > 0]
        per_layer.append(LayerReadout(layer, float(var_frac), chips))
        depth_input.append({"layer": layer, "var_frac": float(var_frac), "coef": coef})
    region = depth_map(depth_input)
    return PromptReadout("", per_layer, region.get("ignition_layer"), region)


class CartographySession:
    def __init__(self, runtime, model_map, shared_lib, model_hash, bases: dict,
                 run_id: str = "run"):
        self.rt = runtime
        self.mm = model_map
        self.lib = shared_lib
        self.model_hash = model_hash
        self.bases = bases                      # {layer: JSpaceBasis}
        self.run_id = run_id
        self.readouts: list = []

    def readout(self, prompt: str) -> PromptReadout:
        ids = self.rt.encode(prompt)[0].tolist()
        acts = self.rt.forward_with_hooks(ids, list(self.bases))
        acts = {l: np.asarray(a) for l, a in acts.items()}
        r = _readout_from_acts(acts, self.bases)
        r.prompt = prompt
        self.readouts.append(r)
        return r

    def run_battery(self, prompts, unit=None, progress=None):
        for i, p in enumerate(prompts):
            if unit is not None and unit.budget.exhausted():
                break
            self.readout(p)
            if unit is not None:
                unit.budget.step(1)
                unit.theta.note_learning(1)
            if progress:
                progress(i + 1, len(prompts), p)
        if unit is not None:
            unit.outcome_ok = len(self.readouts) > 0
        self.write_results()
        return self.readouts

    # -- the separate results file the report points to (C9) ---------------- #
    def results_path(self) -> Path:
        d = Path(self.lib.root) / "results"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"results_{self.model_hash}_{self.run_id}.json"

    def write_results(self) -> Path:
        payload = {
            "model_hash": self.model_hash, "run_id": self.run_id,
            "layers_mapped": sorted(self.bases),
            "readouts": [{
                "prompt": r.prompt,
                "ignition_layer": r.ignition_layer,
                "per_layer": [{"layer": lr.layer, "var_frac": lr.var_frac,
                               "top_concepts": lr.top_concepts} for lr in r.per_layer],
            } for r in self.readouts],
        }
        p = self.results_path()
        p.write_text(json.dumps(payload, indent=2))
        return p
