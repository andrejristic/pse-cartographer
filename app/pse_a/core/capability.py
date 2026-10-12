"""Capability semafor -- [SPEC] Sec 2, prosiren vremenom (I-2).

    ModelSource only                 -> L0, L1
    + Runtime sa hook-ovima          -> + L2, L3
    + Runtime sa autograd-om         -> + L4

Runtime je numpy dekoder (core/runtime): hook-ovi i rucni VJP postoje za svaku
podrzanu arhitekturu, pa L2-L4 zavise samo od (a) arhitekture i (b) memorije.
Ako tezine u fp32 ne staju u budzet RAM-a, runtime radi "streaming" (blok po
blok dequantizacija) -- sporije, ali radi; semafor to kaze unapred.

Invarijanta AZ-2: nikad ne padaj usred run-a iz razloga koji je poznat na L0.
Prosirenje: procenjeno vreme JESTE takav razlog, pa ide u semafor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import MachineRates, l1_sweep_seconds, human
from .source.canonical import ModelMap
from .source.gguf import GGUFFile

MEM_FRACTION = 0.55          # tezine + aktivacije + OS
STREAM_SLOWDOWN = 3.0        # grubo: dequantizacija bloka pri svakom prolazu


@dataclass
class Capabilities:
    layers: dict[str, bool] = field(default_factory=dict)
    estimates: dict[str, str] = field(default_factory=dict)
    blockers: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    streaming: bool = False

    def available(self) -> list[str]:
        return [k for k, v in self.layers.items() if v]


def jspace_seconds(n_nonembed: int, n_prompts: int, n_probes: int, seq_len: int,
                   rates: MachineRates, streaming: bool = False) -> float:
    """Jedan forward + backward za P kotangenti po promptu.

    Backward po rezidualnom toku (bez gradijenata po tezinama) kosta otprilike
    koliko i forward, po kotangenti: 2·N·T·(1 + P) FLOP-a po promptu.
    Izmereno na i3-2330M (Qwen2.5-0.5B, T=10, P=64): procena 12 s, stvarno 12.4 s.
    """
    flops = 2.0 * n_nonembed * seq_len * (1 + n_probes) * n_prompts
    s = flops / (0.9 * rates.gemm_gflops * 1e9)
    return s * (STREAM_SLOWDOWN if streaming else 1.0)


def semaphore(g: GGUFFile, mm: ModelMap, rates: MachineRates, ram_bytes: int,
              n_prompts: int = 8, n_probes: int = 128, seq_len: int = 12,
              profile=None) -> Capabilities:
    from .runtime.arch import UnsupportedModel, spec_from_gguf
    from .source.dequant import SUPPORTED
    c = Capabilities()

    # ---- L0 / L1: dovoljan je ModelSource
    c.layers["L0"] = True
    shapes = [tuple(reversed(g.tensors[n].shape)) for n in mm.weight_matrices()]
    bad_types = sorted({g.tensors[n].type_name for n in g.tensors} - SUPPORTED)
    if bad_types:
        c.layers["L1"] = False
        c.blockers["L1"] = f"quantization type(s) {', '.join(bad_types)} cannot be read yet"
    else:
        c.layers["L1"] = True
        est = l1_sweep_seconds(shapes, rates, g._file_size, eigenvectors=False)
        c.estimates["L1 spectral sweep"] = f"{human(est['total_s'])}  ({est['n_matrices']} matrices)"

    # ---- L2-L4: arhitektura mora biti podrzana, tipovi citljivi
    try:
        spec = spec_from_gguf(g)
        c.notes += spec.notes
    except UnsupportedModel as e:
        for k in ("L2", "L3", "L4"):
            c.layers[k] = False
            c.blockers[k] = str(e)
        return c
    if bad_types:
        for k in ("L2", "L3", "L4"):
            c.layers[k] = False
            c.blockers[k] = c.blockers["L1"]
        return c

    n_params = sum(g.tensors[n].n_elements for n in g.tensors)
    fp32 = n_params * 4
    emb = g.tensors["token_embd.weight"].n_elements * 4 * (1 if "output.weight" not in g.tensors else 2)
    c.streaming = fp32 >= MEM_FRACTION * ram_bytes
    if c.streaming and emb + fp32 / spec.n_layers * 3 >= MEM_FRACTION * ram_bytes:
        for k in ("L2", "L3", "L4"):
            c.layers[k] = False
            c.blockers[k] = (f"even layer-by-layer the model needs more than "
                             f"{MEM_FRACTION * ram_bytes / 1e9:.1f} GB of RAM")
        return c
    for k in ("L2", "L3", "L4"):
        c.layers[k] = True
    if c.streaming:
        c.notes.append(f"weights in fp32 need {fp32 / 1e9:.1f} GB, RAM is {ram_bytes / 1e9:.1f} GB: "
                       f"layers are dequantized one at a time (works, ~{STREAM_SLOWDOWN:.0f}x slower)")
    n_nonembed = n_params - g.tensors["token_embd.weight"].n_elements - (
        g.tensors["output.weight"].n_elements if "output.weight" in g.tensors else 0)
    if spec.n_expert:            # MoE: each token passes through only k of E experts
        exp = sum(g.tensors[n].n_elements for n in g.tensors if "_exps." in n)
        n_nonembed -= int(exp * (1 - spec.n_expert_used / spec.n_expert))
        c.notes.append(f"mixture of experts: {spec.n_expert_used} of {spec.n_expert} experts per "
                       f"token; expert routing is mapped too")
    js = jspace_seconds(n_nonembed, n_prompts, n_probes, seq_len, rates, c.streaming)
    c.estimates[f"J-space cartography ({n_prompts} prompts, {n_probes} probes, all layers)"] = human(js)
    return c
