"""Arhitektura kao skup prekidaca -- jedan genericki dekoder za mnoge porodice.

Gotovo svaki mali GGUF je varijacija istog transformer bloka. Razlike su u
prekidacima koje ArchSpec cita iz GGUF metapodataka i iz samog inventara
tenzora, pa runtime ne zna ime porodice -- zna samo prekidace:

  norma              rms | layer (sa bias-om)
  polozaj norme      pre (llama) · pre + post "sendvic" (gemma2/3) · samo post (olmo2)
  rezidual           sekvencijalan | paralelan: x + attn(n(x)) + mlp(n(x)) (phi2, stablelm-12b)
  MLP                gated (act(gate)·up) | obican (act(up)); act = silu | gelu_tanh
  spojeni tenzori    attn_qkv (phi2/3) · ffn_up = [gate; up] (phi3)
  bias-i             po tenzoru: q/k/v, izlaz paznje, up/down, output
  QK-norma           nema | po glavi (qwen3, gemma3) | preko celog vektora (olmo2)
  RoPE               norm (susedni parovi) | neox (polovine); delimican (rope_dim < head_dim);
                     faktori frekvencija (rope_freqs, rope_factors_short); baza i linearno
                     skaliranje PO SLOJU (gemma3: lokalni slojevi 10k, globalni 1M)
  paznja             skala 1/sqrt(d) ili zadata; softcap logita (gemma2); sliding window PO
                     SLOJU (gemma2: parni slojevi, gemma3: 5 od 6)
  skaliranja         embedding (gemma sqrt(d), granite) · rezidual (granite) · logiti
                     (granite deli, gemma2 softcap)

Detalji koji se tiho kvare ako se pogrese:
  - llama.cpp PERMUTUJE q/k za "llama" (i granite) pri konverziji, pa je RoPE tamo
    "norm" (susedni parovi); qwen/gemma/phi/stablelm/olmo2 nisu permutovani -> neox
  - gemma GGUF vec nosi norme sa +1 (konverter ih doda), pa je norma obican RMS·w
  - Llama 3.1+ nosi rope_freqs.weight (faktori frekvencija) -- deli inv_freq
  - Qwen3/Gemma imaju head_dim != d_model / n_head (attention.key_length)
  - W_U je token_embd kad output.weight ne postoji (vezano)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ROPE_NORM = "norm"      # susedni parovi (x0,x1), (x2,x3) ...
ROPE_NEOX = "neox"      # polovine (x_i, x_{i+hd/2})

# arch -> (rope, aktivacija); ostalo se cita iz tenzora i metapodataka
SUPPORTED_ARCHS = {
    "llama": (ROPE_NORM, "silu"),
    "mistral": (ROPE_NORM, "silu"),
    "granite": (ROPE_NORM, "silu"),
    "qwen2": (ROPE_NEOX, "silu"),
    "qwen3": (ROPE_NEOX, "silu"),
    "gemma": (ROPE_NEOX, "gelu_tanh"),
    "gemma2": (ROPE_NEOX, "gelu_tanh"),
    "gemma3": (ROPE_NEOX, "gelu_tanh"),
    "phi3": (ROPE_NEOX, "silu"),
    "phi2": (ROPE_NEOX, "gelu_tanh"),
    "stablelm": (ROPE_NEOX, "silu"),
    "olmo2": (ROPE_NEOX, "silu"),
    "granitemoe": (ROPE_NORM, "silu"),
    "olmoe": (ROPE_NEOX, "silu"),      # izmereno: perplexity 16 (neox) vs 191 (norm)
}


class UnsupportedModel(Exception):
    """Poznato pri otvaranju (AZ-2): kaze se odmah, ne usred run-a."""


@dataclass
class ArchSpec:
    arch: str
    n_layers: int
    d_model: int
    n_head: int
    n_kv: int
    head_dim: int
    d_ff: int
    vocab: int
    norm_eps: float
    rope: str
    rope_base: float
    rope_dim: int
    act: str
    qkv_bias: bool
    qk_norm: str                 # "" | "head" | "full"
    tied: bool
    has_rope_freqs: bool
    context: int
    sliding_window: int | None = None
    notes: list = field(default_factory=list)
    # --- faza 2 prekidaci
    norm: str = "rms"            # rms | layer
    norm_pre: bool = True
    norm_post: bool = False
    parallel: bool = False
    mlp: str = "gated"           # gated | plain
    fused_qkv: bool = False
    fused_gate_up: bool = False
    attn_scale: float = 0.0      # 0 -> 1/sqrt(head_dim)
    attn_softcap: float = 0.0
    final_softcap: float = 0.0
    embed_scale: float = 1.0
    residual_scale: float = 1.0
    logit_scale: float = 1.0
    rope_factors: str = ""       # ime tenzora sa faktorima frekvencija
    rope_attn_factor: float = 1.0
    swa_layers: frozenset = frozenset()
    rope_local_base: float = 0.0  # >0: SWA slojevi koriste ovu bazu (gemma3)
    rope_global_factor: float = 1.0
    # --- MoE (faza 3)
    n_expert: int = 0             # 0 = gust model
    n_expert_used: int = 0
    expert_norm: bool = True      # ponovo normirati top-k tezine (granite da, olmoe ne)
    shared_expert: bool = False

    @property
    def gqa_group(self) -> int:
        return self.n_head // self.n_kv

    def layer_window(self, L: int) -> int | None:
        if not self.sliding_window:
            return None
        if self.swa_layers:
            return self.sliding_window if L in self.swa_layers else None
        return self.sliding_window

    def layer_rope(self, L: int) -> tuple[float, float]:
        """(baza, linearni faktor) RoPE-a za sloj L."""
        if self.rope_local_base and L in self.swa_layers:
            return self.rope_local_base, 1.0
        return self.rope_base, self.rope_global_factor

    def scale(self) -> float:
        return self.attn_scale or 1.0 / math.sqrt(self.head_dim)


def _m(meta: dict, arch: str, key: str, default=None):
    return meta.get(f"{arch}.{key}", default)


def spec_from_gguf(g) -> ArchSpec:
    meta, T = g.metadata, g.tensors
    arch = str(meta.get("general.architecture", "unknown"))
    if arch not in SUPPORTED_ARCHS:
        raise UnsupportedModel(
            f"architecture '{arch}' is not supported yet (supported: "
            f"{', '.join(sorted(SUPPORTED_ARCHS))})")
    rope, act = SUPPORTED_ARCHS[arch]
    has = lambda n: n in T
    n_layers = int(_m(meta, arch, "block_count"))
    d = int(_m(meta, arch, "embedding_length"))
    n_head = _m(meta, arch, "attention.head_count")
    n_kv = _m(meta, arch, "attention.head_count_kv", n_head)
    if isinstance(n_head, list) or isinstance(n_kv, list):
        raise UnsupportedModel("per-layer attention head counts are not supported")
    n_head, n_kv = int(n_head), int(n_kv)
    head_dim = int(_m(meta, arch, "attention.key_length", d // n_head))
    vh = int(_m(meta, arch, "attention.value_length", head_dim))
    if vh != head_dim:
        raise UnsupportedModel("key/value head sizes differ; not supported")
    d_ff = _m(meta, arch, "feed_forward_length")
    if isinstance(d_ff, list):
        raise UnsupportedModel("per-layer FFN widths are not supported")
    moe = has("blk.0.ffn_gate_inp.weight")
    if moe and not (has("blk.0.ffn_gate_exps.weight") and has("blk.0.ffn_up_exps.weight")):
        raise UnsupportedModel("this mixture-of-experts layout is not supported")
    fused_qkv = has("blk.0.attn_qkv.weight")
    ffn = "_exps" if moe else ""
    need = ["token_embd.weight", "output_norm.weight", f"blk.0.ffn_up{ffn}.weight",
            f"blk.0.ffn_down{ffn}.weight", "blk.0.attn_output.weight"]
    need += [] if fused_qkv else ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"]
    for n in need:
        if not has(n):
            raise UnsupportedModel(f"expected tensor '{n}' is missing")

    gated = has("blk.0.ffn_gate.weight") or moe
    up_rows = T[f"blk.0.ffn_up{ffn}.weight"].shape[1]
    fused_gate_up = (not gated) and arch == "phi3"
    if moe:
        d_ff = _m(meta, arch, "expert_feed_forward_length") or up_rows
    if d_ff is None:
        d_ff = up_rows // (2 if fused_gate_up else 1)
    d_ff = int(d_ff)
    norm = "layer" if has("output_norm.bias") else "rms"
    norm_pre = has("blk.0.attn_norm.weight")
    norm_post = has("blk.0.post_attention_norm.weight")
    parallel = norm_pre and not has("blk.0.ffn_norm.weight")
    if not norm_pre and not norm_post:
        raise UnsupportedModel("block has neither pre- nor post-norms")

    qk_norm = ""
    if has("blk.0.attn_q_norm.weight"):
        n_q = T["blk.0.attn_q_norm.weight"].n_elements
        if n_q == head_dim:
            qk_norm = "head"
        elif n_q == n_head * head_dim:
            qk_norm = "full"
        else:
            raise UnsupportedModel("per-head QK layer norms are not supported yet")
        if norm == "layer":
            raise UnsupportedModel("QK layer norms are not supported yet")

    eps_key = "attention.layer_norm_epsilon" if norm == "layer" else "attention.layer_norm_rms_epsilon"
    eps = float(_m(meta, arch, eps_key, _m(meta, arch, "attention.layer_norm_rms_epsilon", 1e-5)))
    rope_base = float(_m(meta, arch, "rope.freq_base", 10000.0))
    rope_dim = int(_m(meta, arch, "rope.dimension_count", head_dim))
    sw = _m(meta, arch, "attention.sliding_window")
    notes = []
    s = ArchSpec(
        arch=arch, n_layers=n_layers, d_model=d, n_head=n_head, n_kv=n_kv,
        head_dim=head_dim, d_ff=d_ff, vocab=int(T["token_embd.weight"].shape[1]),
        norm_eps=eps, rope=rope, rope_base=rope_base, rope_dim=rope_dim, act=act,
        qkv_bias=has("blk.0.attn_q.bias") or has("blk.0.attn_qkv.bias"),
        qk_norm=qk_norm, tied="output.weight" not in T,
        has_rope_freqs=has("rope_freqs.weight"),
        context=int(_m(meta, arch, "context_length", 4096)),
        sliding_window=int(sw) if sw else None, notes=notes,
        norm=norm, norm_pre=norm_pre, norm_post=norm_post, parallel=parallel,
        mlp="gated" if (gated or fused_gate_up) else "plain",
        fused_qkv=fused_qkv, fused_gate_up=fused_gate_up)
    if has("rope_freqs.weight"):
        s.rope_factors = "rope_freqs.weight"
    if moe:
        s.n_expert = int(_m(meta, arch, "expert_count", T["blk.0.ffn_gate_inp.weight"].shape[1]))
        s.n_expert_used = int(_m(meta, arch, "expert_used_count", 2))
        s.expert_norm = bool(_m(meta, arch, "expert_weights_norm", arch != "olmoe"))
        s.shared_expert = has("blk.0.ffn_up_shexp.weight")

    scaling = _m(meta, arch, "rope.scaling.type")
    factor = float(_m(meta, arch, "rope.scaling.factor", 1.0) or 1.0)

    # ---- porodicne posebnosti (iz llama.cpp-a; vrednosti iz metapodataka gde postoje)
    if arch in ("gemma", "gemma2", "gemma3"):
        s.embed_scale = math.sqrt(d)
        if arch == "gemma2":
            s.attn_softcap = float(_m(meta, arch, "attn_logit_softcapping", 0.0) or 0.0)
            s.final_softcap = float(_m(meta, arch, "final_logit_softcapping", 0.0) or 0.0)
            s.swa_layers = frozenset(L for L in range(n_layers) if L % 2 == 0)
            if d == 4608:                                  # 27B: skala po d/n_head
                s.attn_scale = 1.0 / math.sqrt(d / n_head)
        if arch == "gemma3":
            pattern = int(_m(meta, arch, "attention.sliding_window_pattern", 6) or 6)
            s.swa_layers = frozenset(L for L in range(n_layers) if (L + 1) % pattern != 0)
            s.rope_local_base = 10000.0
            if scaling == "linear" and factor != 1.0:
                s.rope_global_factor = factor
            if d == 5376:                                  # 27B: query_pre_attn_scalar = d/n_head
                s.attn_scale = 1.0 / math.sqrt(d / n_head)
    elif arch == "phi3":
        if has("rope_factors_short.weight"):
            s.rope_factors = "rope_factors_short.weight"
            s.rope_attn_factor = float(_m(meta, arch, "rope.scaling.attn_factor", 1.0) or 1.0)
            notes.append("long-context RoPE: short-range factors are used (prompts here are short)")
    elif arch in ("granite", "granitemoe"):
        s.embed_scale = float(_m(meta, arch, "embedding_scale", 1.0) or 1.0)
        s.residual_scale = float(_m(meta, arch, "residual_scale", 1.0) or 1.0)
        s.logit_scale = float(_m(meta, arch, "logit_scale", 1.0) or 1.0)
        a = float(_m(meta, arch, "attention.scale", 0.0) or 0.0)
        if a:
            s.attn_scale = a
    elif arch == "stablelm" and qk_norm:
        raise UnsupportedModel("StableLM with QK norms (12B) is not supported yet")

    if scaling not in (None, "none", "llama3") and arch != "gemma3" and factor != 1.0:
        notes.append(f"rope scaling '{scaling}' ignored (short prompts are unaffected)")
    if rope_dim % 2 or rope_dim > head_dim:
        raise UnsupportedModel(f"rope dimension {rope_dim} does not fit head size {head_dim}")
    return s
