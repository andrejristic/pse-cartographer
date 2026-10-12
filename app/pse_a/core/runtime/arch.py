"""Arhitektura kao skup prekidaca -- jedan generički dekoder za mnoge porodice.

Gotovo svaki mali GGUF je varijacija istog pre-norm transformer bloka. Razlike
su u desetak prekidaca (vrsta norme, bias na QKV, QK-norm, vrsta RoPE-a, GQA,
aktivacija, vezani W_U...). ArchSpec ih cita iz GGUF metapodataka i iz samog
inventara tenzora, pa runtime ne zna ime porodice -- zna samo prekidace.

Faza 1 (testirano prema transformers-u): qwen2, qwen3, llama (Llama 2/3,
Mistral -- llama.cpp Mistral pise kao 'llama').

Detalji koji se tiho kvare ako se pogrese:
  - llama.cpp PERMUTUJE q/k za 'llama' pri konverziji, pa je RoPE tamo
    "normal" (susedni parovi); za qwen2/qwen3 nije permutovano -> NEOX (polovine)
  - Llama 3.1+ nosi rope_freqs.weight (faktori frekvencija) -- deli inv_freq
  - Qwen3 ima head_dim != d_model / n_head (attention.key_length)
  - W_U je token_embd kad output.weight ne postoji (vezano)
"""
from __future__ import annotations

from dataclasses import dataclass, field

# rope tip po arhitekturi (llama.cpp llama_model_rope_type)
ROPE_NORM = "norm"      # susedni parovi (x0,x1), (x2,x3) ...
ROPE_NEOX = "neox"      # polovine (x_i, x_{i+hd/2})

SUPPORTED_ARCHS = {
    # arch: (rope, aktivacija)
    "llama": (ROPE_NORM, "silu"),
    "mistral": (ROPE_NORM, "silu"),
    "qwen2": (ROPE_NEOX, "silu"),
    "qwen3": (ROPE_NEOX, "silu"),
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
    qk_norm: bool
    tied: bool
    has_rope_freqs: bool
    context: int
    sliding_window: int | None = None
    notes: list = field(default_factory=list)

    @property
    def gqa_group(self) -> int:
        return self.n_head // self.n_kv


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
    n_layers = int(_m(meta, arch, "block_count"))
    d = int(_m(meta, arch, "embedding_length"))
    n_head = int(_m(meta, arch, "attention.head_count"))
    n_kv = _m(meta, arch, "attention.head_count_kv", n_head)
    if isinstance(n_kv, list):
        raise UnsupportedModel("per-layer KV head counts are not supported")
    n_kv = int(n_kv)
    head_dim = int(_m(meta, arch, "attention.key_length", d // n_head))
    vh = int(_m(meta, arch, "attention.value_length", head_dim))
    if vh != head_dim:
        raise UnsupportedModel("key/value head sizes differ; not supported")
    d_ff = _m(meta, arch, "feed_forward_length")
    if isinstance(d_ff, list):
        raise UnsupportedModel("per-layer FFN widths are not supported")
    if f"blk.0.ffn_gate_inp.weight" in T:
        raise UnsupportedModel("mixture-of-experts models are not supported yet")
    for need in ("token_embd.weight", "output_norm.weight", "blk.0.attn_q.weight",
                 "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"):
        if need not in T:
            raise UnsupportedModel(f"expected tensor '{need}' is missing")
    vocab = int(T["token_embd.weight"].shape[1])
    eps = float(_m(meta, arch, "attention.layer_norm_rms_epsilon", 1e-6))
    rope_base = float(_m(meta, arch, "rope.freq_base", 10000.0))
    rope_dim = int(_m(meta, arch, "rope.dimension_count", head_dim))
    scaling = _m(meta, arch, "rope.scaling.type")
    notes = []
    if scaling not in (None, "none", "llama3"):
        if float(_m(meta, arch, "rope.scaling.factor", 1.0) or 1.0) != 1.0:
            notes.append(f"rope scaling '{scaling}' ignored (short prompts are unaffected)")
    sw = _m(meta, arch, "attention.sliding_window")
    return ArchSpec(
        arch=arch, n_layers=n_layers, d_model=d, n_head=n_head, n_kv=n_kv,
        head_dim=head_dim, d_ff=int(d_ff or T["blk.0.ffn_up.weight"].shape[1]),
        vocab=vocab, norm_eps=eps, rope=rope, rope_base=rope_base, rope_dim=rope_dim,
        act=act, qkv_bias="blk.0.attn_q.bias" in T,
        qk_norm="blk.0.attn_q_norm.weight" in T,
        tied="output.weight" not in T,
        has_rope_freqs="rope_freqs.weight" in T,
        context=int(_m(meta, arch, "context_length", 4096)),
        sliding_window=int(sw) if sw else None, notes=notes)
