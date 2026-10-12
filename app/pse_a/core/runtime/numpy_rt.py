"""NumpyRuntime -- [SPEC] Sec 2 Runtime bez torch-a: forward, hook-ovi i VJP.

Zasto numpy, a ne torch: proizvod se isporucuje kao samostalan .deb/.exe od
~100 MB koji radi odmah, bez interneta, i na masinama bez AVX2 (gde je numpy sa
OpenBLAS-om izmereno ~2.5x brzi od torch-a, §13.10). Torch ostaje SAMO kao
nezavisan proverivac u razvojnim testovima (tests/oracle_*), nikad u paketu.

Sta runtime daje ([SPEC] Sec 2):
  forward_with_hooks(tokens, hook_points) -> {tacka: aktivacije}
  vjp_multi(tokens, layers, U, position)  -> {L: d(U·norm(h_final[pos])) / d h_L}
      JEDAN backward prolaz daje gradijente za SVE trazene slojeve -- mapa
      J-prostora preko svih slojeva kosta skoro koliko i za jedan sloj.
  generate(...)                           -> tokeni sa KV kesom (lokalni glas)

Backward je napisan rucno, operaciju po operaciju (RMSNorm, QK-norm, RoPE,
kauzalna paznja sa GQA, SwiGLU). Gradijent se trazi SAMO po rezidualnom toku,
nikad po tezinama. Ispravnost se proverava na dva nezavisna nacina: konacnim
razlikama na mini modelima (selftest, i u CI-ju) i prema torch autograd-u nad
transformers modelom (tests/oracle_*, lokalno).

Tezine (I-5 prosireno): ako model u fp32 staje u budzet RAM-a, slojevi se drze
u memoriji; ako ne staje, svaki blok se dequantizuje kad je na redu i odmah
oslobadja ("streaming"). Sporije, ali radi -- i semafor to kaze unapred.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..source.gguf import open_gguf
from ..source import dequant as dq
from .arch import ROPE_NEOX, ArchSpec, spec_from_gguf
from .tokenizer import GGUFTokenizer

F = np.float32


@dataclass
class Capabilities:
    has_hooks: bool = True
    has_autograd: bool = True       # rucni VJP: L4 radi
    device: str = "cpu"
    max_memory: int = 0
    resident: bool = True


def ram_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:                                       # Windows
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS()
        ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return int(ms.ullTotalPhys)
    except Exception:
        return 8 << 30


def fp32_footprint(g) -> int:
    return sum(t.n_elements for t in g.tensors.values()) * 4


# ------------------------------------------------------------------ primitivi
def rms(x: np.ndarray, w: np.ndarray | None, eps: float):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    y = x * r
    return (y * w if w is not None else y), r


def rms_bwd(dy: np.ndarray, x: np.ndarray, r: np.ndarray, w: np.ndarray | None):
    """y = x·r·w, r = (mean x² + eps)^(-1/2). dy moze imati vodecu P osu."""
    g = dy * w if w is not None else dy
    return r * g - x * (r ** 3) * (g * x).mean(-1, keepdims=True)


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    r = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    xh = xc * r
    y = xh * w if w is not None else xh
    return (y + b if b is not None else y), (xh, r)


def layer_norm_bwd(dy, cache, w):
    xh, r = cache
    g = dy * w if w is not None else dy
    return r * (g - g.mean(-1, keepdims=True) - xh * (g * xh).mean(-1, keepdims=True))


def rope_tables(spec: ArchSpec, n_pos: int, freq_factors: np.ndarray | None, start: int = 0,
                base: float | None = None, linear: float = 1.0):
    rd = spec.rope_dim
    base = spec.rope_base if base is None else base
    inv = 1.0 / (base ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    if freq_factors is not None:
        inv = inv / freq_factors.astype(np.float64)[: inv.shape[0]]
    pos = np.arange(start, start + n_pos, dtype=np.float64) / linear
    ang = pos[:, None] * inv[None, :]
    m = spec.rope_attn_factor
    return (np.cos(ang) * m).astype(F), (np.sin(ang) * m).astype(F)          # (T, rd/2)


def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, kind: str, rd: int,
               inverse: bool = False) -> np.ndarray:
    """x (..., T, hd); rotira prvih rd dimenzija. Bez faktora paznje (attn_factor = 1)
    rotacija je ortogonalna, pa je VJP ista rotacija sa -sin; sa faktorom m je VJP
    transponovana matrica: cos·m i -sin·m -- isti izraz."""
    s = -sin if inverse else sin
    out = x.copy()
    if kind == ROPE_NEOX:
        h = rd // 2
        x1, x2 = x[..., :h], x[..., h:rd]
        out[..., :h] = x1 * cos - x2 * s
        out[..., h:rd] = x2 * cos + x1 * s
    else:
        x1, x2 = x[..., 0:rd:2], x[..., 1:rd:2]
        out[..., 0:rd:2] = x1 * cos - x2 * s
        out[..., 1:rd:2] = x2 * cos + x1 * s
    return out


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


_GC = np.float32(np.sqrt(2.0 / np.pi))


def act_fwd(kind: str, g: np.ndarray):
    """(a, izvod) -- izvod se cuva za backward."""
    if kind == "silu":
        sg = sigmoid(g)
        return g * sg, sg * (1.0 + g * (1.0 - sg))
    u = _GC * (g + 0.044715 * g ** 3)                 # gelu, tanh aproksimacija
    t = np.tanh(u)
    a = 0.5 * g * (1.0 + t)
    da = 0.5 * (1.0 + t) + 0.5 * g * (1.0 - t * t) * _GC * (1.0 + 3 * 0.044715 * g * g)
    return a, da


def mm(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(..., k) @ (k, n) kao JEDNA 2-D matrica: numpy batch-matmul preko vodecih
    osa je ~3x sporiji od jednog velikog GEMM-a (izmereno na i3-2330M)."""
    return (x.reshape(-1, x.shape[-1]) @ W).reshape(x.shape[:-1] + (W.shape[-1],))


# ------------------------------------------------------------------ runtime
class NumpyRuntime:
    def __init__(self, gguf_path, n_layers: int | None = None, resident: bool | None = None,
                 mem_fraction: float = 0.55):
        self.path = str(gguf_path)
        self.g = open_gguf(gguf_path)
        self.spec = spec_from_gguf(self.g)
        sp = self.spec
        self.n_layers = sp.n_layers if n_layers is None else int(n_layers)
        self.d = sp.d_model
        self.tok = GGUFTokenizer(self.g.metadata)
        self.model_hash = self.g.model_hash()

        ram = ram_bytes()
        if resident is None:
            resident = fp32_footprint(self.g) < mem_fraction * ram
        self.caps = Capabilities(max_memory=ram, resident=bool(resident))

        T = lambda name: np.ascontiguousarray(dq.tensor(self.g, name), dtype=F)
        opt = lambda name: T(name) if name in self.g.tensors else None
        self.embed = T("token_embd.weight")                         # (vocab, d)
        self.W_U = self.embed if sp.tied else T("output.weight")
        self.out_bias = opt("output.bias")
        self.final_norm_w = T("output_norm.weight")
        self.final_norm_b = opt("output_norm.bias")
        self.freq_factors = T(sp.rope_factors) if sp.rope_factors else None
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_max = self.n_layers if resident else 2
        self.scale = F(sp.scale())                     # np.float64 bi tiho promovisao sve u fp64

    # -------------------------------------------------------------- tezine
    def block(self, L: int) -> dict:
        b = self._cache.get(L)
        if b is not None:
            self._cache.move_to_end(L)
            return b
        sp = self.spec
        p = f"blk.{L}."
        names = self.g.tensors
        T = lambda name: np.ascontiguousarray(dq.tensor(self.g, p + name), dtype=F)
        opt = lambda name: T(name) if p + name in names else None
        b = {}
        if sp.fused_qkv:
            qkv, bias = T("attn_qkv.weight"), opt("attn_qkv.bias")
            nq, nk = sp.n_head * sp.head_dim, sp.n_kv * sp.head_dim
            b["attn_q"], b["attn_k"], b["attn_v"] = qkv[:nq], qkv[nq:nq + nk], qkv[nq + nk:]
            b["attn_q_b"] = b["attn_k_b"] = b["attn_v_b"] = None
            if bias is not None:
                b["attn_q_b"], b["attn_k_b"], b["attn_v_b"] = bias[:nq], bias[nq:nq + nk], bias[nq + nk:]
        else:
            for k in ("attn_q", "attn_k", "attn_v"):
                b[k] = T(k + ".weight")
                b[k + "_b"] = opt(k + ".bias")
        b["attn_output"], b["attn_output_b"] = T("attn_output.weight"), opt("attn_output.bias")
        if sp.fused_gate_up:
            gu = T("ffn_up.weight")
            b["ffn_gate"], b["ffn_up"] = gu[:sp.d_ff], gu[sp.d_ff:]
            b["ffn_gate_b"] = b["ffn_up_b"] = None
        else:
            b["ffn_up"], b["ffn_up_b"] = T("ffn_up.weight"), opt("ffn_up.bias")
            if sp.mlp == "gated":
                b["ffn_gate"], b["ffn_gate_b"] = T("ffn_gate.weight"), opt("ffn_gate.bias")
        b["ffn_down"], b["ffn_down_b"] = T("ffn_down.weight"), opt("ffn_down.bias")
        for k in ("attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm"):
            b[k], b[k + "_b"] = opt(k + ".weight"), opt(k + ".bias")
        if sp.qk_norm:
            b["q_norm"], b["k_norm"] = T("attn_q_norm.weight"), T("attn_k_norm.weight")
        self._cache[L] = b
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return b

    # -------------------------------------------------------------- tokeni
    def encode(self, text: str, bos: bool | None = None) -> list[int]:
        return self.tok.encode(text, add_bos=bos)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))

    def token_str(self, i: int) -> str:
        return self.tok.decode([int(i)])

    # -------------------------------------------------------------- norme
    def _norm(self, x, w, b):
        if self.spec.norm == "layer":
            return layer_norm(x, w, b, self.spec.norm_eps)
        return rms(x, w, self.spec.norm_eps)

    def _norm_bwd(self, dy, x, cache, w):
        if self.spec.norm == "layer":
            return layer_norm_bwd(dy, cache, w)
        return rms_bwd(dy, x, cache, w)

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        return self._norm(h, self.final_norm_w, self.final_norm_b)[0]

    def unembed(self, z: np.ndarray) -> np.ndarray:
        """Normirano stanje -> logiti, sa svim zavrsnim transformacijama modela."""
        sp = self.spec
        lg = z @ self.W_U.T
        if self.out_bias is not None:
            lg = lg + self.out_bias
        if sp.logit_scale != 1.0:
            lg = lg / F(sp.logit_scale)
        if sp.final_softcap:
            lg = F(sp.final_softcap) * np.tanh(lg / F(sp.final_softcap))
        return lg

    # -------------------------------------------------------------- blok
    def _ropes(self, T: int, start: int = 0) -> dict:
        """{sloj: (cos, sin)} -- tabele se dele medju slojevima iste baze/faktora."""
        tabs, out = {}, {}
        for L in range(self.n_layers):
            key = self.spec.layer_rope(L)
            if key not in tabs:
                tabs[key] = rope_tables(self.spec, T, self.freq_factors, start, *key)
            out[L] = tabs[key]
        return out

    def _mask(self, T: int, S: int, window: int | None) -> np.ndarray:
        """Kauzalna (i sliding-window) maska za T novih upita nad S kljuceva."""
        q = np.arange(S - T, S)[:, None]
        k = np.arange(S)[None, :]
        bad = k > q
        if window:
            bad |= k <= q - window
        return np.where(bad, -np.inf, 0.0).astype(F)

    @staticmethod
    def _lin(x, W, bias):
        y = x @ W.T if x.ndim == 2 else mm(x, W.T)
        return y + bias if bias is not None else y

    def _attn_fwd(self, h, b, L, cos, sin, keep, kv):
        sp = self.spec
        T = h.shape[0]
        H, KV, hd, rd = sp.n_head, sp.n_kv, sp.head_dim, sp.rope_dim
        q = self._lin(h, b["attn_q"], b["attn_q_b"])
        k = self._lin(h, b["attn_k"], b["attn_k_b"])
        v = self._lin(h, b["attn_v"], b["attn_v_b"])
        c = {}
        if sp.qk_norm == "full":
            c["qf"], c["kf"] = q, k
            q, c["rqf"] = rms(q, b["q_norm"], sp.norm_eps)
            k, c["rkf"] = rms(k, b["k_norm"], sp.norm_eps)
        q = q.reshape(T, H, hd).transpose(1, 0, 2)                  # (H, T, hd)
        k = k.reshape(T, KV, hd).transpose(1, 0, 2)
        v = v.reshape(T, KV, hd).transpose(1, 0, 2)
        if sp.qk_norm == "head":
            c["qn"], c["kn"] = q, k
            q, c["rq"] = rms(q, b["q_norm"], sp.norm_eps)
            k, c["rk"] = rms(k, b["k_norm"], sp.norm_eps)
        q = rope_apply(q, cos, sin, sp.rope, rd)
        k = rope_apply(k, cos, sin, sp.rope, rd)
        if kv is not None:                                          # generisanje
            if kv.get("k") is not None:
                k = np.concatenate([kv["k"], k], axis=1)
                v = np.concatenate([kv["v"], v], axis=1)
            kv["k"], kv["v"] = k, v
        S = k.shape[1]
        kk = np.repeat(k, sp.gqa_group, axis=0)                     # (H, S, hd)
        vv = np.repeat(v, sp.gqa_group, axis=0)
        z = q @ kk.transpose(0, 2, 1) * self.scale
        if sp.attn_softcap:
            cap = F(sp.attn_softcap)
            tz = np.tanh(z / cap)
            z = cap * tz
            c["tz"] = tz
        A = softmax(z + self._mask(T, S, sp.layer_window(L)))
        o = (A @ vv).transpose(1, 0, 2).reshape(T, H * hd)
        out = self._lin(o, b["attn_output"], b["attn_output_b"])
        if keep:
            c.update(q=q, kk=kk, vv=vv, A=A)
        return out, c

    def _attn_bwd(self, dout, b, c, cos, sin):
        """dout (P, T, d) -> dh (P, T, d), po ulazu paznje."""
        sp = self.spec
        P, T, _ = dout.shape
        H, KV, hd, rd, grp = sp.n_head, sp.n_kv, sp.head_dim, sp.rope_dim, sp.gqa_group
        do = mm(dout, b["attn_output"]).reshape(P, T, H, hd).transpose(0, 2, 1, 3)  # (P,H,T,hd)
        A, kk, vv, q = c["A"], c["kk"], c["vv"], c["q"]
        dA = do @ vv.transpose(0, 2, 1)[None]                       # (P, H, T, S)
        dvv = np.einsum("hts,phtd->phsd", A, do, optimize=True)
        dS = A[None] * (dA - (dA * A[None]).sum(-1, keepdims=True))
        del dA, do
        if sp.attn_softcap:
            dS = dS * (1.0 - c["tz"] ** 2)[None]
        dS *= self.scale
        dq = dS @ kk[None]                                          # (P, H, T, hd)
        dkk = np.einsum("phts,htd->phsd", dS, q, optimize=True)
        del dS
        dk = dkk.reshape(P, KV, grp, T, hd).sum(2)
        dv = dvv.reshape(P, KV, grp, T, hd).sum(2)
        del dkk, dvv
        dq = rope_apply(dq, cos, sin, sp.rope, rd, inverse=True)
        dk = rope_apply(dk, cos, sin, sp.rope, rd, inverse=True)
        if sp.qk_norm == "head":
            dq = rms_bwd(dq, c["qn"], c["rq"], b["q_norm"])
            dk = rms_bwd(dk, c["kn"], c["rk"], b["k_norm"])
        dq = dq.transpose(0, 2, 1, 3).reshape(P, T, H * hd)
        dk = dk.transpose(0, 2, 1, 3).reshape(P, T, KV * hd)
        dv = dv.transpose(0, 2, 1, 3).reshape(P, T, KV * hd)
        if sp.qk_norm == "full":
            dq = rms_bwd(dq, c["qf"], c["rqf"], b["q_norm"])
            dk = rms_bwd(dk, c["kf"], c["rkf"], b["k_norm"])
        return mm(dq, b["attn_q"]) + mm(dk, b["attn_k"]) + mm(dv, b["attn_v"])

    def _mlp_fwd(self, h, b, keep):
        sp = self.spec
        if sp.mlp == "gated":
            gt = self._lin(h, b["ffn_gate"], b["ffn_gate_b"])
            up = self._lin(h, b["ffn_up"], b["ffn_up_b"])
            a, da = act_fwd(sp.act, gt)
            out = self._lin(a * up, b["ffn_down"], b["ffn_down_b"])
            return out, (dict(da=da, a=a, up=up) if keep else None)
        up = self._lin(h, b["ffn_up"], b["ffn_up_b"])
        a, da = act_fwd(sp.act, up)
        out = self._lin(a, b["ffn_down"], b["ffn_down_b"])
        return out, (dict(da=da) if keep else None)

    def _mlp_bwd(self, dout, b, c):
        dm = mm(dout, b["ffn_down"])                                 # (P, T, ff)
        if self.spec.mlp == "gated":
            dg = dm * c["up"] * c["da"]
            du = dm * c["a"]
            return mm(dg, b["ffn_gate"]) + mm(du, b["ffn_up"])
        return mm(dm * c["da"], b["ffn_up"])

    def _block_fwd(self, x: np.ndarray, b: dict, L: int, cos, sin, keep: bool, kv=None):
        sp = self.spec
        rs = F(sp.residual_scale)
        c = {"x": x}
        h1 = x
        if sp.norm_pre:
            h1, c["n1"] = self._norm(x, b["attn_norm"], b["attn_norm_b"])
        att, c["att"] = self._attn_fwd(h1, b, L, cos, sin, keep, kv)
        if sp.norm_post:
            c["att_raw"] = att
            att, c["pn1"] = self._norm(att, b["post_attention_norm"], b["post_attention_norm_b"])
        if sp.parallel:
            mo, c["mlp"] = self._mlp_fwd(h1, b, keep)
            x3 = x + rs * att + rs * mo
        else:
            x2 = x + rs * att
            c["x2"] = x2
            h2 = x2
            if sp.norm_pre:
                h2, c["n2"] = self._norm(x2, b["ffn_norm"], b["ffn_norm_b"])
            mo, c["mlp"] = self._mlp_fwd(h2, b, keep)
            if sp.norm_post:
                c["mlp_raw"] = mo
                mo, c["pn2"] = self._norm(mo, b["post_ffw_norm"], b["post_ffw_norm_b"])
            x3 = x2 + rs * mo
        return x3, (c if keep else None)

    def _block_bwd(self, dx3: np.ndarray, b: dict, c: dict, cos, sin) -> np.ndarray:
        """dx3 (P, T, d) -> dx (P, T, d). Samo po rezidualnom toku."""
        sp = self.spec
        rs = F(sp.residual_scale)
        if sp.parallel:
            dh1 = self._mlp_bwd(dx3 * rs, b, c["mlp"])
            datt = dx3 * rs
            if sp.norm_post:
                datt = self._norm_bwd(datt, c["att_raw"], c["pn1"], b["post_attention_norm"])
            dh1 = dh1 + self._attn_bwd(datt, b, c["att"], cos, sin)
            return dx3 + self._norm_bwd(dh1, c["x"], c["n1"], b["attn_norm"])
        dmo = dx3 * rs
        if sp.norm_post:
            dmo = self._norm_bwd(dmo, c["mlp_raw"], c["pn2"], b["post_ffw_norm"])
        dh2 = self._mlp_bwd(dmo, b, c["mlp"])
        dx2 = dx3 + (self._norm_bwd(dh2, c["x2"], c["n2"], b["ffn_norm"]) if sp.norm_pre else dh2)
        del dh2, dmo
        datt = dx2 * rs
        if sp.norm_post:
            datt = self._norm_bwd(datt, c["att_raw"], c["pn1"], b["post_attention_norm"])
        dh1 = self._attn_bwd(datt, b, c["att"], cos, sin)
        return dx2 + (self._norm_bwd(dh1, c["x"], c["n1"], b["attn_norm"]) if sp.norm_pre else dh1)

    # -------------------------------------------------------------- [SPEC] Sec 2
    def _embed(self, ids) -> np.ndarray:
        x = self.embed[np.asarray(ids, dtype=np.int64)].astype(F)
        return x * F(self.spec.embed_scale) if self.spec.embed_scale != 1.0 else x

    def run_resid(self, ids, start_layer: int = 0, x=None, stop_layer: int | None = None,
                  hook_points=None):
        """Rezidualni tok: resid_pre.L je ULAZ u blok L. Vraca (x_final, acts)."""
        if x is None:
            x = self._embed(ids)
        ropes = self._ropes(x.shape[0])
        stop = self.n_layers if stop_layer is None else stop_layer
        acts = {}
        for L in range(start_layer, stop):
            if hook_points is None or f"resid_pre.{L}" in hook_points:
                acts[f"resid_pre.{L}"] = x
            x, _ = self._block_fwd(x, self.block(L), L, *ropes[L], keep=False)
        acts["resid_final"] = x
        return x, acts

    def forward_with_hooks(self, tokens, hook_points=None) -> dict:
        return self.run_resid(tokens, hook_points=hook_points)[1]

    def logits(self, tokens) -> np.ndarray:
        x, _ = self.run_resid(tokens)
        return self.unembed(self.final_norm(x))

    def vjp_multi(self, tokens, layers, cotangents, position: int = -1,
                  normed: bool = True, chunk: int = 64) -> dict:
        """{L: (P, T, d)} = d(U_p · out) / d resid_pre.L, out = [norm](h_final[pos]).

        [SPEC] Sec 7: pun Jakobijan se nikad ne materijalizuje. Kotangente se
        obradjuju u komadima od `chunk` da memorija ostane ogranicena.
        """
        layers = sorted(set(int(L) for L in layers))
        lo = layers[0]
        x = self._embed(tokens)
        Tn = x.shape[0]
        pos = position % Tn
        ropes = self._ropes(Tn)
        caches = {}
        for L in range(self.n_layers):
            x, c = self._block_fwd(x, self.block(L), L, *ropes[L], keep=L >= lo)
            if c is not None:
                caches[L] = c
        U = np.asarray(cotangents, dtype=F)
        out = {L: [] for L in layers}
        for s in range(0, U.shape[0], chunk):
            Uc = U[s:s + chunk]
            dx = np.zeros((Uc.shape[0], Tn, self.d), dtype=F)
            if normed:
                xf = x[pos]
                _, cache = self._norm(xf, self.final_norm_w, self.final_norm_b)
                dx[:, pos, :] = self._norm_bwd(Uc, xf, cache, self.final_norm_w)
            else:
                dx[:, pos, :] = Uc
            for L in range(self.n_layers - 1, lo - 1, -1):
                dx = self._block_bwd(dx, self.block(L), caches[L], *ropes[L])
                if L in out:
                    out[L].append(dx)
        return {L: np.concatenate(v, axis=0) for L, v in out.items()}

    def vjp(self, tokens, to_layer: int, cotangents, position: int = -1,
            normed: bool = False, batched: bool = True) -> np.ndarray:
        """Kompatibilno sa [SPEC] potpisom: (P, T, d) za jedan sloj."""
        return self.vjp_multi(tokens, [to_layer], cotangents, position, normed)[to_layer]

    # -------------------------------------------------------------- generisanje
    def generate(self, ids, max_new: int = 256, temperature: float = 0.7, top_p: float = 0.9,
                 stop_ids: tuple = (), seed: int = 0):
        rng = np.random.default_rng(seed)
        kvs = [dict(k=None, v=None) for _ in range(self.n_layers)]
        pos, cur = 0, list(ids)
        for _ in range(max_new):
            x = self._embed(cur)
            T = x.shape[0]
            ropes = self._ropes(T, start=pos)
            for L in range(self.n_layers):
                x, _ = self._block_fwd(x, self.block(L), L, *ropes[L], keep=False, kv=kvs[L])
            pos += T
            lg = self.unembed(self.final_norm(x[-1])).astype(np.float64)
            if temperature <= 0:
                nxt = int(lg.argmax())
            else:
                p = softmax(lg / temperature)
                order = np.argsort(-p)
                ps = p[order]
                keep = np.cumsum(ps) - ps < top_p
                ps = ps * keep
                nxt = int(order[rng.choice(len(ps), p=ps / ps.sum())])
            if nxt in stop_ids:
                return
            yield nxt
            cur = [nxt]

    def chat_ids(self, system: str, turns: list) -> list[int]:
        return self.tok.chat_ids(system, turns)
