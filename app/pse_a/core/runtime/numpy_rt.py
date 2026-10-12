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
_BLOCK_MATS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


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


def rope_tables(spec: ArchSpec, n_pos: int, freq_factors: np.ndarray | None, start: int = 0):
    rd = spec.rope_dim
    inv = 1.0 / (spec.rope_base ** (np.arange(0, rd, 2, dtype=np.float64) / rd))
    if freq_factors is not None:
        inv = inv / freq_factors.astype(np.float64)[: inv.shape[0]]
    ang = np.arange(start, start + n_pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(F), np.sin(ang).astype(F)          # (T, rd/2)


def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, kind: str, rd: int,
               inverse: bool = False) -> np.ndarray:
    """x (..., T, hd). Rotacija je ortogonalna, pa je VJP ista rotacija sa -sin."""
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
        self.embed = T("token_embd.weight")                         # (vocab, d)
        self.W_U = self.embed if sp.tied else T("output.weight")
        self.final_norm_w = T("output_norm.weight")
        self.freq_factors = T("rope_freqs.weight") if sp.has_rope_freqs else None
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cache_max = self.n_layers if resident else 2
        self.scale = F(1.0 / np.sqrt(sp.head_dim))     # np.float64 bi tiho promovisao sve u fp64

    # -------------------------------------------------------------- tezine
    def block(self, L: int) -> dict:
        b = self._cache.get(L)
        if b is not None:
            self._cache.move_to_end(L)
            return b
        p = f"blk.{L}."
        T = lambda name: np.ascontiguousarray(dq.tensor(self.g, p + name), dtype=F)
        b = {k: T(k + ".weight") for k in _BLOCK_MATS}
        b["attn_norm"] = T("attn_norm.weight")
        b["ffn_norm"] = T("ffn_norm.weight")
        if self.spec.qkv_bias:
            for k in ("attn_q", "attn_k", "attn_v"):
                b[k + "_b"] = T(k + ".bias")
        if self.spec.qk_norm:
            b["q_norm"] = T("attn_q_norm.weight")
            b["k_norm"] = T("attn_k_norm.weight")
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

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        return rms(h, self.final_norm_w, self.spec.norm_eps)[0]

    # -------------------------------------------------------------- blok
    def _rope(self, T: int, start: int = 0):
        return rope_tables(self.spec, T, self.freq_factors, start)

    def _mask(self, T: int, S: int) -> np.ndarray:
        """Kauzalna (i sliding-window) maska za T novih upita nad S kljuceva."""
        q = np.arange(S - T, S)[:, None]
        k = np.arange(S)[None, :]
        bad = k > q
        if self.spec.sliding_window:
            bad |= k <= q - self.spec.sliding_window
        return np.where(bad, -np.inf, 0.0).astype(F)

    def _block_fwd(self, x: np.ndarray, b: dict, cos, sin, keep: bool, kv=None):
        sp = self.spec
        T = x.shape[0]
        H, KV, hd, rd = sp.n_head, sp.n_kv, sp.head_dim, sp.rope_dim
        h1, r1 = rms(x, b["attn_norm"], sp.norm_eps)
        q = h1 @ b["attn_q"].T
        k = h1 @ b["attn_k"].T
        v = h1 @ b["attn_v"].T
        if sp.qkv_bias:
            q = q + b["attn_q_b"]
            k = k + b["attn_k_b"]
            v = v + b["attn_v_b"]
        q = q.reshape(T, H, hd).transpose(1, 0, 2)                  # (H, T, hd)
        k = k.reshape(T, KV, hd).transpose(1, 0, 2)
        v = v.reshape(T, KV, hd).transpose(1, 0, 2)
        qn = kn = rq = rk = None
        if sp.qk_norm:
            qn, kn = q, k
            q, rq = rms(q, b["q_norm"], sp.norm_eps)
            k, rk = rms(k, b["k_norm"], sp.norm_eps)
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
        A = softmax(q @ kk.transpose(0, 2, 1) * self.scale + self._mask(T, S))
        o = (A @ vv).transpose(1, 0, 2).reshape(T, H * hd)
        x2 = x + o @ b["attn_output"].T
        h2, r2 = rms(x2, b["ffn_norm"], sp.norm_eps)
        gt = h2 @ b["ffn_gate"].T
        up = h2 @ b["ffn_up"].T
        sg = sigmoid(gt)
        x3 = x2 + (gt * sg * up) @ b["ffn_down"].T
        if not keep:
            return x3, None
        return x3, dict(x=x, r1=r1, q=q, kk=kk, vv=vv, A=A, qn=qn, kn=kn, rq=rq, rk=rk,
                        x2=x2, r2=r2, gt=gt, sg=sg, up=up)

    def _block_bwd(self, dx3: np.ndarray, b: dict, c: dict, cos, sin) -> np.ndarray:
        """dx3 (P, T, d) -> dx (P, T, d). Samo po rezidualnom toku."""
        sp = self.spec
        P, T, _ = dx3.shape
        H, KV, hd, rd, grp = sp.n_head, sp.n_kv, sp.head_dim, sp.rope_dim, sp.gqa_group
        # --- MLP: x3 = x2 + (silu(g)·u) Wd^T
        da = mm(dx3, b["ffn_down"])                                    # (P, T, ff)
        gt, sg, up = c["gt"], c["sg"], c["up"]
        du = da * (gt * sg)
        dg = da * up * (sg * (1.0 + gt * (1.0 - sg)))
        dh2 = mm(dg, b["ffn_gate"]) + mm(du, b["ffn_up"])
        del da, du, dg
        dx2 = dx3 + rms_bwd(dh2, c["x2"], c["r2"], b["ffn_norm"])
        del dh2
        # --- paznja: x2 = x + o Wo^T
        do = mm(dx2, b["attn_output"]).reshape(P, T, H, hd).transpose(0, 2, 1, 3)  # (P,H,T,hd)
        A, kk, vv, q = c["A"], c["kk"], c["vv"], c["q"]
        dA = do @ vv.transpose(0, 2, 1)[None]                       # (P, H, T, S)
        dvv = np.einsum("hts,phtd->phsd", A, do, optimize=True)
        dS = A[None] * (dA - (dA * A[None]).sum(-1, keepdims=True)) * self.scale
        del dA, do
        dq = dS @ kk[None]                                          # (P, H, T, hd)
        dkk = np.einsum("phts,htd->phsd", dS, q, optimize=True)
        del dS
        dk = dkk.reshape(P, KV, grp, T, hd).sum(2)
        dv = dvv.reshape(P, KV, grp, T, hd).sum(2)
        del dkk, dvv
        dq = rope_apply(dq, cos, sin, sp.rope, rd, inverse=True)
        dk = rope_apply(dk, cos, sin, sp.rope, rd, inverse=True)
        if sp.qk_norm:
            dq = rms_bwd(dq, c["qn"], c["rq"], b["q_norm"])
            dk = rms_bwd(dk, c["kn"], c["rk"], b["k_norm"])
        dq = dq.transpose(0, 2, 1, 3).reshape(P, T, H * hd)
        dk = dk.transpose(0, 2, 1, 3).reshape(P, T, KV * hd)
        dv = dv.transpose(0, 2, 1, 3).reshape(P, T, KV * hd)
        dh1 = mm(dq, b["attn_q"]) + mm(dk, b["attn_k"]) + mm(dv, b["attn_v"])
        return dx2 + rms_bwd(dh1, c["x"], c["r1"], b["attn_norm"])

    # -------------------------------------------------------------- [SPEC] Sec 2
    def run_resid(self, ids, start_layer: int = 0, x=None, stop_layer: int | None = None,
                  hook_points=None):
        """Rezidualni tok: resid_pre.L je ULAZ u blok L. Vraca (x_final, acts)."""
        ids = np.asarray(ids, dtype=np.int64)
        if x is None:
            x = self.embed[ids].astype(F)
        cos, sin = self._rope(x.shape[0])
        stop = self.n_layers if stop_layer is None else stop_layer
        acts = {}
        for L in range(start_layer, stop):
            if hook_points is None or f"resid_pre.{L}" in hook_points:
                acts[f"resid_pre.{L}"] = x
            x, _ = self._block_fwd(x, self.block(L), cos, sin, keep=False)
        acts["resid_final"] = x
        return x, acts

    def forward_with_hooks(self, tokens, hook_points=None) -> dict:
        return self.run_resid(tokens, hook_points=hook_points)[1]

    def logits(self, tokens) -> np.ndarray:
        x, _ = self.run_resid(tokens)
        return self.final_norm(x) @ self.W_U.T

    def vjp_multi(self, tokens, layers, cotangents, position: int = -1,
                  normed: bool = True, chunk: int = 64) -> dict:
        """{L: (P, T, d)} = d(U_p · out) / d resid_pre.L, out = [norm](h_final[pos]).

        [SPEC] Sec 7: pun Jakobijan se nikad ne materijalizuje. Kotangente se
        obradjuju u komadima od `chunk` da memorija ostane ogranicena.
        """
        ids = np.asarray(tokens, dtype=np.int64)
        layers = sorted(set(int(L) for L in layers))
        lo = layers[0]
        Tn = ids.shape[0]
        pos = position % Tn
        cos, sin = self._rope(Tn)
        x = self.embed[ids].astype(F)
        caches = {}
        for L in range(self.n_layers):
            x, c = self._block_fwd(x, self.block(L), cos, sin, keep=L >= lo)
            if c is not None:
                caches[L] = c
        U = np.asarray(cotangents, dtype=F)
        out = {L: [] for L in layers}
        for s in range(0, U.shape[0], chunk):
            Uc = U[s:s + chunk]
            dx = np.zeros((Uc.shape[0], Tn, self.d), dtype=F)
            if normed:
                xf = x[pos]
                _, r = rms(xf, self.final_norm_w, self.spec.norm_eps)
                dx[:, pos, :] = rms_bwd(Uc, xf, r, self.final_norm_w)
            else:
                dx[:, pos, :] = Uc
            for L in range(self.n_layers - 1, lo - 1, -1):
                dx = self._block_bwd(dx, self.block(L), caches[L], cos, sin)
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
            x = self.embed[np.asarray(cur, dtype=np.int64)].astype(F)
            T = x.shape[0]
            cos, sin = self._rope(T, start=pos)
            for L in range(self.n_layers):
                x, _ = self._block_fwd(x, self.block(L), cos, sin, keep=False, kv=kvs[L])
            pos += T
            lg = (self.final_norm(x[-1]) @ self.W_U.T).astype(np.float64)
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


def threads_hint() -> None:
    """Fizicka jezgra za BLAS, ako korisnik nije sam postavio (doctor upozorava)."""
    if "OMP_NUM_THREADS" not in os.environ:
        n = os.cpu_count() or 1
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, n)))
