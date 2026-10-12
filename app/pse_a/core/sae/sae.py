"""SAE -- recnik pojmova, [SPEC] Sec 6, sa ispravkama I-12.

Zasto mrtvi feature-i eksplodiraju (§6 plana, faza 6.3): kod TopK SAE-a
feature koji nikad ne pobedi u top-k takmicenju dobija NULA gradijenta i
nikad se ne oporavi -- smrt je apsorbujuce stanje. Mere, po stvarnoj
delotvornosti:
  1. manji expansion na malim podacima (profil, ne kod)
  2. AuxK: pomocni gubitak koji top-k_aux MRTVIH feature-a tera da
     rekonstruisu ostatak greske -> ozivljavanje gradijentom, ne re-init
  3. resampling kako treba: dekoder na primer sa VELIKIM gubitkom, enkoder
     = transponovani dekoder, Adam stanje za te parametre RESETOVANO
  4. dekoder inicijalizovan iz podataka, jedinicne norme kolona
Sve je cista funkcija (AZ-1): podaci unutra, recnik i metrike napolje;
budzet dolazi kao povratni poziv.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SAEConfig:
    d: int
    expansion: int = 4
    k: int = 32
    k_aux: int = 256
    aux_coef: float = 1 / 32
    lr: float = 3e-4
    batch: int = 256
    steps: int = 5000
    dead_after: int = 2000          # koraka bez aktivacije -> mrtav
    resample_every: int = 1000
    seed: int = 0

    @property
    def m(self) -> int:
        return self.d * self.expansion


@dataclass
class SAEResult:
    W_enc: np.ndarray
    b_enc: np.ndarray
    W_dec: np.ndarray
    b_dec: np.ndarray
    fvu: float
    l0: float
    dead_frac: float
    history: list = field(default_factory=list)
    stopped: str = "budzet"


def encode_np(x: np.ndarray, W_enc, b_enc, b_dec, k: int):
    """TopK enkoder: (z, pre, idx). z je retka (n, m) matrica, ReLU na pobednicima."""
    pre = (x - b_dec) @ W_enc + b_enc
    idx = np.argpartition(-pre, k - 1, axis=-1)[:, :k]
    top = np.take_along_axis(pre, idx, axis=-1)
    z = np.zeros_like(pre)
    np.put_along_axis(z, idx, np.maximum(top, 0.0), axis=-1)
    return z, pre, idx


class _Adam:
    """Adam kao u torch.optim.Adam (bias korekcija, eps van korena)."""

    def __init__(self, params: dict, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            self.p[k] -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def _fvu_l0(sample_batch, P, cfg, n):
    xv = sample_batch(n).astype(np.float32)
    zv, _, _ = encode_np(xv, P["W_enc"], P["b_enc"], P["b_dec"], cfg.k)
    ev = xv - (zv @ P["W_dec"] + P["b_dec"])
    fvu = float((ev ** 2).sum() / ((xv - xv.mean(0)) ** 2).sum())
    return fvu, float((zv > 0).sum(-1).mean())


def train_sae(sample_batch, cfg: SAEConfig, init_data: np.ndarray,
              on_progress=None) -> SAEResult:
    """sample_batch(n) -> np.ndarray (n, d), vec normalizovan Welford statistikom.

    on_progress(step, metrics) -> "continue" | "stop"  ([PSE-A] 8.1: plato
    SKRACUJE zivot, divergencija ga produzava ili deli -- odluka nije ovde).

    Gradijenti su izvedeni rucno (numpy, bez torch-a); tests/test_sae_grad.py ih
    proverava konacnim razlikama.
    """
    rng = np.random.default_rng(cfg.seed)
    d, m = cfg.d, cfg.m
    f32 = np.float32

    # dekoder iz podataka, jedinicne norme; enkoder = transponovani dekoder
    idx = rng.choice(init_data.shape[0], m, replace=init_data.shape[0] < m)
    Wd = init_data[idx].astype(f32)
    Wd /= np.maximum(np.linalg.norm(Wd, axis=1, keepdims=True), 1e-8)
    P = {"W_dec": Wd.copy(), "W_enc": np.ascontiguousarray(Wd.T),
         "b_enc": np.zeros(m, f32), "b_dec": init_data.mean(0).astype(f32)}
    opt = _Adam(P, cfg.lr)
    last_fired = np.zeros(m, dtype=np.int64)
    hist = []
    stopped = "budzet"

    for step in range(1, cfg.steps + 1):
        x = sample_batch(cfg.batch).astype(f32)
        B = x.shape[0]
        xc = x - P["b_dec"]
        z, pre, idx = encode_np(x, P["W_enc"], P["b_enc"], P["b_dec"], cfg.k)
        err = x - (z @ P["W_dec"] + P["b_dec"])
        G = (-2.0 / B) * err                                 # dL/dx_hat
        gW_dec = z.T @ G
        gb_dec = G.sum(0)
        dz = G @ P["W_dec"].T
        dpre = np.zeros_like(pre)
        live = np.take_along_axis(pre, idx, axis=-1) > 0
        np.put_along_axis(dpre, idx, np.take_along_axis(dz, idx, axis=-1) * live, axis=-1)

        fired = (z > 0).any(0)
        last_fired[fired] = step
        dead = (step - last_fired) > cfg.dead_after
        if dead.any() and cfg.aux_coef > 0:                 # AuxK
            pre_d = np.where(dead[None, :], pre, -1e9).astype(f32)
            ka = int(min(cfg.k_aux, dead.sum()))
            ia = np.argpartition(-pre_d, ka - 1, axis=-1)[:, :ka]
            ta = np.take_along_axis(pre_d, ia, axis=-1)
            za = np.zeros_like(pre)
            np.put_along_axis(za, ia, np.maximum(ta, 0.0), axis=-1)
            ra = err - za @ P["W_dec"]                       # err je "detach"
            Ga = (-2.0 * cfg.aux_coef / B) * ra
            gW_dec += za.T @ Ga
            dza = Ga @ P["W_dec"].T
            np.add.at(dpre, (np.arange(B)[:, None], ia),
                      np.take_along_axis(dza, ia, axis=-1) * (ta > 0))
        gW_enc = xc.T @ dpre
        gb_enc = dpre.sum(0)
        gb_dec = gb_dec - (dpre @ P["W_enc"].T).sum(0)
        opt.step({"W_enc": gW_enc, "b_enc": gb_enc, "W_dec": gW_dec, "b_dec": gb_dec})
        P["W_dec"] /= np.maximum(np.linalg.norm(P["W_dec"], axis=1, keepdims=True), 1e-8)

        if step % cfg.resample_every == 0 and dead.any():
            _resample(P, opt, dead, sample_batch, cfg, rng)
            last_fired[dead] = step

        if step % 250 == 0 or step == cfg.steps:
            fvu, l0 = _fvu_l0(sample_batch, P, cfg, 2048)
            met = {"step": step, "fvu": fvu, "l0": l0,
                   "dead_frac": float(((step - last_fired) > cfg.dead_after).mean())}
            hist.append(met)
            if on_progress and on_progress(step, met) == "stop":
                stopped = "plato"
                break

    fvu, l0 = _fvu_l0(sample_batch, P, cfg, 4096)
    return SAEResult(P["W_enc"], P["b_enc"], P["W_dec"], P["b_dec"],
                     fvu, l0, float(((cfg.steps - last_fired) > cfg.dead_after).mean()),
                     hist, stopped)


def _resample(P, opt, dead, sample_batch, cfg, rng):
    """Mrtvi feature -> pravac PRIMERA SA VELIKIM GUBITKOM, i reset Adam stanja.

    Naivan nasumicni re-init ih samo ponovo ubije: nov pravac bez gradijenta u
    starom Adam momentu dobija korak skaliran tudjom istorijom.
    """
    x = sample_batch(8192).astype(np.float32)
    z, _, _ = encode_np(x, P["W_enc"], P["b_enc"], P["b_dec"], cfg.k)
    e = ((x - (z @ P["W_dec"] + P["b_dec"])) ** 2).sum(-1).astype(np.float64)
    p = e / e.sum()
    di = np.nonzero(dead)[0]
    pick = rng.choice(x.shape[0], len(di), replace=len(di) > x.shape[0], p=p)
    v = x[pick] - P["b_dec"]
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    P["W_dec"][di] = v
    alive_norm = (np.linalg.norm(P["W_enc"], axis=0)[~dead].mean()
                  if (~dead).any() else 1.0)
    P["W_enc"][:, di] = v.T * alive_norm * 0.2
    P["b_enc"][di] = 0.0
    for key in ("W_enc", "b_enc", "W_dec"):
        for st in (opt.m[key], opt.v[key]):
            if key == "W_enc":
                st[:, di] = 0.0
            else:
                st[di] = 0.0


def superposition_profile(res: SAEResult, sample_batch, cfg: SAEConfig,
                          window: int = 5, eps: float = 0.05, delta: float = 0.10,
                          community_tag: str = "", rel_thr: float = 0.5) -> dict:
    """[SPEC] Sec 6 export: §9 junction. SAE feature-i postaju stable_features,
    izmerene stope ko-aktivacije postaju epsilon_profile, efektivni broj
    feature-a postaje N_eff. Od prvog zavrsenog L3 run-a [PSE §9] prestaje da
    bude specifikacija i postaje merenje."""
    x = sample_batch(8192).astype(np.float32)
    Z, _, _ = encode_np(x, res.W_enc, res.b_enc, res.b_dec, cfg.k)
    # I-22: "aktivan" u jedinicama SOPSTVENE tipicne jacine feature-a. TopK sa k
    # iznad prave retkosti puni visak slotova sitnim aktivacijama koje prate
    # glavne; one su stvarne u kodu SAE-a, ali ne u podacima, i preplave profil
    # (izmereno: k=4 na retkosti 3.4 -> 1/173 nezavisnih "stabilno"). Prag po
    # feature-u = rel_thr × 90. percentil njegovih nenultih aktivacija.
    typ = np.full(Z.shape[1], np.inf)
    live = (Z > 0).any(0)
    if live.any():
        pos = np.where(Z[:, live] > 0, Z[:, live], np.nan)
        typ[live] = np.nanpercentile(pos, 90, axis=0)
    A = ((Z > 0) & (Z > rel_thr * typ[None, :])).astype(np.float32)
    freq = A.mean(0)
    alive = freq > 0
    # ko-aktivacija u prozoru od `window` uzastopnih uzoraka
    n = (A.shape[0] // window) * window
    W = A[:n].reshape(-1, window, A.shape[1]).max(1)
    co = (W.T @ W) / max(W.shape[0], 1)
    fw = W.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.where(fw[:, None] > 0, co / fw[:, None], 0.0)   # P(j | i)
    # I-21: VISAK iznad nezavisnosti, ne sirova stopa. U prozoru od `window`
    # uzoraka sa k aktivnih od m, SLUCAJNA stopa bilo kog para je vec ~k·window/m
    # (za 4, 5, 256 to je 0.078 > eps), pa bi apsolutni prag svaki feature
    # proglasio superponiranim. Peti put isti obrazac (I-15, 16, 17, 20).
    cond = cond - fw[None, :]
    np.fill_diagonal(cond, 0.0)
    # ...i ZNACAJNOST: max preko m-1 sumnih procena prelazi eps slucajno
    # (SE ~ sqrt(p(1-p)/n_i) ~ 0.024 za tipicne vrednosti). Par je superponiran
    # samo ako je visak i smislen (> eps) i znacajan (z > Bonferroni za m poredjenja).
    n_i = W.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(np.maximum(fw * (1 - fw), 1e-12)[None, :] / np.maximum(n_i, 1)[:, None])
        z = np.where(se > 0, cond / se, 0.0)
    from scipy.stats import norm
    z_crit = float(norm.isf(0.05 / max(A.shape[1], 2)))
    # I-23: normalna aproksimacija ne vazi na malim brojevima. Sa ~7 pojavljivanja
    # feature-a JEDNO zajednicko pojavljivanje daje P(j|i)=0.14 i z~3.7 -> "znacajno".
    # Zato: par se broji tek uz >= MIN_CO zajednickih pojavljivanja, a feature se
    # uopste ocenjuje tek uz >= MIN_N svojih. Ostali su "nedovoljno podataka" --
    # ni stabilni ni superponirani.
    MIN_N, MIN_CO = 20, 5
    counts = W.T @ W
    sig = (cond > eps) & (z > z_crit) & (counts >= MIN_CO)
    evaluable = alive & (n_i >= MIN_N)
    stable = [int(i) for i in np.nonzero(evaluable)[0] if not sig[i].any()]
    p = freq[alive] / freq[alive].sum()
    n_eff = float(np.exp(-(p * np.log(p)).sum())) if p.size else 0.0
    exceptional = float(sig[alive][:, alive].mean()) if alive.any() else 0.0
    return {"n_eff": n_eff, "n_alive": int(alive.sum()), "n_total": int(A.shape[1]),
            "stable_features": stable, "n_stable": len(stable), "z_crit": z_crit,
            "n_evaluable": int(evaluable.sum()),
            "epsilon": eps, "delta": delta, "exceptional_share": exceptional,
            "delta_stable": exceptional < delta, "community_tag": community_tag,
            "coactivation_top": _top_pairs(cond, 50)}


def _top_pairs(cond: np.ndarray, n: int) -> list:
    iu = np.dstack(np.unravel_index(np.argsort(-cond.ravel())[:n], cond.shape))[0]
    return [(int(i), int(j), float(cond[i, j])) for i, j in iu if cond[i, j] > 0]
