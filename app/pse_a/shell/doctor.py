"""pse-a doctor -- AZ-2 capability semafor prosiren sa modela na SAMU MASINU.

Ovo je i alat kojim se .deb proverava posle prenosa na drugu masinu (§8.1).
Nista se ne pretpostavlja: BLAS i brzina se MERE.

Najvaznija zamka koju hvata: apt-ov python3-numpy se ume linkovati na
referentni libblas3 umesto na OpenBLAS. Razlika je ~60x i potpuno je tiha.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class MachineReport:
    cpu_model: str = "?"
    cores_physical: int = 0
    cores_logical: int = 0
    flags: list[str] = field(default_factory=list)
    ram_bytes: int = 0
    disk_free_bytes: int = 0
    data_root: str = ""
    blas_name: str = "?"
    gemm_gflops: float = 0.0
    eig_c: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def has_avx2(self) -> bool:
        return "avx2" in self.flags

    @property
    def has_fma(self) -> bool:
        return "fma" in self.flags


def data_root() -> str:
    """XDG. §8: koren podataka MORA biti konfigurabilan od prvog dana.

    Zamka iz stvarnog zivota: snap paketi (VS Code, Chromium...) postavljaju
    XDG_DATA_HOME unutar svog sandbox-a. Harvest od desetina GB tamo ne sme
    da zavrsi -- taj direktorijum nestaje sa revizijom snap-a.
    """
    env = os.environ.get("PSE_CARTOGRAPHER_DATA") or os.environ.get("PSE_A_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.name == "nt" and not os.environ.get("XDG_DATA_HOME"):
        return os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"),
                            "PSE Cartographer")
    base = os.environ.get("XDG_DATA_HOME") or ""
    if not base or "/snap/" in base or "/.snap" in base or not os.path.isabs(base):
        base = os.path.expanduser("~/.local/share")
    return os.path.join(base, "pse-cartographer")


def _cpu() -> tuple[str, int, int, list[str]]:
    model, flags, physical = (platform.processor() or "?"), [], set()
    logical = os.cpu_count() or 0
    try:
        with open("/proc/cpuinfo") as f:
            cur_phys = cur_core = None
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                elif line.startswith("flags") and not flags:
                    want = {"avx", "avx2", "fma", "avx512f", "sse4_2", "f16c", "bf16"}
                    flags = sorted(set(line.split(":", 1)[1].split()) & want)
                elif line.startswith("physical id"):
                    cur_phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    cur_core = line.split(":", 1)[1].strip()
                    if cur_phys is not None:
                        physical.add((cur_phys, cur_core))
    except OSError:
        pass
    return model, len(physical) or logical, logical, flags


def _ram() -> int:
    from ..core.runtime.numpy_rt import ram_bytes      # /proc/meminfo, ili Windows API
    return ram_bytes()


def _blas_name() -> str:
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts") or {}
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        return f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except Exception:
        return "?"


def probe(benchmark: bool = True) -> MachineReport:
    r = MachineReport()
    r.cpu_model, r.cores_physical, r.cores_logical, r.flags = _cpu()
    r.ram_bytes = _ram()
    r.data_root = data_root()
    probe_dir = r.data_root if os.path.isdir(r.data_root) else os.path.expanduser("~")
    r.disk_free_bytes = shutil.disk_usage(probe_dir).free
    r.blas_name = _blas_name()

    if benchmark:
        from ..core.cost import MachineRates
        rates = MachineRates.measure()
        r.gemm_gflops, r.eig_c = rates.gemm_gflops, rates.eig_c
        if r.gemm_gflops < 2.0:
            r.warnings.append(
                "BLAS IS NOT OPTIMIZED — this is ~60x slower than it should be, and silent.")

    nt = os.environ.get("OMP_NUM_THREADS")
    if r.cores_physical and nt and nt != str(r.cores_physical):
        r.warnings.append(
            f"OMP_NUM_THREADS={nt or 'not set'}; {r.cores_physical} (physical cores) is "
            f"recommended — hyperthreading hurts GEMM")
    if r.flags and not r.has_avx2:                    # bez /proc/cpuinfo flagovi su nepoznati
        r.warnings.append("CPU has no AVX2/FMA — everything works, but expect slower runs")
    return r


def rates_of(r: MachineReport):
    from ..core.cost import MachineRates
    return MachineRates(gemm_gflops=r.gemm_gflops or 1.0, eig_c=r.eig_c or 1e-9)


def select_profile(r: MachineReport):
    from ..profiles import select
    return select(r.gemm_gflops, r.ram_bytes / 1e9, True)


# ---------------------------------------------------------------- ispis
def _ok(b: bool) -> str:
    return "OK " if b else "-- "


def render(r: MachineReport, model_path: str | None = None) -> str:
    L = []
    A = L.append
    A("PSE Cartographer doctor")
    A("=" * 62)
    A(f"CPU      {r.cpu_model}")
    A(f"         {r.cores_physical}c/{r.cores_logical}t   {' '.join(r.flags) or '?'}")
    A(f"RAM      {r.ram_bytes/1e9:.1f} GB")
    A(f"disk     {r.disk_free_bytes/1e9:.0f} GB free in {r.data_root}")
    if r.gemm_gflops > 0:
        A(f"BLAS     {r.blas_name}   {r.gemm_gflops:.1f} GFLOPS fp32   "
          f"{_ok(r.gemm_gflops >= 2.0)}")
    else:
        A(f"BLAS     {r.blas_name}   speed not measured")
    A("runtime  numpy decoder (forward, hooks, hand-written VJP) — built in")
    prof = select_profile(r)
    A(f"profile  -> {prof.name}")

    if model_path:
        from ..core.source.gguf import open_gguf
        from ..core.source.canonical import build_map
        from ..core.capability import semaphore
        g = open_gguf(model_path)
        mm = build_map(g)
        A("")
        A(f"model    {os.path.basename(model_path)}")
        A(f"         {mm.summary()}")
        inv = g.type_inventory()
        tot = sum(v["elements"] for v in inv.values())
        A("         " + "  ".join(f"{k} {100*v['elements']/tot:.0f}%" for k, v in inv.items()))
        A(f"         model_hash {g.model_hash()[:16]}")
        caps = semaphore(g, mm, rates_of(r), r.ram_bytes, profile=prof)
        A("")
        A("available " + " ".join(k for k in ("L0", "L1", "L2", "L3", "L4")
                                  if caps.layers.get(k)))
        for k in ("L1", "L2", "L3", "L4"):
            if not caps.layers.get(k) and k in caps.blockers:
                A(f"         {k} unavailable: {caps.blockers[k]}")
        for n in caps.notes:
            A(f"         note: {n}")
        A("")
        A("estimates")
        for k, v in caps.estimates.items():
            A(f"         {k:52} {v}")

    if r.warnings:
        A("")
        A("warnings")
        for w in r.warnings:
            A(f"  !  {w}")
    return "\n".join(L)
