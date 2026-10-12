"""AZ-8: nijedno ime forme i nijedan klinicki termin ni u jednom pikselu.

Grep-uje izvorni kod GUI-ja, sve sto prognostics.py moze da emituje, i SSE tipove
dogadjaja. Dozvoljen je arhitektonski recnik (unit, mission, budget,
absorption, fission, seal) -- Inbox po [PSE-A] 10.3 prikazuje bas to.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"\bnash\b", r"attrition", r"signall?ing", r"stag[ _]?hunt", r"\bchicken\b",
             r"iterated[_ ]?pd", r"prisoner", r"\bess\b", r"evolutionar", r"global[_ ]games",
             r"schizoid", r"shizoid", r"narcis", r"borderline", r"cluster[ _]?b",
             r"patholog", r"patolog", r"codependent", r"kodependent", r"defect"]


def scan_text(label: str, text: str) -> list[str]:
    bad = []
    low = text.lower()
    for pat in FORBIDDEN:
        for m in re.finditer(pat, low):
            ctx = low[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")
            bad.append(f"{label}: '{m.group(0)}' u …{ctx}…")
    return bad


def _sample_stories() -> list[str]:
    """Izgovorene recenice ljudskog registra na sintetickim artefaktima (radi i bez izvora)."""
    from pse_a.shell import narrative as n
    h = "0" * 64
    out = []
    out += n.l1({"model": "m", "kquant_clean": False, "matrices": [
        {"regime": "heavy_tailed", "alpha": 3.0, "role": "ffn_down", "mp_fit_ok": True},
        {"regime": "random_like", "alpha": 7.0, "role": "attn_v", "mp_fit_ok": False}]}, h)
    out += n.jspace({"layers": [1, 2], "prompts": ["a"], "per_layer": {"1": {"var_frac": .03},
                     "2": {"var_frac": .05}}, "regions": {"ignition": 2}}, h)
    out += n.cartography({"model": "m", "layers_mapped": [1, 2], "summary": {
        "n_prompts": 3, "mean_drift": .5, "max_drift": .9, "ignition_layers_seen": [2],
        "ignition_detected_fraction": .5, "n_clusters": 2}}, h)
    out += n.lens({"items": [{"prompt": "p", "top": [[["x"]]], "commit_layer": 3}]}, h)
    return out


def run() -> list[str]:
    bad = []
    app = ROOT.parent / "cartographer"            # GUI i tekstovi Cartographer-a
    for p in sorted(app.rglob("*.py")) if app.is_dir() else []:
        bad += scan_text(str(p.relative_to(ROOT.parent)), p.read_text(encoding="utf-8"))
    from pse_a.shell import prognostics
    for s in prognostics.all_phrases():
        bad += scan_text("prognostics", s)
    nar = ROOT / "shell" / "narrative.py"
    if nar.exists():                              # u zamrznutom paketu nema izvornog koda;
        bad += scan_text("narrative.py", nar.read_text(encoding="utf-8"))   # CI ga proverava
    bad += scan_text("narrative (sample)", "\n".join(_sample_stories()))
    bad += scan_text("psel.py SYSTEM", __import__("pse_a.shell.psel", fromlist=["x"]).SYSTEM)
    return bad


if __name__ == "__main__":
    bad = run()
    if bad:
        print("AZ-8 PREKRSEN:")
        for b in bad:
            print("  " + b)
        raise SystemExit(1)
    print("AZ-8 OK -- neither the GUI, prognostics nor narratives name a form or a diagnosis")
