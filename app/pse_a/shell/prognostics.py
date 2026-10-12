"""Jedini emiter teksta prema coveku -- AZ-8 / [PSE-A] Sec 9.

"No game form is ever named to the human -- not in text, and not in the GUI."
Sloj formi vraca (form_id, posterior, horizon); ovde se to prevodi u
PROGNOSTICKI DOMENSKI JEZIK. Nijedan drugi modul ne sme da pravi recenice o
stanju jedinica. tests/test_az8.py grep-uje sve sto ovaj modul moze da emituje,
sve sablone i sve SSE seme, protiv zabranjenog recnika.
"""
from __future__ import annotations


def _runs(h: float | None) -> str:
    if h is None:
        return ""
    n = max(1, round(h))
    return {1: "about one run", 2: "about two runs",
            3: "about three runs"}.get(n, f"about {n} runs")


def prognosis(form_id: str, posterior: float, horizon: float | None = None) -> str:
    """(forma, posterior, horizont) -> recenica. Nesigurnost ostaje u recenici."""
    sure = "probably" if posterior >= 0.7 else "possibly"
    if form_id == "war_of_attrition":
        return (f"This direction is {sure} not converging; {_runs(horizon)} to a decision "
                "point — change the approach or stop.")
    if form_id == "signalling":
        return (f"These results {sure} look better than they can be verified; "
                "worth repeating them before they enter knowledge.")
    if form_id == "stag_hunt":
        return ("Trust in the analysis flow has been shaken; the log and reproduce-on-demand "
                "are available for every result.")
    if form_id == "iterated_pd":
        return "A unit near the end of its budget finishes early rather than cutting corners."
    if form_id == "ess":
        return "The default methodology does not change until enough runs are confirmed."
    return "Work is proceeding at its usual pace."


_GATE_TEXT = {
    "theta_u_opsegu": "the unit's engagement left the healthy range",
    "varijansa_u_granicama": "the unit's engagement was too unstable",
    "registar_ima_sadrzaj": "the run did not change any expectation about the model",
    "angazman_ne_izolacija": "the unit's engagement declined during the run",
}


def refusal(failed: list[str]) -> str:
    """Zasto rezultat nije primljen u znanje -- bez klinickih i strateskih imena."""
    parts = [_GATE_TEXT.get(f, "a check did not pass") for f in failed]
    return "The result was not accepted into knowledge: " + "; ".join(parts) + "."


def regime_label(regime: str) -> str:
    return {"random_like": "random-like", "spiked": "spiked directions",
            "heavy_tailed": "heavy tail", "mixed": "mixed"}.get(regime, regime)


RUN_KIND = {"l1_sweep": "spectral map", "qdiff": "precision comparison",
            "l2_lens": "thought by layer", "l3_sae": "concept dictionary",
            "l4_jspace": "J-space map", "ppl_ladder": "perplexity ladder",
            "cartography": "J-space cartography", "findings": "findings & conclusions"}


def run_kind_label(kind: str) -> str:
    return RUN_KIND.get(kind, kind)


def gate_reasons(refused: str) -> list[str]:
    """Iz poruke SealRefused izvuci kljuceve uslova koji nisu prosli."""
    return [k for k in _GATE_TEXT if k in refused]


def all_phrases() -> list[str]:
    """Sve sto modul moze da emituje -- za AZ-8 test."""
    out = [prognosis(f, p, h) for f in ("war_of_attrition", "signalling", "stag_hunt",
                                         "iterated_pd", "ess", "nash_substrate")
           for p in (0.5, 0.9) for h in (None, 1, 2, 5)]
    out += [refusal([k]) for k in _GATE_TEXT]
    out += [regime_label(r) for r in ("random_like", "spiked", "heavy_tailed", "mixed")]
    out += list(RUN_KIND.values())
    return out
