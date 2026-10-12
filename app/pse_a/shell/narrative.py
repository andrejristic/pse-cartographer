"""Ljudski registar -- AZ-7, [SPEC] 11.2.

"Every result tells two stories": matematicka (paneli) i ljudska (ovde).
Obe iz ISTIH artefakata; svaka tvrdnja nosi hes artefakta iz kog dolazi, pa
se dva registra ne mogu razici. Generator je DETERMINISTICKI: isti zapecaceni
artefakti -> isti tekst, pa je prica nepromenljiva koliko i pecat, i racuna se
bez jezickog modela (nikakav LLM poziv po otvaranju stranice, §7.4).

Pravila [PSE §5] vaze: domenski jezik, nesigurnost u samoj recenici,
bez imena formi i dijagnoza (AZ-8 test prolazi i kroz ovaj modul),
ograniceno trajanje po odeljku.
"""
from __future__ import annotations

import math


def _c(h: str) -> str:
    return f"[{h[:8]}]"


def _pct(x: float, nd: int = 1) -> str:
    return f"{100*x:.{nd}f} %"


def l1(rep: dict, h: str) -> list[str]:
    M = rep["matrices"]
    reg = {}
    for m in M:
        reg[m["regime"]] = reg.get(m["regime"], 0) + 1
    ht = reg.get("heavy_tailed", 0) + reg.get("mixed", 0)
    al = [m["alpha"] for m in M if m.get("alpha") is not None and math.isfinite(m["alpha"])]
    by_role = {}
    for m in M:
        if m.get("alpha") is not None and math.isfinite(m["alpha"]):
            by_role.setdefault(m["role"], []).append(m["alpha"])
    roles = sorted(((sum(v) / len(v), r) for r, v in by_role.items()))
    mp_ok = sum(1 for m in M if m.get("mp_fit_ok"))
    out = [f"{len(M)} weight matrices of {rep['model']} were examined {_c(h)}."]
    out.append(f"In {ht} of them the spectrum has a heavy tail, meaning training wrote "
               f"structure into them; the rest are closer to random {_c(h)}.")
    if roles:
        out.append(f"By mean alpha, the strongest structure is in {roles[0][1]} (≈{roles[0][0]:.1f}) "
                   f"and the weakest in {roles[-1][1]} (≈{roles[-1][0]:.1f}); the smaller the alpha, "
                   f"the more pronounced the structure {_c(h)}.")
    if "mp_fit_ok" in M[0]:
        out.append(f"In {mp_ok} of {len(M)} matrices the bulk of the spectrum really does look like "
                   f"noise of one strength; for the others that picture does not hold, so only alpha "
                   f"is relied on {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("Note: this model's dimensions are not divisible by 256, so its quantized "
                   "versions differ in composition, not only in precision.")
    return out


def qdiff(rep: dict, h: str) -> list[str]:
    S = rep["strata"]
    worst = min(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    best = max(S.items(), key=lambda kv: kv[1]["kept_quarter"])
    conc = [k.split("|")[0] for k, v in S.items() if v["noise_ratio"] > 1.2]
    dE = sum(v["energy_change"] * v["n"] for v in S.values()) / max(sum(v["n"] for v in S.values()), 1)
    out = [f"Two precisions of the same model were compared: {rep['ref']} and {rep['test']} {_c(h)}."]
    lab = lambda k: "{} ({})".format(k.split("|")[0], k.split("|")[1].replace("->", "→"))
    out.append(f"The main directions of the matrices are mostly preserved: most is kept in "
               f"{lab(best[0])}, {best[1]['kept_quarter']:.3f}, and least in "
               f"{lab(worst[0])}, {worst[1]['kept_quarter']:.3f} {_c(h)}.")
    out.append(f"Total weight energy changes by {_pct(dE, 2)} on average — quantization does not "
               f"move it, it slightly reduces it {_c(h)}.")
    if conc:
        out.append(f"In {', '.join(sorted(set(conc)))} the quantization error falls on learned "
                   f"directions more often than on the rest {_c(h)}.")
    out.append("Whether this shows in the model's behaviour the spectrum alone cannot say; "
               "that needs the model's output measured on text.")
    return out


def lens(rep: dict, h: str) -> list[str]:
    it = rep["items"]
    out = [f"For {len(it)} short sentences we looked at what the model would say if forced to "
           f"speak at each layer {_c(h)}."]
    for x in it[:3]:
        out.append(f"“{x['prompt']}” — the final choice “{x['top'][-1][-1][0]}” first appears "
                   f"at layer {x['commit_layer']} {_c(h)}.")
    out.append("This view only sees what is already sayable; computation that is not yet in "
               "words stays hidden from it.")
    return out


def jspace(m: dict, h: str) -> list[str]:
    pl = m["per_layer"]
    vs = [pl[str(L)]["var_frac"] for L in m["layers"]]
    out = [f"We measured how much of its internal state the model holds in a ready-to-say form, "
           f"over {len(m['layers'])} layers and {len(m['prompts'])} deliberately ambiguous sentences {_c(h)}."]
    out.append(f"That share is between {_pct(min(vs))} and {_pct(max(vs))} {_c(h)}.")
    r = m["regions"]
    if r.get("ignition") is not None:
        out.append(f"At layer {r['ignition']} the model sharply commits to one reading {_c(h)}.")
    elif r.get("resolution_ok") is False:
        out.append(f"No sharp moment of commitment was seen, but the map is coarse (every "
                   f"{r.get('stride')}th layer), so this is not proof that there is none {_c(h)}.")
    else:
        out.append(f"No sharp moment of commitment was seen; for this model the split into a sensory "
                   f"part, a workspace and an output part is not confirmed {_c(h)}.")
    return out


def sae(rep: dict, h: str) -> list[str]:
    p = rep["superposition_profile"]
    out = [f"From {rep['n_tokens']} positions at layer {rep['layer']} a dictionary of "
           f"{rep['n_features']} concepts was learned {_c(h)}."]
    out.append(f"The dictionary reconstructs the state with {_pct(rep['fvu'])} error left, using about "
               f"{rep['l0']:.0f} active concepts per position {_c(h)}.")
    if "n_evaluable" in p:
        out.append(f"{p['n_evaluable']} of {p['n_alive']} live concepts occur often enough to "
                   f"judge; of those, {p['n_stable']} do not occur with others more often than "
                   f"chance would give {_c(h)}.")
    else:
        out.append(f"Of {p['n_alive']} live concepts, {p['n_stable']} do not occur with others "
                   f"more often than chance would give {_c(h)}.")
    if not rep.get("accepted", True):
        out.append("The dictionary did not pass acceptance, so it did not enter knowledge.")
    return out


def ppl(rep: dict, h: str) -> list[str]:
    R = rep["rows"]
    out = [f"The same {rep['n_tokens']}-token text was given to each of {len(R)} versions of the model {_c(h)}."]
    worst = max(R[1:], key=lambda r: r["ppl_rel"]) if len(R) > 1 else None
    if worst:
        out.append(f"{worst['file']} suffers most: perplexity rises by {_pct(worst['ppl_rel'])} "
                   f"over the reference {_c(h)}.")
    both = [r for r in R[1:] if "kept_quarter_mean" in r]
    if len(both) >= 2:
        both.sort(key=lambda r: 1 - r["kept_quarter_mean"])
        same = [r["ppl_rel"] for r in both] == sorted(r["ppl_rel"] for r in both)
        out.append("The ranking by lost main directions " + ("matches" if same else "does not match")
                   + f" the ranking by output degradation, over {len(both)} compared versions {_c(h)}.")
    if not rep.get("kquant_clean", True):
        out.append("This model's versions differ in composition, not only in precision, so this "
                   "is a comparison of files, not a clean curve by bit width.")
    return out


def cartography(m: dict, h: str) -> list[str]:
    s = m.get("summary", {})
    out = [f"{s.get('n_prompts', 0)} prompts were fed to {m.get('model', 'the model')} and its "
           f"J-space was read at {len(m.get('layers_mapped', []))} layers {_c(h)}."]
    out.append(f"Between consecutive prompts the concepts it holds move by {s.get('mean_drift', 0):.2f} "
               f"on average (0 = same, 1 = completely different), at most {s.get('max_drift', 0):.2f} {_c(h)}.")
    ign = s.get("ignition_layers_seen", [])
    if ign:
        out.append(f"A sharp commitment appears at layer(s) {ign}, in "
                   f"{100 * s.get('ignition_detected_fraction', 0):.0f} % of prompts {_c(h)}.")
    else:
        out.append(f"No sharp commitment was seen for these prompts {_c(h)}.")
    out.append(f"By the concepts they light up, the prompts form {s.get('n_clusters', 0)} group(s) {_c(h)}.")
    rs = s.get("routing")
    if rs:
        out.append(f"This model routes each token to {rs['experts_per_token']} of {rs['n_experts']} "
                   f"experts; the experts chosen change by {rs['mean_routing_drift']:.2f} on average "
                   f"between prompts {_c(h)}.")
    return out


GEN = {"cartography": ("cartography.json", cartography), "l1_sweep": ("l1_report.json", l1), "qdiff": ("qdiff_report.json", qdiff),
       "l2_lens": ("lens_report.json", lens), "l4_jspace": ("jspace_map.json", jspace),
       "l3_sae": ("sae_report.json", sae), "ppl_ladder": ("ppl_ladder.json", ppl)}


def story(kind: str, artifacts: dict, load) -> list[str]:
    if kind not in GEN:
        return []
    name, fn = GEN[kind]
    h = artifacts.get(name)
    if not h:
        return []
    return fn(load(h), h)
