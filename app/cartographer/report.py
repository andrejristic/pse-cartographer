"""
Three text registers over the results (C9).

Every result carries three registers, generated from the SAME numbers:
  1. FORMAL      — the raw numbers/charts (rendered by the GUI, not this module).
  2. EXPERT NOTE — 1-3 concise domain-language sentences per result (spokesperson).
  3. NARODSKI    — a plain-language "what happened" report (spokesperson) that also POINTS
                   the human to the results file and the right tab for a deeper look.

All spokesperson text passes the engine's AZ-8 scrub (no strategic or clinical naming,
whatever the model). Without a spokesperson, both the expert note and the plain report fall
back to deterministic, number-bearing text so the app still says something true (just not
rephrased).
"""
from __future__ import annotations


class Reporter:
    def __init__(self, voice):
        self.voice = voice                       # cartographer.spokesperson.Voice

    def _has_voice(self) -> bool:
        return bool(self.voice and self.voice.has_voice())

    # -- register 2: short expert note per result --------------------------- #
    def expert_note(self, kind: str, facts: str) -> str:
        if not self._has_voice():
            return facts
        try:
            return self.voice.ask(
                f"In 1-3 sentences of precise, technical domain language, explain these "
                f"{kind} measurements to a researcher. Keep the numbers. Do not add anything "
                f"not present:\n{facts}",
                system="You write precise, short notes on neural-network measurements. Use "
                       "only the numbers you are given.")
        except Exception:
            return facts

    # -- register 3: the plain-language report + where-to-look -------------- #
    def narodski_report(self, bundle: dict) -> str:
        facts = self.facts_block(bundle)
        guide = self.where_to_look(bundle)
        if not self._has_voice():
            return facts + "\n\n" + guide
        try:
            return self.voice.ask(
                "Write a plain-language report for a curious non-expert about what this "
                "analysis of a neural network found. Everyday words, no jargon, but KEEP the "
                "concrete numbers and say what they mean. Say plainly that whether small "
                "models have this structure at all is an open question. Then, in a final short "
                "paragraph, tell them exactly where to look if they want more detail (use the "
                f"file and tab names given). Base it ONLY on:\n\n{facts}\n\nWhere to look:\n{guide}",
                system="You explain measurements of neural networks to non-experts, honestly "
                       "and only from the numbers you are given.")
        except Exception:
            return facts + "\n\n" + guide

    # -- deterministic, number-bearing floors ------------------------------- #
    @staticmethod
    def facts_block(b: dict) -> str:
        mm = b.get("model", {})
        l1 = b.get("l1", {})
        s = b.get("session_summary", {})
        parts = [f"Model: {mm.get('display_name', '?')} — {mm.get('architecture', '?')}, "
                 f"{mm.get('n_layers', '?')} layers."]
        if l1:
            parts.append(f"Anatomy (L1): {l1.get('n', 0)} weight matrices; "
                         f"{l1.get('trained', 0)} carry a trained signature, "
                         f"{l1.get('random', 0)} still look noise-like.")
        if s:
            ign = s.get("ignition_layers_seen", [])
            parts.append(
                f"J-space cartography: {s.get('n_prompts', 0)} prompts fed, "
                f"{len(b.get('layers_mapped', []))} layers mapped with "
                f"{b.get('n_probes', '?')} probe words; average movement of held concepts "
                f"between prompts is {s.get('mean_drift', 0):.2f} (0 = identical, 1 = totally "
                f"different), peak {s.get('max_drift', 0):.2f}; "
                + (f"the model committed ('ignition') at layer(s) {ign} in "
                   f"{s.get('ignition_detected_fraction', 0) * 100:.0f}% of prompts; "
                   if ign else "no sharp commitment ('ignition') was detected; ")
                + f"the prompts fell into {s.get('n_clusters', 0)} J-space group(s).")
        return "\n".join(parts)

    @staticmethod
    def where_to_look(b: dict) -> str:
        rp = b.get("results_path", "the results file")
        return (f"For the full per-prompt detail, open {rp} and the 'Drift / Diff' tab; the "
                f"'Session' tab lists what the model held in mind for each prompt, and the "
                f"'Spectral Map' tab shows which layers are trained versus noise-like.")
