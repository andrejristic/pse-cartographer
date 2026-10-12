"""
Three text layers over the results (C9).

Every result carries three registers, generated from the SAME sealed numbers:
  1. FORMAL      — the raw numbers/charts (rendered elsewhere; not this module).
  2. EXPERT NOTE — 1-3 concise domain-language sentences per result (spokesperson).
  3. NARODSKI    — a plain-language "what happened" report (spokesperson) that also POINTS
                   the human to the results file and the right tab for a deeper look.

All spokesperson text goes through the engine's PSE-L generator, so the I-4/I-5
no-strategic-naming guardrail applies unchanged. Without a spokesperson, both the expert
note and the narodski report fall back to deterministic, number-bearing templates so the
app still says something true (just not rephrased).
"""
from __future__ import annotations


class Reporter:
    def __init__(self, psel):
        self.psel = psel                       # pse_a.llm.psel.PSELGenerator

    def _has_voice(self) -> bool:
        return bool(self.psel and self.psel.backend and self.psel.backend.available())

    # -- layer 2: short expert note per result ------------------------------ #
    def expert_note(self, kind: str, facts: str) -> str:
        if not self._has_voice():
            return facts                        # deterministic fact line as the floor
        try:
            msgs = [{"role": "user",
                     "content": f"In 1-3 sentences of precise, technical domain language, "
                                f"explain these {kind} measurements to a researcher. Keep "
                                f"the numbers. Do not add anything not present:\n{facts}"}]
            return self.psel._scrub(self.psel.backend.complete(self.psel._system(), msgs,
                                                               max_tokens=220))
        except Exception:
            return facts

    # -- layer 3: the narodski report + where-to-look ----------------------- #
    def narodski_report(self, bundle: dict) -> str:
        facts = self._facts_block(bundle)
        guide = self._where_to_look(bundle)
        if not self._has_voice():
            return facts + "\n\n" + guide
        try:
            msgs = [{"role": "user",
                     "content":
                        "Write a plain-language report for a curious non-expert about what "
                        "this analysis of a neural network found. Everyday words, no jargon, "
                        "but KEEP the concrete numbers and say what they mean. Then, in a "
                        "final short paragraph, tell them exactly where to look if they want "
                        "more detail (use the file and tab names given). Base it ONLY on:\n\n"
                        f"{facts}\n\nWhere to look:\n{guide}"}]
            return self.psel._scrub(self.psel.backend.complete(self.psel._system(), msgs,
                                                               max_tokens=900))
        except Exception:
            return facts + "\n\n" + guide

    # -- deterministic, number-bearing floors ------------------------------ #
    @staticmethod
    def _facts_block(b: dict) -> str:
        mm = b.get("model", {})
        l1 = b.get("l1", {})
        s = b.get("session_summary", {})
        parts = [
            f"Model: {mm.get('display_name','?')} — {mm.get('architecture','?')}, "
            f"{mm.get('n_layers','?')} layers.",
        ]
        if l1:
            parts.append(f"Anatomy (L1): {l1.get('n',0)} weight matrices; "
                         f"{l1.get('trained',0)} carry a trained signature, "
                         f"{l1.get('random',0)} still look noise-like.")
        if s:
            parts.append(
                f"J-space cartography: {s.get('n_prompts',0)} prompts fed; average "
                f"movement of held concepts between prompts is "
                f"{s.get('mean_drift',0):.2f} (0=identical, 1=totally different), peak "
                f"{s.get('max_drift',0):.2f}; the model committed ('ignition') at "
                f"layer(s) {s.get('ignition_layers_seen',[])} in "
                f"{s.get('ignition_detected_fraction',0)*100:.0f}% of prompts; the prompts "
                f"fell into {s.get('n_clusters',0)} J-space cluster(s).")
        return "\n".join(parts)

    @staticmethod
    def _where_to_look(b: dict) -> str:
        rp = b.get("results_path", "the results file")
        return (f"For the full per-prompt detail, open {rp} and the 'Drift / Diff' tab; "
                f"the 'J-space' tab shows the depth profile and ignition marker, and the "
                f"'Spectral Map' tab shows which layers are trained versus noise-like.")
