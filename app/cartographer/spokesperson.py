"""Spokesperson (working model) for Cartographer (C1 + C2).

The voice that drives the conversation and writes the reports. It reuses the
engine's provider layer (pse_a.shell.llm): Claude through the official SDK,
Gemini / Mistral / Groq and ~15 others through their OpenAI-compatible
endpoints, local servers (Ollama, LM Studio...), or a local GGUF run by our own
numpy runtime (offline). Model names are NOT hard-coded (except the Claude
default): after a key is entered the provider's own /models list is offered,
so the list never goes stale.

The patient (the model being dissected) is always a local GGUF; the
spokesperson only ever receives summaries of sealed results, never weights or
activations.
"""
from __future__ import annotations

from pse_a.shell import llm, psel

# shown first in the picker (C1); every other provider follows
FEATURED = ["anthropic", "gemini", "mistral", "groq", "local_gguf"]


def provider_choices() -> list[tuple[str, str]]:
    rest = [p for p in llm.PROVIDERS if p not in FEATURED]
    return [(p, llm.PROVIDERS[p][0]) for p in FEATURED + rest]


class Spokesperson:
    """One configured provider: id + entry (key / model / base_url / gguf)."""

    def __init__(self, pid: str, entry: dict):
        self.pid = pid
        self.entry = dict(entry)

    @property
    def name(self) -> str:
        label = llm.PROVIDERS[self.pid][0]
        m = self.entry.get("model") or (llm.CLAUDE_DEFAULT if self.pid == "anthropic" else "")
        if self.pid == "local_gguf":
            from pathlib import Path
            m = Path(self.entry.get("gguf", "")).name
        return f"{label} · {m}" if m else label

    def available(self) -> bool:
        kind, env = llm.PROVIDERS[self.pid][2], llm.PROVIDERS[self.pid][4]
        if kind == "local_gguf":
            return bool(self.entry.get("gguf"))
        if kind == "anthropic":
            return bool(self.entry.get("key")) or bool(env and __import__("os").environ.get(env))
        if not self.entry.get("model") or not self.entry.get("base_url"):
            return False
        return bool(self.entry.get("key")) or not env

    def stream(self, system: str, messages: list):
        yield from llm.chat(self.pid, self.entry, system, messages)

    def complete(self, system: str, messages: list, max_tokens: int = 2000) -> str:
        return "".join(self.stream(system, messages))


class Voice:
    """PSE-L for Cartographer: a spokesperson (or none) + the AZ-8 scrub."""

    def __init__(self, lib, backend: Spokesperson | None = None):
        self.lib = lib
        self.backend = backend

    def has_voice(self) -> bool:
        return bool(self.backend and self.backend.available())

    def system(self, model_hashes: list | None = None) -> str:
        budget = psel.budget_for(self.backend.pid) if self.backend else 24000
        return psel.SYSTEM.format(context=psel.build_context(
            self.lib, max_runs=40, max_chars=budget, model_hashes=model_hashes))

    @staticmethod
    def scrub(text: str) -> str:
        clean, n = psel.enforce_az8(text)
        if n:
            clean += ("\n\n(Part of the text was removed: it contained terms this system "
                      "does not display.)")
        return clean

    def ask(self, prompt: str, model_hashes: list | None = None, system: str | None = None) -> str:
        msgs = [{"role": "user", "content": prompt}]
        return self.scrub(self.backend.complete(system or self.system(model_hashes), msgs))


# ---------------------------------------------------------------- persistence (C1)
def load_saved(lib) -> Spokesperson | None:
    cfg = llm.load_config()
    pid = cfg.get("active")
    if not pid or pid not in llm.PROVIDERS:
        return None
    return Spokesperson(pid, llm.entry(cfg, pid))


def save(sp: Spokesperson | None, remember_key: bool = True, dismissed: bool = False) -> None:
    cfg = llm.load_config()
    cfg["dismissed"] = dismissed
    cfg["remember_keys"] = remember_key
    if sp is not None:
        cfg["active"] = sp.pid
        cfg.setdefault("providers", {})[sp.pid] = dict(sp.entry)
    llm.save_config(cfg)


def first_run_needed() -> bool:
    cfg = llm.load_config()
    return not cfg.get("active") and not cfg.get("dismissed")
