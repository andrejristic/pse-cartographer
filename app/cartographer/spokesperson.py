"""
Spokesperson factory for Cartographer (C1 + C2).

Adds Gemini as a fifth option WITHOUT touching the shared engine: Gemini is reached
through Google's OpenAI-compatible endpoint, so it reuses the same chat-completions shape
as Mistral/Groq. The other four choices (Claude via the anthropic SDK, Mistral, Groq,
local GGUF) delegate straight to the engine's own factory.

Choice strings used across the GUI: "claude", "gemini", "mistral", "groq", "local".
"""
from __future__ import annotations

import os

from pse_a.llm.backend import make_backend as _engine_make_backend
from pse_a.llm.cloud import CloudLLM

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

# What each choice offers / needs — drives the first-run modal copy (C1).
CHOICES = [
    ("claude", "Claude (Anthropic API)", "ANTHROPIC_API_KEY"),
    ("gemini", "Gemini (Google API)", "GEMINI_API_KEY"),
    ("mistral", "Mistral (API)", "MISTRAL_API_KEY"),
    ("groq", "Groq (API)", "GROQ_API_KEY"),
    ("local", "Local GGUF (offline, give a path)", None),
]


class GeminiBackend(CloudLLM):
    """Gemini via the OpenAI-compatible endpoint (kept out of the shared engine)."""

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(provider="gemini",
                         model=model or GEMINI_DEFAULT_MODEL,
                         api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        self.name = "cloud:gemini"

    def complete(self, system: str, messages: list, max_tokens: int = 700) -> str:
        from openai import OpenAI  # lazy
        client = OpenAI(api_key=self.api_key or None, base_url=GEMINI_BASE_URL)
        msgs = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=max_tokens)
        return (resp.choices[0].message.content or "").strip()


def make_backend(choice: str, api_key: str | None = None, model: str | None = None,
                 model_path: str | None = None):
    """
    Build a spokesperson backend from a choice string.
      cloud choices: claude | gemini | mistral | groq  (api_key/model optional)
      local:         local  (model_path required)
    """
    if choice == "gemini":
        return GeminiBackend(model=model, api_key=api_key)
    if choice == "local":
        return _engine_make_backend("local", model_path=model_path)
    return _engine_make_backend("cloud", provider=choice, api_key=api_key, model=model)
