"""Model providers. The stub provider needs no API key; the OpenAI one does."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Raised when a provider cannot produce a completion."""


@dataclass
class Completion:
    text: str
    latency_ms: float
    tokens: int | None = None


class Provider:
    id: str

    def complete(self, prompt: str) -> Completion:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stub: a deterministic fake model. Seeded by provider id + prompt, so evals
# are reproducible without any API key.
# ---------------------------------------------------------------------------

_SENTENCE_FRAGMENTS = [
    "Here is a thoughtful answer",
    "Let me break this down",
    "The key points are",
    "In summary",
    "To answer directly",
    "Here's what you should know",
]

_DETAIL_FRAGMENTS = [
    "the core idea is surprisingly simple",
    "most people overlook the details",
    "it helps to think of it in layers",
    "there are a few angles worth considering",
    "the short answer is yes, with caveats",
    "consistency matters more than intensity here",
]


class StubProvider(Provider):
    """Deterministic fake LLM: ``stub`` or ``stub:<seed>``.

    Responses are seeded by (provider id, prompt), so the same eval always
    produces the same outputs. Different seeds give different "models" —
    handy for comparing prompt variants side by side.
    """

    def __init__(self, id: str = "stub"):
        self.id = id
        seed_material = id.encode()
        self._seed = int(hashlib.sha256(seed_material).hexdigest(), 16) % (2**32)

    def complete(self, prompt: str) -> Completion:
        start = time.perf_counter()
        rng = random.Random(
            self._seed ^ int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        )
        # Pull a "topic" out of the prompt: the longest alphabetic word,
        # which usually survives {{var}} substitution as the subject.
        words = re.findall(r"[A-Za-z]+", prompt.replace("\n", " "))
        topic = max(words, key=len) if words else "that"
        opener = rng.choice(_SENTENCE_FRAGMENTS)
        detail = rng.choice(_DETAIL_FRAGMENTS)
        closer = rng.choice(_DETAIL_FRAGMENTS)
        text = (
            f"{opener} about {topic.lower()}: {detail}, and {closer}. "
            f"In short — {topic.lower()} rewards a little curiosity."
        )
        latency_ms = (time.perf_counter() - start) * 1000 + rng.uniform(5, 40)
        return Completion(
            text=text, latency_ms=latency_ms, tokens=len(text.split())
        )


class EchoProvider(Provider):
    """Returns the rendered prompt verbatim. Useful for testing the harness."""

    def __init__(self, id: str = "echo"):
        self.id = id

    def complete(self, prompt: str) -> Completion:
        start = time.perf_counter()
        return Completion(
            text=prompt, latency_ms=(time.perf_counter() - start) * 1000, tokens=None
        )


class OpenAIProvider(Provider):
    """OpenAI-compatible chat completions over stdlib urllib.

    Provider id: ``openai`` or ``openai:<model>`` (default ``gpt-4o-mini``).
    Requires the ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(self, id: str = "openai"):
        self.id = id
        parts = id.split(":", 1)
        self.model = parts[1] if len(parts) == 2 and parts[1] else "gpt-4o-mini"
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError(
                f"provider {id!r} needs the OPENAI_API_KEY environment variable"
            )

    def complete(self, prompt: str) -> Completion:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except Exception as exc:  # network / API errors
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            text = payload["choices"][0]["message"]["content"]
            tokens = payload.get("usage", {}).get("total_tokens")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected OpenAI response shape: {exc}") from exc
        return Completion(text=text or "", latency_ms=latency_ms, tokens=tokens)


def make_provider(provider_id: str) -> Provider:
    """Build a provider from its config id."""
    kind = provider_id.split(":", 1)[0].lower()
    if kind == "stub":
        return StubProvider(provider_id)
    if kind == "echo":
        return EchoProvider(provider_id)
    if kind == "openai":
        return OpenAIProvider(provider_id)
    raise ProviderError(
        f"unknown provider {provider_id!r} "
        f"(supported: stub[:seed], echo, openai[:model])"
    )
