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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Cost tracking. Prices are per 1M tokens as (input, output) USD. They are
# ballpark public rates for cost *estimation* in eval reports — always check
# the provider's current pricing before quoting them.
# ---------------------------------------------------------------------------

_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "gpt-3.5-turbo": (0.50, 1.50),
}

# Blended fallback rate (per 1M tokens) for models not in the table.
_DEFAULT_PRICE_PER_1M = 0.60


def estimate_cost_usd(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None = None,
) -> float | None:
    """Estimate the USD cost of one completion from token usage."""
    if prompt_tokens is not None and completion_tokens is not None:
        in_price, out_price = _MODEL_PRICES.get(
            model, (_DEFAULT_PRICE_PER_1M, _DEFAULT_PRICE_PER_1M)
        )
        return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
    if total_tokens is not None:
        return total_tokens * _DEFAULT_PRICE_PER_1M / 1_000_000
    return None


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
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())
        return Completion(
            text=text,
            latency_ms=latency_ms,
            tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,  # stub is free — synthesized for report shape parity
        )


class EchoProvider(Provider):
    """Returns the rendered prompt verbatim. Useful for testing the harness."""

    def __init__(self, id: str = "echo"):
        self.id = id

    def complete(self, prompt: str) -> Completion:
        start = time.perf_counter()
        prompt_tokens = len(prompt.split())
        return Completion(
            text=prompt,
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens=prompt_tokens * 2,
            prompt_tokens=prompt_tokens,
            completion_tokens=prompt_tokens,
            cost_usd=0.0,  # echo is free — synthesized for report shape parity
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
            usage = payload.get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected OpenAI response shape: {exc}") from exc
        cost = estimate_cost_usd(
            self.model, prompt_tokens, completion_tokens, total_tokens
        )
        return Completion(
            text=text or "",
            latency_ms=latency_ms,
            tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
        )


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
