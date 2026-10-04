"""Tests for providers."""

import os

import pytest

from promptfoo_lite.providers import (
    EchoProvider,
    OpenAIProvider,
    ProviderError,
    StubProvider,
    make_provider,
)


def test_stub_is_deterministic():
    p = StubProvider("stub:test")
    assert p.complete("Write about oceans.").text == p.complete("Write about oceans.").text


def test_stub_seeds_differ():
    a = StubProvider("stub:a").complete("Write about oceans.").text
    b = StubProvider("stub:b").complete("Write about oceans.").text
    assert a != b


def test_stub_mentions_prompt_topic():
    out = StubProvider("stub:x").complete("Write a haiku about oceans.").text
    assert "oceans" in out.lower()


def test_echo():
    p = EchoProvider()
    c = p.complete("hello")
    assert c.text == "hello"
    assert c.latency_ms >= 0


def test_make_provider_dispatch():
    assert isinstance(make_provider("stub"), StubProvider)
    assert isinstance(make_provider("stub:custom"), StubProvider)
    assert isinstance(make_provider("echo"), EchoProvider)
    with pytest.raises(ProviderError, match="unknown provider"):
        make_provider("nope:model")


def test_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider("openai")


def test_openai_parses_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert OpenAIProvider("openai:gpt-4o").model == "gpt-4o"
    assert OpenAIProvider("openai").model == "gpt-4o-mini"
