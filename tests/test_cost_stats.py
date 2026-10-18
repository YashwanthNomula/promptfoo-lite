"""Tests for latency/token/cost tracking and the cost assertion."""

import json

from promptfoo_lite import report
from promptfoo_lite.assertions import evaluate, supported_types
from promptfoo_lite.config import (
    AssertionSpec,
    Prompt,
    ProviderSpec,
    TestCase as ConfigTestCase,
)
from promptfoo_lite.providers import (
    EchoProvider,
    OpenAIProvider,
    StubProvider,
    estimate_cost_usd,
)
from promptfoo_lite.runner import EvalResult
from promptfoo_lite.runner import TestCaseResult as RunnerCaseResult


def spec(type_, **kw):
    return AssertionSpec(type=type_, **kw)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def test_estimate_cost_exact_model_prices():
    # gpt-4o-mini: $0.15 / $0.60 per 1M tokens
    cost = estimate_cost_usd("gpt-4o-mini", 100, 50)
    assert cost == (100 * 0.15 + 50 * 0.60) / 1_000_000


def test_estimate_cost_unknown_model_uses_fallback():
    cost = estimate_cost_usd("some-future-model", 1000, 1000)
    assert cost == 2000 * 0.60 / 1_000_000


def test_estimate_cost_total_tokens_only():
    assert estimate_cost_usd("whatever", None, None, 500) == 500 * 0.60 / 1_000_000


def test_estimate_cost_no_usage_returns_none():
    assert estimate_cost_usd("gpt-4o-mini", None, None) is None


# ---------------------------------------------------------------------------
# Providers synthesize token/cost data
# ---------------------------------------------------------------------------

def test_stub_reports_tokens_and_zero_cost():
    c = StubProvider("stub:base").complete("Write a haiku about oceans.")
    assert c.prompt_tokens == 5
    assert c.completion_tokens == len(c.text.split())
    assert c.tokens == c.prompt_tokens + c.completion_tokens
    assert c.cost_usd == 0.0


def test_echo_reports_tokens_and_zero_cost():
    c = EchoProvider().complete("hello world")
    assert c.text == "hello world"
    assert c.tokens == 4
    assert c.cost_usd == 0.0


def _fake_openai_response(monkeypatch, usage):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "hi there"}}],
                    "usage": usage,
                }
            ).encode()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=60: FakeResp())


def test_openai_parses_usage_and_estimates_cost(monkeypatch):
    _fake_openai_response(
        monkeypatch,
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    c = OpenAIProvider("openai:gpt-4o-mini").complete("say hi")
    assert c.text == "hi there"
    assert c.prompt_tokens == 100
    assert c.completion_tokens == 50
    assert c.tokens == 150
    assert c.cost_usd == (100 * 0.15 + 50 * 0.60) / 1_000_000


def test_openai_total_tokens_only_falls_back(monkeypatch):
    _fake_openai_response(monkeypatch, {"total_tokens": 200})
    c = OpenAIProvider("openai:gpt-4o").complete("say hi")
    assert c.tokens == 200
    assert c.cost_usd == 200 * 0.60 / 1_000_000


# ---------------------------------------------------------------------------
# Cost assertion (the 13th grader)
# ---------------------------------------------------------------------------

def test_cost_assertion_in_supported_types():
    assert "cost" in supported_types()


def test_cost_assertion_passes_under_limit():
    r = evaluate("anything", spec("cost", threshold=0.01), cost_usd=0.0002)
    assert r.passed
    assert "$0.000200" in r.reason


def test_cost_assertion_fails_over_limit():
    r = evaluate("anything", spec("cost", threshold=0.0001), cost_usd=0.0002)
    assert not r.passed


def test_cost_assertion_default_limit():
    assert evaluate("x", spec("cost"), cost_usd=0.009).passed
    assert not evaluate("x", spec("cost"), cost_usd=0.02).passed


def test_cost_assertion_unknown_cost_fails():
    r = evaluate("x", spec("cost", threshold=1.0), cost_usd=None)
    assert not r.passed
    assert "not reported" in r.reason


# ---------------------------------------------------------------------------
# EvalResult.stats()
# ---------------------------------------------------------------------------

def _case(prompt_id, provider_id, latency_ms, tokens, cost_usd, error=None):
    return RunnerCaseResult(
        test=ConfigTestCase("t", {}, [], [], [], None),
        prompt=Prompt(prompt_id, prompt_id, "raw"),
        provider=ProviderSpec(provider_id, provider_id),
        rendered_prompt="raw",
        output="out",
        latency_ms=latency_ms,
        tokens=tokens,
        cost_usd=cost_usd,
        error=error,
    )


def test_stats_aggregates_latency_tokens_cost():
    results = [
        _case("p1", "stub:a", 10.0, 100, 0.0),
        _case("p1", "stub:a", 20.0, 100, 0.0),
        _case("p1", "stub:b", 30.0, 200, 0.001),
        _case("p1", "stub:b", 40.0, 200, 0.002),
    ]
    stats = EvalResult(results=results, elapsed_s=1.0).stats()
    assert stats["cases_measured"] == 4
    lat = stats["latency_ms"]
    assert lat["mean"] == 25.0
    assert lat["p50"] == 25.0
    assert lat["p95"] == 38.5  # linear interpolation between 30 and 40
    assert lat["max"] == 40.0
    assert stats["tokens"]["total"] == 600
    assert stats["tokens"]["by_provider"] == {"stub:a": 200, "stub:b": 400}
    assert stats["cost_usd"]["total"] == 0.003
    assert stats["cost_usd"]["by_provider"] == {"stub:a": 0.0, "stub:b": 0.003}


def test_stats_excludes_error_cases():
    results = [
        _case("p1", "stub:a", 10.0, 100, 0.0),
        _case("p1", "stub:a", 0.0, None, None, error="provider error: boom"),
    ]
    stats = EvalResult(results=results, elapsed_s=1.0).stats()
    assert stats["cases_measured"] == 1
    assert stats["latency_ms"]["mean"] == 10.0
    assert stats["tokens"]["total"] == 100


def test_stats_empty_eval():
    stats = EvalResult(results=[], elapsed_s=0.0).stats()
    assert stats["cases_measured"] == 0
    assert stats["latency_ms"]["mean"] == 0.0
    assert stats["tokens"]["total"] == 0
    assert stats["cost_usd"]["total"] == 0.0


# ---------------------------------------------------------------------------
# Reports surface the stats
# ---------------------------------------------------------------------------

def test_report_dict_includes_stats_and_case_cost():
    result = EvalResult(
        results=[_case("p1", "stub:a", 10.0, 100, 0.000045)], elapsed_s=1.0
    )
    d = report.to_dict(result)
    assert d["summary"]["latency_ms"]["mean"] == 10.0
    assert d["summary"]["tokens"]["total"] == 100
    assert d["summary"]["cost_usd"]["total"] == 0.000045
    assert d["results"][0]["cost_usd"] == 0.000045


def test_terminal_report_shows_cost_lines():
    result = EvalResult(
        results=[_case("p1", "stub:a", 10.0, 100, 0.000045)], elapsed_s=1.0
    )
    text = report.terminal(result)
    assert "est. cost $0.0000" in text
    assert "latency mean 10ms · p50 10ms · p95 10ms" in text
    assert "100 tokens" in text
    assert "$0.000045" in text


def test_html_report_shows_cost_column_and_summary():
    result = EvalResult(
        results=[_case("p1", "stub:a", 10.0, 100, 0.000045)], elapsed_s=1.0
    )
    page = report.to_html(result)
    assert "<th>Cost</th>" in page
    assert "$0.000045" in page
    assert "estimated cost" in page
