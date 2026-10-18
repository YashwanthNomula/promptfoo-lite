"""Run an eval: render prompts, call providers concurrently, grade outputs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from promptfoo_lite import templating
from promptfoo_lite.assertions import (
    AssertResult,
    TestOutcome,
    evaluate,
    score_asserts,
)
from promptfoo_lite.config import EvalConfig, Prompt, ProviderSpec, TestCase
from promptfoo_lite.providers import Completion, Provider, make_provider


@dataclass
class TestCaseResult:
    test: TestCase
    prompt: Prompt
    provider: ProviderSpec
    rendered_prompt: str
    output: str
    latency_ms: float
    asserts: list[AssertResult] = field(default_factory=list)
    score: float = 0.0
    passed: bool = False
    error: str | None = None
    tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class EvalResult:
    results: list[TestCaseResult]
    elapsed_s: float
    filter_text: str | None = None  # substring filter applied, if any
    planned_total: int | None = None  # runs the unfiltered plan would have made

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def avg_score(self) -> float:
        return sum(r.score for r in self.results) / self.total if self.total else 0.0

    def by_prompt(self) -> dict[str, list[TestCaseResult]]:
        groups: dict[str, list[TestCaseResult]] = {}
        for r in self.results:
            groups.setdefault(r.prompt.id, []).append(r)
        return groups

    def by_provider(self) -> dict[str, list[TestCaseResult]]:
        groups: dict[str, list[TestCaseResult]] = {}
        for r in self.results:
            groups.setdefault(r.provider.id, []).append(r)
        return groups

    def stats(self) -> dict:
        """Aggregate latency, token and cost stats across measured cases.

        Only cases that produced a completion are included; template and
        provider errors are excluded.
        """
        measured = [r for r in self.results if r.error is None]
        latencies = sorted(r.latency_ms for r in measured)
        n = len(latencies)
        total_tokens = sum(r.tokens for r in measured if r.tokens is not None)
        total_cost = sum(r.cost_usd for r in measured if r.cost_usd is not None)
        per_provider_tokens: dict[str, int] = {}
        per_provider_cost: dict[str, float] = {}
        for r in measured:
            pid = r.provider.id
            if r.tokens is not None:
                per_provider_tokens[pid] = per_provider_tokens.get(pid, 0) + r.tokens
            if r.cost_usd is not None:
                per_provider_cost[pid] = per_provider_cost.get(pid, 0.0) + r.cost_usd
        return {
            "cases_measured": n,
            "latency_ms": {
                "mean": _mean(latencies),
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "max": latencies[-1] if latencies else 0.0,
            },
            "tokens": {
                "total": total_tokens,
                "by_provider": per_provider_tokens,
            },
            "cost_usd": {
                "total": total_cost,
                "by_provider": {k: round(v, 6) for k, v in per_provider_cost.items()},
            },
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Percentile of pre-sorted values (linear interpolation, 0..100)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = pct / 100 * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def _run_one(
    test: TestCase,
    prompt: Prompt,
    provider_spec: ProviderSpec,
    providers: dict[str, Provider],
) -> TestCaseResult:
    try:
        rendered = templating.render(prompt.raw, test.vars)
    except templating.TemplateError as exc:
        return TestCaseResult(
            test=test, prompt=prompt, provider=provider_spec,
            rendered_prompt=prompt.raw, output="", latency_ms=0.0,
            error=f"template error: {exc}",
        )

    provider = providers[provider_spec.id]
    try:
        completion: Completion = provider.complete(rendered)
    except Exception as exc:  # provider failures become failing cases, not crashes
        return TestCaseResult(
            test=test, prompt=prompt, provider=provider_spec,
            rendered_prompt=rendered, output="", latency_ms=0.0,
            error=f"provider error: {exc}",
        )

    asserts = [
        evaluate(completion.text, spec, completion.latency_ms, completion.cost_usd)
        for spec in test.assert_
    ]
    score = score_asserts(asserts)
    threshold = test.threshold if test.threshold is not None else 1.0
    passed = all(a.passed for a in asserts) and score >= threshold
    return TestCaseResult(
        test=test, prompt=prompt, provider=provider_spec,
        rendered_prompt=rendered, output=completion.text,
        latency_ms=completion.latency_ms, asserts=asserts,
        score=score, passed=passed, tokens=completion.tokens,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        cost_usd=completion.cost_usd,
    )


def plan(config: EvalConfig, filter_text: str | None = None) -> list[tuple[TestCase, Prompt, ProviderSpec]]:
    """Expand the config into (test, prompt, provider) runs.

    With ``filter_text``, keep only runs whose test description, prompt id /
    label, or provider id / label contains the substring (case-insensitive).
    A blank filter means no filtering — so ``--filter ""`` behaves like no
    filter instead of matching nothing.
    """
    prompts = {p.id: p for p in config.prompts}
    providers = {p.id: p for p in config.providers}
    runs = []
    for test in config.tests:
        test_prompts = (
            [prompts[pid] for pid in test.prompts] if test.prompts else config.prompts
        )
        test_providers = (
            [providers[pid] for pid in test.providers]
            if test.providers
            else config.providers
        )
        for prompt in test_prompts:
            for provider_spec in test_providers:
                runs.append((test, prompt, provider_spec))
    needle = (filter_text or "").strip().lower()
    if not needle:
        return runs
    return [
        run
        for run in runs
        if needle in run[0].description.lower()
        or needle in run[1].id.lower()
        or needle in run[1].label.lower()
        or needle in run[2].id.lower()
        or needle in run[2].label.lower()
    ]


def run_eval(config: EvalConfig, filter_text: str | None = None) -> EvalResult:
    """Run every (test, prompt, provider) combination concurrently."""
    full_plan = plan(config)
    runs = plan(config, filter_text)
    providers = {spec.id: make_provider(spec.id) for spec in config.providers}
    started = time.perf_counter()
    results: list[TestCaseResult] = []
    workers = max(1, min(config.max_concurrency, len(runs) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_one, test, prompt, provider_spec, providers)
            for test, prompt, provider_spec in runs
        ]
        # Preserve plan order for deterministic output.
        for future in futures:
            results.append(future.result())
    return EvalResult(
        results=results,
        elapsed_s=time.perf_counter() - started,
        filter_text=(filter_text.strip() if filter_text and filter_text.strip() else None),
        planned_total=len(full_plan) if filter_text and filter_text.strip() else None,
    )
