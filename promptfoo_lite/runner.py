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


@dataclass
class EvalResult:
    results: list[TestCaseResult]
    elapsed_s: float

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
        evaluate(completion.text, spec, completion.latency_ms)
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
    )


def plan(config: EvalConfig) -> list[tuple[TestCase, Prompt, ProviderSpec]]:
    """Expand the config into (test, prompt, provider) runs."""
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
    return runs


def run_eval(config: EvalConfig) -> EvalResult:
    """Run every (test, prompt, provider) combination concurrently."""
    runs = plan(config)
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
    return EvalResult(results=results, elapsed_s=time.perf_counter() - started)
