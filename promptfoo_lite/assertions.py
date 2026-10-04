"""Assertion types: the graders that score a model's output."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from promptfoo_lite.config import AssertionSpec


@dataclass
class AssertResult:
    passed: bool
    score: float  # 0.0 .. 1.0
    reason: str
    weight: float = 1.0


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _check_contains(output: str, spec: AssertionSpec) -> AssertResult:
    values = spec.values or ([spec.value] if spec.value is not None else [])
    missing = [v for v in values if str(v) not in output]
    ok = not missing
    return AssertResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=(
            "contains all of " + ", ".join(repr(str(v)) for v in values)
            if ok
            else "missing " + ", ".join(repr(str(v)) for v in missing)
        ),
        weight=spec.weight,
    )


def _check_icontains(output: str, spec: AssertionSpec) -> AssertResult:
    values = spec.values or ([spec.value] if spec.value is not None else [])
    lowered = output.lower()
    missing = [v for v in values if str(v).lower() not in lowered]
    ok = not missing
    return AssertResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=(
            "contains (case-insensitive) all expected strings"
            if ok
            else "missing (case-insensitive) " + ", ".join(repr(str(v)) for v in missing)
        ),
        weight=spec.weight,
    )


def _check_not_contains(output: str, spec: AssertionSpec) -> AssertResult:
    values = spec.values or ([spec.value] if spec.value is not None else [])
    found = [v for v in values if str(v).lower() in output.lower()]
    ok = not found
    return AssertResult(
        passed=ok,
        score=1.0 if ok else 0.0,
        reason=(
            "contains none of the forbidden strings"
            if ok
            else "forbidden string(s) present: " + ", ".join(repr(str(v)) for v in found)
        ),
        weight=spec.weight,
    )


def _check_equals(output: str, spec: AssertionSpec) -> AssertResult:
    ok = output.strip() == str(spec.value or "").strip()
    return AssertResult(
        passed=ok, score=1.0 if ok else 0.0,
        reason="exact match" if ok else "output differs from expected",
        weight=spec.weight,
    )


def _check_regex(output: str, spec: AssertionSpec) -> AssertResult:
    try:
        matched = re.search(str(spec.value or ""), output) is not None
    except re.error as exc:
        return AssertResult(
            passed=False, score=0.0, reason=f"invalid regex: {exc}", weight=spec.weight
        )
    return AssertResult(
        passed=matched, score=1.0 if matched else 0.0,
        reason="regex matched" if matched else f"regex {spec.value!r} did not match",
        weight=spec.weight,
    )


def _check_starts_ends(output: str, spec: AssertionSpec, start: bool) -> AssertResult:
    target = str(spec.value or "")
    ok = output.startswith(target) if start else output.endswith(target)
    verb = "starts with" if start else "ends with"
    return AssertResult(
        passed=ok, score=1.0 if ok else 0.0,
        reason=f"output {verb} {target!r}" if ok else f"output does not {verb} {target!r}",
        weight=spec.weight,
    )


def _check_word_count(output: str, spec: AssertionSpec) -> AssertResult:
    n = len(_words(output))
    lo = spec.min if spec.min is not None else 0
    hi = spec.max if spec.max is not None else float("inf")
    ok = lo <= n <= hi
    return AssertResult(
        passed=ok, score=1.0 if ok else 0.0,
        reason=f"{n} words (allowed {lo}..{hi if hi != float('inf') else '∞'})",
        weight=spec.weight,
    )


def _check_char_count(output: str, spec: AssertionSpec) -> AssertResult:
    n = len(output)
    lo = spec.min if spec.min is not None else 0
    hi = spec.max if spec.max is not None else float("inf")
    ok = lo <= n <= hi
    return AssertResult(
        passed=ok, score=1.0 if ok else 0.0,
        reason=f"{n} chars (allowed {lo}..{hi if hi != float('inf') else '∞'})",
        weight=spec.weight,
    )


def _check_similarity(output: str, spec: AssertionSpec) -> AssertResult:
    """Word-overlap similarity (Jaccard) against an expected reference."""
    threshold = float(spec.threshold if spec.threshold is not None else 0.5)
    ref = {w for w in _words(str(spec.value or ""))}
    got = {w for w in _words(output)}
    score = len(ref & got) / len(ref | got) if ref or got else 1.0
    ok = score >= threshold
    return AssertResult(
        passed=ok, score=score,
        reason=f"similarity {score:.2f} (threshold {threshold:.2f})",
        weight=spec.weight,
    )


def _check_latency_ms(latency_ms: float, spec: AssertionSpec) -> AssertResult:
    limit = float(spec.threshold if spec.threshold is not None else 5000)
    ok = latency_ms <= limit
    return AssertResult(
        passed=ok, score=1.0 if ok else 0.0,
        reason=f"{latency_ms:.0f}ms (limit {limit:.0f}ms)",
        weight=spec.weight,
    )


_HANDLERS = {
    "contains": _check_contains,
    "icontains": _check_icontains,
    "not-contains": _check_not_contains,
    "not_contains": _check_not_contains,
    "equals": _check_equals,
    "regex": _check_regex,
    "starts-with": lambda o, s: _check_starts_ends(o, s, True),
    "ends-with": lambda o, s: _check_starts_ends(o, s, False),
    "word-count": _check_word_count,
    "char-count": _check_char_count,
    "similarity": _check_similarity,
}


def supported_types() -> list[str]:
    return sorted(_HANDLERS) + ["latency"]


def evaluate(output: str, spec: AssertionSpec, latency_ms: float = 0.0) -> AssertResult:
    """Evaluate a single assertion against a model output."""
    if spec.type == "latency":
        return _check_latency_ms(latency_ms, spec)
    handler = _HANDLERS.get(spec.type)
    if handler is None:
        return AssertResult(
            passed=False, score=0.0,
            reason=f"unknown assertion type {spec.type!r} "
                   f"(supported: {', '.join(supported_types())})",
            weight=spec.weight,
        )
    return handler(output, spec)


def score_asserts(results: list[AssertResult]) -> float:
    """Weighted average score of assertion results."""
    total = sum(r.weight for r in results)
    if total == 0:
        return 1.0
    return sum(r.score * r.weight for r in results) / total


@dataclass
class TestOutcome:
    passed: bool
    score: float
    asserts: list[AssertResult] = field(default_factory=list)
