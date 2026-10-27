"""Tests for the not-regex assertion type."""

from promptfoo_lite.assertions import evaluate
from promptfoo_lite.config import AssertionSpec


def spec(type_, **kw):
    return AssertionSpec(type=type_, **kw)


def test_not_regex_passes_when_pattern_absent():
    assert evaluate("all good here", spec("not-regex", value=r"sk-\w+")).passed


def test_not_regex_fails_when_pattern_present():
    r = evaluate("leaked key sk-abc123 here", spec("not-regex", value=r"sk-\w+"))
    assert not r.passed
    assert "sk-" in r.reason


def test_not_regex_invalid_pattern_fails_closed():
    r = evaluate("x", spec("not-regex", value="(["))
    assert not r.passed and "invalid regex" in r.reason


def test_not_regex_underscore_alias():
    assert evaluate("clean", spec("not_regex", value=r"\d+")).passed
    assert not evaluate("has 42", spec("not_regex", value=r"\d+")).passed
