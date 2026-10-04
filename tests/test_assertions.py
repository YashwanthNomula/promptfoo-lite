"""Tests for assertion types."""

from promptfoo_lite.assertions import evaluate, score_asserts
from promptfoo_lite.config import AssertionSpec


def spec(type_, **kw):
    return AssertionSpec(type=type_, **kw)


def test_contains():
    r = evaluate("the quick brown fox", spec("contains", value="quick"))
    assert r.passed and r.score == 1.0
    r = evaluate("the quick brown fox", spec("contains", values=["quick", "cat"]))
    assert not r.passed
    assert "cat" in r.reason


def test_icontains():
    assert evaluate("Hello World", spec("icontains", value="hello")).passed


def test_not_contains():
    assert evaluate("all good here", spec("not-contains", values=["sorry", "cannot"])).passed
    r = evaluate("sorry, cannot do that", spec("not-contains", values=["sorry"]))
    assert not r.passed


def test_equals():
    assert evaluate("  yes  ", spec("equals", value="yes")).passed
    assert not evaluate("yes!", spec("equals", value="yes")).passed


def test_regex():
    assert evaluate("order #12345 shipped", spec("regex", value=r"#\d+")).passed
    assert not evaluate("no number here", spec("regex", value=r"#\d+")).passed
    r = evaluate("x", spec("regex", value="(["))
    assert not r.passed and "invalid regex" in r.reason


def test_starts_ends_with():
    assert evaluate("hello world", spec("starts-with", value="hello")).passed
    assert evaluate("hello world", spec("ends-with", value="world")).passed
    assert not evaluate("hello world", spec("starts-with", value="world")).passed


def test_word_count():
    assert evaluate("one two three", spec("word-count", min=2, max=4)).passed
    assert not evaluate("one two three", spec("word-count", max=2)).passed
    assert not evaluate("one", spec("word-count", min=2)).passed


def test_char_count():
    assert evaluate("abc", spec("char-count", min=1, max=5)).passed
    assert not evaluate("abcdef", spec("char-count", max=5)).passed


def test_similarity():
    r = evaluate("the cat sat on the mat", spec("similarity", value="cat sat mat", threshold=0.4))
    assert r.passed and 0.0 <= r.score <= 1.0
    r = evaluate("completely unrelated sentence here", spec("similarity", value="cat sat mat", threshold=0.9))
    assert not r.passed


def test_latency():
    assert evaluate("x", spec("latency", threshold=100), latency_ms=42).passed
    assert not evaluate("x", spec("latency", threshold=10), latency_ms=42).passed


def test_unknown_type():
    r = evaluate("x", spec("bogus"))
    assert not r.passed and "unknown assertion type" in r.reason


def test_score_asserts_weighted():
    from promptfoo_lite.assertions import AssertResult

    results = [
        AssertResult(passed=True, score=1.0, reason="", weight=1.0),
        AssertResult(passed=False, score=0.0, reason="", weight=3.0),
    ]
    assert score_asserts(results) == 0.25
    assert score_asserts([]) == 1.0
