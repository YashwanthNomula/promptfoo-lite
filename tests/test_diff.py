"""Tests for the run-to-run regression diff (``pfeval diff``)."""

import json

import pytest

from promptfoo_lite import diff as diff_mod
from promptfoo_lite.cli import main as cli_main


def _case(test, prompt="haiku", provider="stub:base", passed=True,
          latency_ms=100.0, tokens=35, cost=0.000100, score=1.0):
    return {
        "test": test,
        "prompt": prompt,
        "provider": provider,
        "rendered_prompt": "Write a haiku about oceans.",
        "output": "some output",
        "latency_ms": latency_ms,
        "tokens": tokens,
        "prompt_tokens": 20,
        "completion_tokens": 15,
        "cost_usd": cost,
        "score": score,
        "passed": passed,
        "error": None,
        "asserts": [],
    }


def _report(cases):
    return {
        "generated_at": "2026-10-10T00:00:00+00:00",
        "summary": {"total": len(cases)},
        "results": cases,
    }


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(path)


def test_identical_runs_no_regression(tmp_path, capsys):
    cases = [_case("a"), _case("b", passed=False, score=0.5)]
    old = _write(tmp_path, "old.json", _report(cases))
    new = _write(tmp_path, "new.json", _report(cases))
    assert cli_main(["diff", old, new]) == 0
    out = capsys.readouterr().out
    assert "newly failing: 0" in out
    assert "pass rate 50.0% → 50.0%" in out


def test_regression_and_fix_detected(tmp_path, capsys):
    old_cases = [
        _case("regresses", passed=True, score=1.0),
        _case("gets fixed", passed=False, score=0.5),
        _case("steady", passed=True),
    ]
    new_cases = [
        _case("regresses", passed=False, score=0.4),
        _case("gets fixed", passed=True, score=1.0),
        _case("steady", passed=True),
    ]
    old = _write(tmp_path, "old.json", _report(old_cases))
    new = _write(tmp_path, "new.json", _report(new_cases))
    assert cli_main(["diff", old, new]) == 2
    out = capsys.readouterr().out
    assert "REGRESSION" in out
    assert "regresses" in out
    assert "fixed" in out
    assert "gets fixed" in out
    assert "newly failing: 1" in out


def test_added_removed_cases_are_not_regressions(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report([_case("kept"), _case("gone")]))
    new = _write(tmp_path, "new.json", _report([_case("kept"), _case("fresh")]))
    assert cli_main(["diff", old, new]) == 0
    out = capsys.readouterr().out
    assert "compared 1 case(s) · 1 added · 1 removed" in out
    assert "+ fresh" in out
    assert "- gone" in out
    assert "newly failing: 0" in out


def test_invalid_json_rejected(tmp_path, capsys):
    old = _write(tmp_path, "old.json", "{not json")
    new = _write(tmp_path, "new.json", _report([_case("a")]))
    assert cli_main(["diff", old, new]) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_missing_file_rejected(tmp_path, capsys):
    new = _write(tmp_path, "new.json", _report([_case("a")]))
    assert cli_main(["diff", str(tmp_path / "nope.json"), new]) == 1
    assert "no such file" in capsys.readouterr().err


def test_non_eval_json_rejected(tmp_path, capsys):
    old = _write(tmp_path, "old.json", {"hello": "world"})
    new = _write(tmp_path, "new.json", _report([_case("a")]))
    assert cli_main(["diff", old, new]) == 1
    assert "not an eval json report" in capsys.readouterr().err.lower()


def test_no_overlap_errors_clearly(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report([_case("old only")]))
    new = _write(tmp_path, "new.json", _report([_case("new only")]))
    assert cli_main(["diff", old, new]) == 1
    assert "no overlapping cases" in capsys.readouterr().err.lower()


def test_json_output_parseable(tmp_path):
    old = _write(
        tmp_path, "old.json",
        _report([_case("a", latency_ms=100.0), _case("b", passed=False)]),
    )
    new = _write(
        tmp_path, "new.json",
        _report([_case("a", latency_ms=150.0), _case("b", passed=True)]),
    )
    out_path = str(tmp_path / "diff.json")
    assert cli_main(["diff", old, new, "--json", out_path]) == 0
    payload = json.loads((tmp_path / "diff.json").read_text())
    s = payload["summary"]
    assert s["compared"] == 2
    assert s["regressions"] == []
    assert len(s["fixed"]) == 1
    assert len(payload["cases"]) == 2
    assert payload["added"] == []
    assert payload["removed"] == []


def test_delta_math():
    old = _report([_case("a", latency_ms=100.0, tokens=40, cost=0.000200,
                          score=1.0)])
    new = _report([_case("a", latency_ms=150.0, tokens=50, cost=0.000300,
                          score=0.75)])
    diff = diff_mod.diff_runs(old, new)
    c = diff["cases"][0]
    assert c["latency_ms"]["delta"] == 50.0
    assert c["latency_ms"]["delta_pct"] == 50.0
    assert c["tokens"]["delta"] == 10
    assert c["cost_usd"]["delta"] == pytest.approx(0.000100)
    assert c["score"]["delta"] == pytest.approx(-0.25)
    assert diff["summary"]["latency_ms_delta"] == 50.0
    assert diff["summary"]["pass_rate_delta_pp"] == 0.0


def test_zero_old_latency_has_no_pct():
    old = _report([_case("a", latency_ms=0.0)])
    new = _report([_case("a", latency_ms=120.0)])
    c = diff_mod.diff_runs(old, new)["cases"][0]
    assert c["latency_ms"]["delta"] == 120.0
    assert c["latency_ms"]["delta_pct"] is None


def test_terminal_shows_deltas(tmp_path, capsys):
    old = _write(tmp_path, "old.json", _report([_case("a", latency_ms=100.0)]))
    new = _write(tmp_path, "new.json", _report([_case("a", latency_ms=300.0)]))
    cli_main(["diff", old, new])
    out = capsys.readouterr().out
    assert "latency 100ms → 300ms (+200ms, +200.0%)" in out
    assert "total deltas:" in out


def test_load_run_returns_dict(tmp_path):
    path = _write(tmp_path, "r.json", _report([_case("a")]))
    data = diff_mod.load_run(path)
    assert data["results"][0]["test"] == "a"
