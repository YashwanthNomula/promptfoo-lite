"""Tests for the token-budget gate: --budget flag, config "budget" key,
validation, exit codes, and the budget verdict in all three reports."""

import json

import pytest

from promptfoo_lite import cli, report
from promptfoo_lite.budget import BudgetError, check_budget, parse_budget
from promptfoo_lite.config import (
    ConfigError,
    Prompt,
    ProviderSpec,
    TestCase as ConfigTestCase,
    load_config,
)
from promptfoo_lite.runner import EvalResult, TestCaseResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case(cost_usd, passed=True):
    return TestCaseResult(
        test=ConfigTestCase(
            description="t", vars={}, prompts=[], providers=[], assert_=[]
        ),
        prompt=Prompt(id="p", label="p", raw="hi"),
        provider=ProviderSpec(id="stub", label="stub"),
        rendered_prompt="hi",
        output="hi",
        latency_ms=1.0,
        score=1.0,
        passed=passed,
        tokens=10,
        cost_usd=cost_usd,
    )


def _result(costs, passed=True):
    return EvalResult(
        results=[_case(c, passed=passed) for c in costs], elapsed_s=0.1
    )


def _echo_config(tmp_path, extra=None):
    cfg = {
        "prompts": ["Say {{thing}}."],
        "providers": ["echo"],
        "tests": [
            {
                "description": "echo test",
                "vars": {"thing": "hi"},
                "assert": [{"type": "contains", "value": "hi"}],
            }
        ],
    }
    if extra:
        cfg.update(extra)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(cfg))
    return str(path)


# ---------------------------------------------------------------------------
# parse_budget: fail-closed validation
# ---------------------------------------------------------------------------

def test_parse_budget_valid_values():
    assert parse_budget(0) == 0.0
    assert parse_budget(1.5) == 1.5
    assert parse_budget(10) == 10.0
    assert parse_budget("0.25") == 0.25
    assert parse_budget(" 2 ") == 2.0  # whitespace-tolerant strings


def test_parse_budget_zero_is_allowed():
    assert parse_budget(0) == 0.0
    assert parse_budget("0") == 0.0


def test_parse_budget_rejects_negative():
    with pytest.raises(BudgetError):
        parse_budget(-0.01)
    with pytest.raises(BudgetError):
        parse_budget("-1")


def test_parse_budget_rejects_non_numeric():
    for bad in ("lots", "", None, ["1"], {"x": 1}):
        with pytest.raises(BudgetError):
            parse_budget(bad)


def test_parse_budget_rejects_bool_nan_inf():
    for bad in (True, False, float("nan"), float("inf"), "nan", "inf"):
        with pytest.raises(BudgetError):
            parse_budget(bad)


# ---------------------------------------------------------------------------
# check_budget: verdict + boundary rule
# ---------------------------------------------------------------------------

def test_check_budget_under():
    v = check_budget(0.004, 0.01)
    assert v == {
        "budget_usd": 0.01, "cost_usd": 0.004,
        "exceeded": False, "status": "ok",
    }


def test_check_budget_over():
    v = check_budget(0.015, 0.01)
    assert v["exceeded"] is True
    assert v["status"] == "exceeded"


def test_check_budget_exact_boundary_passes():
    # Documented rule: cost exactly at the budget passes.
    v = check_budget(0.01, 0.01)
    assert v["exceeded"] is False
    assert v["status"] == "ok"


# ---------------------------------------------------------------------------
# Config file: "budget" key
# ---------------------------------------------------------------------------

def test_config_accepts_budget(tmp_path):
    cfg = _echo_config(tmp_path, {"budget": 0.05})
    assert load_config(cfg).budget_usd == 0.05


def test_config_budget_defaults_none(tmp_path):
    assert load_config(_echo_config(tmp_path)).budget_usd is None


def test_config_rejects_bad_budget(tmp_path):
    for bad in (-1, "lots", True, float("nan")):
        with pytest.raises(ConfigError):
            load_config(_echo_config(tmp_path, {"budget": bad}))


# ---------------------------------------------------------------------------
# Reports carry the budget verdict
# ---------------------------------------------------------------------------

def test_terminal_shows_exceeded_line():
    text = report.terminal(_result([0.05]), check_budget(0.05, 0.01))
    assert "BUDGET EXCEEDED: $0.0500 > $0.0100" in text


def test_terminal_shows_ok_line():
    text = report.terminal(_result([0.004]), check_budget(0.004, 0.01))
    assert "BUDGET OK: $0.0040 of $0.0100 budget" in text


def test_terminal_has_no_budget_line_without_verdict():
    text = report.terminal(_result([0.004]))
    assert "BUDGET" not in text


def test_json_summary_carries_verdict():
    summary = report.to_dict(_result([0.05]), check_budget(0.05, 0.01))["summary"]
    assert summary["budget"]["budget_usd"] == 0.01
    assert summary["budget"]["cost_usd"] == 0.05
    assert summary["budget"]["exceeded"] is True


def test_json_summary_budget_none_without_verdict():
    summary = report.to_dict(_result([0.0]))["summary"]
    assert summary["budget"] is None


def test_html_shows_budget_banner():
    page = report.to_html(_result([0.05]), check_budget(0.05, 0.01))
    assert 'class="summary budget exceeded"' in page
    assert "exceeded budget" in page


def test_html_shows_ok_banner():
    page = report.to_html(_result([0.004]), check_budget(0.004, 0.01))
    assert 'class="summary budget ok"' in page
    assert "within budget" in page


def test_html_has_no_banner_without_verdict():
    page = report.to_html(_result([0.0]))
    assert 'class="summary budget' not in page


# ---------------------------------------------------------------------------
# CLI: --budget flag, exit codes, precedence (in-process with stubbed runner)
# ---------------------------------------------------------------------------

def test_cli_over_budget_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.05]))
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "0.01"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "BUDGET EXCEEDED: $0.0500 > $0.0100" in out


def test_cli_under_budget_exits_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.004]))
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "0.01"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "BUDGET OK: $0.0040 of $0.0100 budget" in out


def test_cli_exact_boundary_exits_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.01]))
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "0.01"])
    assert rc == 0
    assert "BUDGET OK" in capsys.readouterr().out


def test_cli_invalid_budget_exits_1(tmp_path, capsys):
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "nope"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cli_negative_budget_exits_1(tmp_path, capsys):
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "-1"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


def test_cli_config_budget_used_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.05]))
    rc = cli.main(["run", _echo_config(tmp_path, {"budget": 0.001})])
    assert rc == 2


def test_cli_flag_overrides_config_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.05]))
    rc = cli.main(
        ["run", _echo_config(tmp_path, {"budget": 0.001}), "--budget", "100"]
    )
    assert rc == 0
    assert "BUDGET OK" in capsys.readouterr().out


def test_cli_failed_cases_still_exit_2_with_ok_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.0], passed=False))
    rc = cli.main(["run", _echo_config(tmp_path), "--budget", "1.0"])
    assert rc == 2


def test_cli_json_and_html_include_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "run_eval", lambda config: _result([0.05]))
    cfg = _echo_config(tmp_path)
    json_path = tmp_path / "out.json"
    html_path = tmp_path / "out.html"
    rc = cli.main(["run", cfg, "--budget", "0.01",
                   "--json", str(json_path), "--html", str(html_path)])
    assert rc == 2
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["budget"]["exceeded"] is True
    page = html_path.read_text()
    assert 'class="summary budget exceeded"' in page


# ---------------------------------------------------------------------------
# CLI: real subprocess runs against the echo provider (cost $0.00)
# ---------------------------------------------------------------------------

def _subprocess_run(*argv, cwd):
    import os
    import subprocess
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "promptfoo_lite", *argv],
        cwd=cwd, capture_output=True, text=True, env=env,
    )


def test_subprocess_run_under_budget_exits_0(tmp_path):
    proc = _subprocess_run(
        "run", _echo_config(tmp_path), "--budget", "5", cwd=tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert "BUDGET OK" in proc.stdout


def test_subprocess_run_config_budget_exits_0(tmp_path):
    proc = _subprocess_run(
        "run", _echo_config(tmp_path, {"budget": 0.5}), cwd=tmp_path
    )
    assert proc.returncode == 0, proc.stderr
    assert "BUDGET OK" in proc.stdout


def test_subprocess_run_invalid_budget_exits_1(tmp_path):
    proc = _subprocess_run(
        "run", _echo_config(tmp_path), "--budget", "nope", cwd=tmp_path
    )
    assert proc.returncode == 1
    assert "error" in proc.stderr.lower()
