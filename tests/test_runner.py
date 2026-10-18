"""Tests for config loading, the runner, and reports."""

import json

import pytest

from promptfoo_lite import load_config, run_eval
from promptfoo_lite.config import ConfigError
from promptfoo_lite.runner import plan
from promptfoo_lite import report as report_mod


def _write(tmp_path, data):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(data))
    return str(path)


def _minimal(**over):
    cfg = {
        "prompts": ["Say {{thing}}."],
        "providers": ["echo"],
        "tests": [
            {
                "description": "echo repeats the prompt",
                "vars": {"thing": "hi"},
                "assert": [{"type": "contains", "value": "hi"}],
            }
        ],
    }
    cfg.update(over)
    return cfg


def test_load_minimal(tmp_path):
    cfg = load_config(_write(tmp_path, _minimal()))
    assert len(cfg.prompts) == 1 and len(cfg.providers) == 1 and len(cfg.tests) == 1


def test_load_rejects_bad_config(tmp_path):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {"prompts": []}))
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, _minimal(tests=[])))
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "missing.json"))


def test_load_rejects_unknown_selector(tmp_path):
    bad = _minimal(tests=[{"vars": {}, "prompt": "nope", "assert": []}])
    with pytest.raises(ConfigError, match="unknown prompt"):
        load_config(_write(tmp_path, bad))


def test_default_test_applies(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            _minimal(
                defaultTest={"assert": [{"type": "contains", "value": "hi"}]},
                tests=[{"description": "t", "vars": {"thing": "hi"}}],
            ),
        )
    )
    assert len(cfg.tests[0].assert_) == 1


def test_plan_expands_cartesian(tmp_path):
    data = _minimal()
    data["prompts"] = ["a {{x}}", "b {{x}}"]
    data["providers"] = ["echo", "stub"]
    cfg = load_config(_write(tmp_path, data))
    assert len(plan(cfg)) == 1 * 2 * 2


def test_plan_respects_selectors(tmp_path):
    data = _minimal()
    data["prompts"] = [{"id": "p1", "raw": "a {{x}}"}, {"id": "p2", "raw": "b {{x}}"}]
    data["tests"] = [{"description": "t", "vars": {"x": "1"}, "prompt": "p2", "assert": []}]
    cfg = load_config(_write(tmp_path, data))
    runs = plan(cfg)
    assert len(runs) == 1 and runs[0][1].id == "p2"


def test_run_eval_all_pass(tmp_path):
    result = run_eval(load_config(_write(tmp_path, _minimal())))
    assert result.total == 1 and result.passed == 1
    assert result.results[0].output == "Say hi."


def test_run_eval_failure_recorded(tmp_path):
    data = _minimal()
    data["tests"][0]["assert"] = [{"type": "contains", "value": "definitely-not-there"}]
    result = run_eval(load_config(_write(tmp_path, data)))
    assert result.failed == 1 and not result.results[0].passed


def test_run_eval_template_error_is_a_failing_case(tmp_path):
    data = _minimal()
    data["tests"][0]["vars"] = {}  # {{thing}} missing
    result = run_eval(load_config(_write(tmp_path, data)))
    assert result.failed == 1
    assert "template error" in (result.results[0].error or "")


def test_run_eval_stub_deterministic(tmp_path):
    data = _minimal()
    data["providers"] = ["stub:seed"]
    data["tests"][0]["assert"] = [{"type": "contains", "value": "thing"}]
    r1 = run_eval(load_config(_write(tmp_path, data)))
    r2 = run_eval(load_config(_write(tmp_path, data)))
    assert r1.results[0].output == r2.results[0].output


def test_threshold_gates_pass(tmp_path):
    data = _minimal()
    data["tests"][0]["assert"] = [
        {"type": "contains", "value": "hi", "weight": 1},
        {"type": "contains", "value": "missing", "weight": 1},
    ]
    data["tests"][0]["threshold"] = 0.4  # score 0.5 clears it, but an assert failed
    result = run_eval(load_config(_write(tmp_path, data)))
    assert not result.results[0].passed  # all asserts must pass


def test_terminal_report_contains_summary(tmp_path):
    result = run_eval(load_config(_write(tmp_path, _minimal())))
    text = report_mod.terminal(result)
    assert "passed 1/1" in text and "by prompt:" in text


def test_json_report_shape(tmp_path):
    result = run_eval(load_config(_write(tmp_path, _minimal())))
    payload = json.loads(report_mod.to_json(result))
    assert payload["summary"]["total"] == 1
    assert payload["results"][0]["passed"] is True


def test_html_report_is_standalone(tmp_path):
    result = run_eval(load_config(_write(tmp_path, _minimal())))
    page = report_mod.to_html(result)
    assert "<html" in page and "passed" in page.lower()


# ---- --filter ---------------------------------------------------------------

def _two_case_config():
    return {
        "prompts": [
            {"id": "haiku", "label": "Haiku about a topic", "raw": "Haiku {{topic}}."},
            {"id": "pitch", "label": "One-sentence pitch", "raw": "Pitch {{topic}}."},
        ],
        "providers": ["echo", "stub:base"],
        "tests": [
            {
                "description": "haiku mentions the topic",
                "vars": {"topic": "oceans"},
                "prompt": "haiku",
                "assert": [{"type": "icontains", "value": "ocean"}],
            },
            {
                "description": "pitch is short",
                "vars": {"topic": "oceans"},
                "prompt": "pitch",
                "assert": [{"type": "icontains", "value": "ocean"}],
            },
        ],
    }


def test_plan_filter_by_description(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    assert len(plan(cfg)) == 4  # 2 tests x 1 prompt each x 2 providers
    runs = plan(cfg, "haiku")
    assert len(runs) == 2
    assert all(r[0].description.startswith("haiku") for r in runs)


def test_plan_filter_by_provider(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    runs = plan(cfg, "stub:base")
    assert len(runs) == 2
    assert all(r[2].id == "stub:base" for r in runs)


def test_plan_filter_is_case_insensitive_and_matches_label(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    assert len(plan(cfg, "HAIKU")) == 2
    assert len(plan(cfg, "one-sentence")) == 2  # prompt label, not id


def test_plan_filter_blank_means_no_filter(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    assert len(plan(cfg, "   ")) == 4


def test_plan_filter_no_match_is_empty(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    assert plan(cfg, "definitely-not-there") == []


def test_run_eval_filter_records_filter_and_planned_total(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    result = run_eval(cfg, filter_text="pitch")
    assert result.total == 2
    assert result.filter_text == "pitch"
    assert result.planned_total == 4
    assert all(r.test.description == "pitch is short" for r in result.results)


def test_run_eval_without_filter_has_no_filter_metadata(tmp_path):
    result = run_eval(load_config(_write(tmp_path, _minimal())))
    assert result.filter_text is None
    assert result.planned_total is None


def test_terminal_report_shows_filter_line(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    text = report_mod.terminal(run_eval(cfg, filter_text="pitch"))
    assert "filter 'pitch': 2 of 4 planned runs" in text


def test_json_report_includes_filter(tmp_path):
    cfg = load_config(_write(tmp_path, _two_case_config()))
    payload = json.loads(report_mod.to_json(run_eval(cfg, filter_text="pitch")))
    assert payload["summary"]["filter"] == {"text": "pitch", "planned_total": 4}
    assert payload["summary"]["total"] == 2


def test_json_report_without_filter_has_no_filter_key(tmp_path):
    payload = json.loads(report_mod.to_json(run_eval(load_config(_write(tmp_path, _minimal())))))
    assert "filter" not in payload["summary"]
