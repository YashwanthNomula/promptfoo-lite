"""Tests for the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _run(*argv, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "promptfoo_lite", *argv],
        cwd=cwd, capture_output=True, text=True, env=env,
    )


def _example_config(tmp_path):
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
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_init_scaffolds(tmp_path):
    proc = _run("init", "myeval.json", cwd=tmp_path)
    assert proc.returncode == 0
    assert (tmp_path / "myeval.json").exists()


def test_init_refuses_overwrite(tmp_path):
    (tmp_path / "myeval.json").write_text("{}")
    proc = _run("init", "myeval.json", cwd=tmp_path)
    assert proc.returncode == 1


def test_run_green(tmp_path):
    proc = _run("run", _example_config(tmp_path), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "passed 1/1" in proc.stdout


def test_run_red_exit_code(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({
        "prompts": ["Say {{thing}}."],
        "providers": ["echo"],
        "tests": [{
            "description": "will fail",
            "vars": {"thing": "hi"},
            "assert": [{"type": "contains", "value": "nope"}],
        }],
    }))
    proc = _run("run", str(path), cwd=tmp_path)
    assert proc.returncode == 2
    assert "passed 0/1" in proc.stdout


def test_run_writes_json_and_html(tmp_path):
    cfg = _example_config(tmp_path)
    proc = _run("run", cfg, "--json", "out.json", "--html", "out.html", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "out.html").exists()
    assert json.loads((tmp_path / "out.json").read_text())["summary"]["passed"] == 1


def test_run_bad_config_exit_code(tmp_path):
    proc = _run("run", str(tmp_path / "missing.json"), cwd=tmp_path)
    assert proc.returncode == 1
    assert "error" in proc.stderr.lower()


def test_list_shows_plan(tmp_path):
    proc = _run("list", _example_config(tmp_path), cwd=tmp_path)
    assert proc.returncode == 0
    assert "planned runs: 1" in proc.stdout
