"""Run-to-run regression diff: compare two eval JSON reports case by case.

Loads two ``pfeval run --json`` outputs, matches cases by (test, prompt,
provider), and reports pass/fail changes plus latency, token and cost
deltas, with an aggregate summary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

PASS = "✓"
FAIL = "✗"


class DiffError(Exception):
    """Raised when a diff input is not a valid eval JSON report."""


def load_run(path: str) -> dict:
    """Load and validate an eval JSON report (as written by ``--json``)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise DiffError(f"no such file: {path}")
    except json.JSONDecodeError as exc:
        raise DiffError(f"{path} is not valid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("summary"), dict):
        raise DiffError(f"{path} is not an eval JSON report (missing 'summary')")
    if not isinstance(data.get("results"), list):
        raise DiffError(f"{path} is not an eval JSON report (missing 'results')")
    return data


def case_name(case: dict) -> str:
    return (
        f"{case.get('test', '?')} "
        f"(prompt: {case.get('prompt', '?')}, "
        f"provider: {case.get('provider', '?')})"
    )


def _index(results: list[dict]) -> dict[tuple, dict]:
    """Map cases by (test, prompt, provider), de-duplicating repeats."""
    index: dict[tuple, dict] = {}
    for case in results:
        base = (case.get("test"), case.get("prompt"), case.get("provider"))
        key: tuple = base
        n = 1
        while key in index:
            n += 1
            key = (*base, n)
        index[key] = case
    return index


def _num(value) -> float | None:
    return value if isinstance(value, (int, float)) else None


def _pct_delta(old: float, new: float) -> float | None:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def _compare_case(old: dict, new: dict) -> dict:
    old_passed = bool(old.get("passed"))
    new_passed = bool(new.get("passed"))
    if old_passed and not new_passed:
        status = "regression"
    elif not old_passed and new_passed:
        status = "fixed"
    elif new_passed:
        status = "still-passing"
    else:
        status = "still-failing"

    old_lat, new_lat = _num(old.get("latency_ms")), _num(new.get("latency_ms"))
    lat_delta = new_lat - old_lat if old_lat is not None and new_lat is not None else None
    lat_pct = (
        _pct_delta(old_lat, new_lat)
        if old_lat is not None and new_lat is not None
        else None
    )
    old_tok, new_tok = _num(old.get("tokens")), _num(new.get("tokens"))
    old_cost, new_cost = _num(old.get("cost_usd")), _num(new.get("cost_usd"))
    old_score, new_score = _num(old.get("score")), _num(new.get("score"))

    return {
        "name": case_name(new),
        "old_passed": old_passed,
        "new_passed": new_passed,
        "status": status,
        "score": {
            "old": old_score,
            "new": new_score,
            "delta": new_score - old_score
            if old_score is not None and new_score is not None
            else None,
        },
        "latency_ms": {
            "old": old_lat,
            "new": new_lat,
            "delta": lat_delta,
            "delta_pct": round(lat_pct, 1) if lat_pct is not None else None,
        },
        "tokens": {
            "old": old_tok,
            "new": new_tok,
            "delta": new_tok - old_tok
            if old_tok is not None and new_tok is not None
            else None,
        },
        "cost_usd": {
            "old": old_cost,
            "new": new_cost,
            "delta": round(new_cost - old_cost, 6)
            if old_cost is not None and new_cost is not None
            else None,
        },
    }


def diff_runs(old: dict, new: dict) -> dict:
    """Diff two validated eval report dicts. Raises DiffError on no overlap."""
    old_idx = _index(old["results"])
    new_idx = _index(new["results"])

    matched_keys = sorted(
        set(old_idx) & set(new_idx), key=lambda k: str(k)
    )
    added = [case_name(new_idx[k]) for k in sorted(set(new_idx) - set(old_idx), key=str)]
    removed = [case_name(old_idx[k]) for k in sorted(set(old_idx) - set(new_idx), key=str)]

    if not matched_keys:
        raise DiffError(
            "no overlapping cases between the two runs — nothing to compare"
        )

    cases = [_compare_case(old_idx[k], new_idx[k]) for k in matched_keys]

    regressions = [c["name"] for c in cases if c["status"] == "regression"]
    fixed = [c["name"] for c in cases if c["status"] == "fixed"]
    still_passing = sum(1 for c in cases if c["status"] == "still-passing")
    still_failing = sum(1 for c in cases if c["status"] == "still-failing")

    old_pass = sum(1 for c in cases if c["old_passed"])
    new_pass = sum(1 for c in cases if c["new_passed"])
    n = len(cases)

    def _sum(key: str) -> float | None:
        deltas = [c[key]["delta"] for c in cases if c[key]["delta"] is not None]
        return round(sum(deltas), 6) if deltas else None

    summary = {
        "compared": n,
        "added": len(added),
        "removed": len(removed),
        "old_pass_rate": round(old_pass / n, 4),
        "new_pass_rate": round(new_pass / n, 4),
        "pass_rate_delta_pp": round((new_pass - old_pass) / n * 100, 1),
        "regressions": regressions,
        "fixed": fixed,
        "still_passing": still_passing,
        "still_failing": still_failing,
        "latency_ms_delta": _sum("latency_ms"),
        "tokens_delta": _sum("tokens"),
        "cost_usd_delta": _sum("cost_usd"),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "cases": cases,
        "added": added,
        "removed": removed,
    }


def to_json(diff: dict) -> str:
    return json.dumps(diff, indent=2, ensure_ascii=False)


def _signed(value: float | None, precision: int = 0, unit: str = "") -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.{precision}f}{unit}"


def terminal(diff: dict) -> str:
    """Render a human-friendly terminal report of a run diff."""
    status_marks = {
        "regression": f"[{PASS}→{FAIL}] REGRESSION",
        "fixed": f"[{FAIL}→{PASS}] fixed",
        "still-passing": f"[{PASS}→{PASS}]",
        "still-failing": f"[{FAIL}→{FAIL}]",
    }
    order = ["regression", "fixed", "still-failing", "still-passing"]

    lines: list[str] = []
    lines.append("")
    lines.append("promptfoo-lite run diff")
    lines.append("=" * 60)

    for status in order:
        for c in diff["cases"]:
            if c["status"] != status:
                continue
            lat = c["latency_ms"]
            tok = c["tokens"]
            cost = c["cost_usd"]
            score = c["score"]
            lat_pct = (
                f", {_signed(lat['delta_pct'], 1, '%')}"
                if lat["delta_pct"] is not None
                else ""
            )
            lines.append(f"\n{status_marks[status]}  {c['name']}")
            parts = []
            if score["old"] is not None and score["new"] is not None:
                parts.append(
                    f"score {score['old']:.2f} → {score['new']:.2f}"
                )
            if lat["old"] is not None and lat["new"] is not None:
                parts.append(
                    f"latency {lat['old']:.0f}ms → {lat['new']:.0f}ms"
                    f" ({_signed(lat['delta'], 0, 'ms')}{lat_pct})"
                )
            if tok["old"] is not None and tok["new"] is not None:
                parts.append(
                    f"tokens {tok['old']:.0f} → {tok['new']:.0f}"
                    f" ({_signed(tok['delta'], 0)})"
                )
            if cost["old"] is not None and cost["new"] is not None:
                dc = cost["delta"]
                csign = "+" if dc >= 0 else "-"
                parts.append(
                    f"cost ${cost['old']:.6f} → ${cost['new']:.6f}"
                    f" ({csign}${abs(dc):.6f})"
                )
            lines.append("    " + " · ".join(parts))

    if diff["added"]:
        lines.append(f"\nadded in new run ({len(diff['added'])}):")
        for name in diff["added"]:
            lines.append(f"  + {name}")
    if diff["removed"]:
        lines.append(f"\nremoved from new run ({len(diff['removed'])}):")
        for name in diff["removed"]:
            lines.append(f"  - {name}")

    s = diff["summary"]
    lines.append("")
    lines.append("-" * 60)
    lines.append(
        f"compared {s['compared']} case(s)"
        + (f" · {s['added']} added" if s["added"] else "")
        + (f" · {s['removed']} removed" if s["removed"] else "")
    )
    lines.append(
        f"pass rate {s['old_pass_rate'] * 100:.1f}% → {s['new_pass_rate'] * 100:.1f}%"
        f" ({_signed(s['pass_rate_delta_pp'], 1, ' pp')})"
    )
    lines.append(
        f"newly failing: {len(s['regressions'])} · "
        f"fixed: {len(s['fixed'])} · "
        f"still passing: {s['still_passing']} · "
        f"still failing: {s['still_failing']}"
    )
    totals = []
    if s["latency_ms_delta"] is not None:
        totals.append(f"latency {_signed(s['latency_ms_delta'], 0, 'ms')}")
    if s["tokens_delta"] is not None:
        totals.append(f"tokens {_signed(s['tokens_delta'], 0)}")
    if s["cost_usd_delta"] is not None:
        totals.append(f"cost {_signed(s['cost_usd_delta'], 6)}")
    if totals:
        lines.append("total deltas: " + " · ".join(totals))
    lines.append("")
    return "\n".join(lines)
