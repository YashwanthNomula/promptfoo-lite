"""Terminal, JSON and HTML reports for eval results."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from promptfoo_lite.runner import EvalResult, TestCaseResult

PASS = "✓"
FAIL = "✗"


def _bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def terminal(result: EvalResult, budget_verdict: dict | None = None) -> str:
    """Render a human-friendly terminal report.

    When ``budget_verdict`` (from ``promptfoo_lite.budget.check_budget``) is
    given, a ``BUDGET EXCEEDED`` / ``BUDGET OK`` line is appended after the
    cost summary.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("promptfoo-lite eval report")
    lines.append("=" * 60)
    if result.filter_text is not None and result.planned_total is not None:
        lines.append(
            f"filter {result.filter_text!r}: {result.total} of "
            f"{result.planned_total} planned runs"
        )

    for i, r in enumerate(result.results, 1):
        mark = PASS if r.passed else FAIL
        lines.append(
            f"\n[{mark}] case {i}: {r.test.description} "
            f"(prompt: {r.prompt.id}, provider: {r.provider.id})"
        )
        lines.append(f"    prompt → {_truncate(r.rendered_prompt, 72)}")
        if r.error:
            lines.append(f"    ERROR: {r.error}")
            continue
        lines.append(f"    output → {_truncate(r.output, 72)}")
        for a in r.asserts:
            amark = PASS if a.passed else FAIL
            lines.append(f"    {amark} {a.reason}")
        lines.append(
            f"    score {r.score:.2f} · {r.latency_ms:.0f}ms"
            + (f" · {r.tokens} tokens" if r.tokens is not None else "")
            + (_fmt_cost(r.cost_usd) if r.cost_usd is not None else "")
        )

    lines.append("")
    lines.append("-" * 60)
    stats = result.stats()
    lat = stats["latency_ms"]
    lines.append(
        f"passed {result.passed}/{result.total} "
        f"({result.pass_rate * 100:.1f}%) · "
        f"avg score {result.avg_score:.2f} · "
        f"{result.elapsed_s:.1f}s"
    )
    lines.append(
        f"latency mean {lat['mean']:.0f}ms · p50 {lat['p50']:.0f}ms · "
        f"p95 {lat['p95']:.0f}ms"
    )
    lines.append(
        f"{stats['tokens']['total']} tokens"
        f" · est. cost ${stats['cost_usd']['total']:.4f}"
    )
    if budget_verdict is not None:
        if budget_verdict["exceeded"]:
            lines.append(
                f"BUDGET EXCEEDED: ${budget_verdict['cost_usd']:.4f} > "
                f"${budget_verdict['budget_usd']:.4f}"
            )
        else:
            lines.append(
                f"BUDGET OK: ${budget_verdict['cost_usd']:.4f} of "
                f"${budget_verdict['budget_usd']:.4f} budget"
            )

    lines.append("\nby prompt:")
    prompt_map = {r.prompt.id: r.prompt for r in result.results}
    for pid, group in result.by_prompt().items():
        rate = sum(1 for r in group if r.passed) / len(group)
        label = _truncate(prompt_map[pid].label, 34)
        lines.append(f"  {_bar(rate)} {rate * 100:5.1f}%  {pid:<12} {label}")

    lines.append("\nby provider:")
    stats = result.stats()
    for pid, group in result.by_provider().items():
        rate = sum(1 for r in group if r.passed) / len(group)
        tokens = stats["tokens"]["by_provider"].get(pid, 0)
        cost = stats["cost_usd"]["by_provider"].get(pid, 0.0)
        lines.append(
            f"  {_bar(rate)} {rate * 100:5.1f}%  {pid:<14} "
            f"{tokens} tokens · est. ${cost:.4f}"
        )

    lines.append("")
    return "\n".join(lines)


def _fmt_cost(cost_usd: float) -> str:
    return f" · ${cost_usd:.6f}"


def to_dict(result: EvalResult, budget_verdict: dict | None = None) -> dict:
    def _case(r: TestCaseResult) -> dict:
        return {
            "test": r.test.description,
            "prompt": r.prompt.id,
            "provider": r.provider.id,
            "rendered_prompt": r.rendered_prompt,
            "output": r.output,
            "latency_ms": round(r.latency_ms, 1),
            "tokens": r.tokens,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "cost_usd": round(r.cost_usd, 6) if r.cost_usd is not None else None,
            "score": round(r.score, 4),
            "passed": r.passed,
            "error": r.error,
            "asserts": [
                {
                    "passed": a.passed,
                    "score": round(a.score, 4),
                    "reason": a.reason,
                    "weight": a.weight,
                }
                for a in r.asserts
            ],
        }

    stats = result.stats()
    summary = {
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "pass_rate": round(result.pass_rate, 4),
        "avg_score": round(result.avg_score, 4),
        "elapsed_s": round(result.elapsed_s, 2),
        "latency_ms": {
            k: round(v, 1) for k, v in stats["latency_ms"].items()
        },
        "tokens": stats["tokens"],
        "cost_usd": stats["cost_usd"],
        "budget": budget_verdict,
    }
    if result.filter_text is not None:
        summary["filter"] = {
            "text": result.filter_text,
            "planned_total": result.planned_total,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "results": [_case(r) for r in result.results],
    }


def to_json(result: EvalResult, budget_verdict: dict | None = None) -> str:
    return json.dumps(to_dict(result, budget_verdict), indent=2, ensure_ascii=False)


def to_html(result: EvalResult, budget_verdict: dict | None = None) -> str:
    """Standalone HTML report (no external assets).

    When ``budget_verdict`` (from ``promptfoo_lite.budget.check_budget``) is
    given, a green/red budget banner is shown under the summary.
    """
    rows = []
    for r in result.results:
        cls = "pass" if r.passed else "fail"
        mark = PASS if r.passed else FAIL
        asserts = "<br>".join(
            f"<span class='{'ok' if a.passed else 'bad'}'>"
            f"{'✓' if a.passed else '✗'}</span> {html.escape(a.reason)}"
            for a in r.asserts
        ) or "<i>error</i>"
        cost = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "—"
        rows.append(
            "<tr class='{cls}'>"
            "<td>{mark}</td><td>{desc}</td><td>{prompt}</td><td>{prov}</td>"
            "<td>{out}</td><td>{asserts}</td><td>{score:.2f}</td><td>{lat:.0f}ms</td>"
            "<td>{cost}</td>"
            "</tr>".format(
                cls=cls, mark=mark, desc=html.escape(r.test.description),
                prompt=html.escape(r.prompt.id), prov=html.escape(r.provider.id),
                out=html.escape(_truncate(r.output or (r.error or ""), 160)),
                asserts=asserts, score=r.score, lat=r.latency_ms, cost=cost,
            )
        )
    stats = result.stats()
    lat = stats["latency_ms"]
    summary = result.passed, result.total, result.pass_rate * 100, result.avg_score
    budget_html = ""
    if budget_verdict is not None:
        cls = "exceeded" if budget_verdict["exceeded"] else "ok"
        verb = "exceeded" if budget_verdict["exceeded"] else "within"
        budget_html = (
            f'<p class="summary budget {cls}">Budget '
            f'<b>${budget_verdict["budget_usd"]:.4f}</b>: '
            f'estimated cost <b>${budget_verdict["cost_usd"]:.4f}</b> — '
            f'{verb} budget</p>'
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>promptfoo-lite eval report</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 2rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
.summary {{ font-size: 1.1rem; margin-bottom: 1rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
tr.pass td:first-child {{ color: #1a7f37; font-weight: bold; }}
tr.fail td:first-child {{ color: #cf222e; font-weight: bold; }}
tr.fail {{ background: #fff5f5; }}
.ok {{ color: #1a7f37; }} .bad {{ color: #cf222e; }}
.budget {{ padding: 0.5rem 0.75rem; border-radius: 4px; font-size: 1rem; }}
.budget.ok {{ background: #e9f7ef; color: #1a7f37; }}
.budget.exceeded {{ background: #fdecec; color: #cf222e; font-weight: bold; }}
</style></head>
<body>
<h1>promptfoo-lite eval report</h1>
<p class="summary">Passed <b>{summary[0]}/{summary[1]}</b> ({summary[2]:.1f}%) ·
avg score <b>{summary[3]:.2f}</b></p>
<p class="summary">Latency mean <b>{lat["mean"]:.0f}ms</b> ·
p50 <b>{lat["p50"]:.0f}ms</b> · p95 <b>{lat["p95"]:.0f}ms</b> ·
max <b>{lat["max"]:.0f}ms</b><br>
{stats["tokens"]["total"]} tokens across
{len(stats["tokens"]["by_provider"])} provider(s) ·
estimated cost <b>${stats["cost_usd"]["total"]:.4f}</b></p>
{budget_html}
<table><thead><tr>
<th></th><th>Test</th><th>Prompt</th><th>Provider</th>
<th>Output</th><th>Assertions</th><th>Score</th><th>Latency</th><th>Cost</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
