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


def terminal(result: EvalResult) -> str:
    """Render a human-friendly terminal report."""
    lines: list[str] = []
    lines.append("")
    lines.append("promptfoo-lite eval report")
    lines.append("=" * 60)

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
        )

    lines.append("")
    lines.append("-" * 60)
    lines.append(
        f"passed {result.passed}/{result.total} "
        f"({result.pass_rate * 100:.1f}%) · "
        f"avg score {result.avg_score:.2f} · "
        f"{result.elapsed_s:.1f}s"
    )

    lines.append("\nby prompt:")
    prompt_map = {r.prompt.id: r.prompt for r in result.results}
    for pid, group in result.by_prompt().items():
        rate = sum(1 for r in group if r.passed) / len(group)
        label = _truncate(prompt_map[pid].label, 34)
        lines.append(f"  {_bar(rate)} {rate * 100:5.1f}%  {pid:<12} {label}")

    lines.append("\nby provider:")
    for pid, group in result.by_provider().items():
        rate = sum(1 for r in group if r.passed) / len(group)
        lines.append(f"  {_bar(rate)} {rate * 100:5.1f}%  {pid}")

    lines.append("")
    return "\n".join(lines)


def to_dict(result: EvalResult) -> dict:
    def _case(r: TestCaseResult) -> dict:
        return {
            "test": r.test.description,
            "prompt": r.prompt.id,
            "provider": r.provider.id,
            "rendered_prompt": r.rendered_prompt,
            "output": r.output,
            "latency_ms": round(r.latency_ms, 1),
            "tokens": r.tokens,
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

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "pass_rate": round(result.pass_rate, 4),
            "avg_score": round(result.avg_score, 4),
            "elapsed_s": round(result.elapsed_s, 2),
        },
        "results": [_case(r) for r in result.results],
    }


def to_json(result: EvalResult) -> str:
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False)


def to_html(result: EvalResult) -> str:
    """Standalone HTML report (no external assets)."""
    rows = []
    for r in result.results:
        cls = "pass" if r.passed else "fail"
        mark = PASS if r.passed else FAIL
        asserts = "<br>".join(
            f"<span class='{'ok' if a.passed else 'bad'}'>"
            f"{'✓' if a.passed else '✗'}</span> {html.escape(a.reason)}"
            for a in r.asserts
        ) or "<i>error</i>"
        rows.append(
            "<tr class='{cls}'>"
            "<td>{mark}</td><td>{desc}</td><td>{prompt}</td><td>{prov}</td>"
            "<td>{out}</td><td>{asserts}</td><td>{score:.2f}</td><td>{lat:.0f}ms</td>"
            "</tr>".format(
                cls=cls, mark=mark, desc=html.escape(r.test.description),
                prompt=html.escape(r.prompt.id), prov=html.escape(r.provider.id),
                out=html.escape(_truncate(r.output or (r.error or ""), 160)),
                asserts=asserts, score=r.score, lat=r.latency_ms,
            )
        )
    summary = result.passed, result.total, result.pass_rate * 100, result.avg_score
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
</style></head>
<body>
<h1>promptfoo-lite eval report</h1>
<p class="summary">Passed <b>{summary[0]}/{summary[1]}</b> ({summary[2]:.1f}%) ·
avg score <b>{summary[3]:.2f}</b></p>
<table><thead><tr>
<th></th><th>Test</th><th>Prompt</th><th>Provider</th>
<th>Output</th><th>Assertions</th><th>Score</th><th>Latency</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
