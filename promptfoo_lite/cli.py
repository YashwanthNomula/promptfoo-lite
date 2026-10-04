"""CLI: ``pfeval init`` · ``pfeval run config.json`` · ``pfeval list config.json``."""

from __future__ import annotations

import argparse
import json
import sys

from promptfoo_lite import __version__, assertions, load_config, run_eval
from promptfoo_lite import report as report_mod
from promptfoo_lite import templating

EXAMPLE_CONFIG = {
    "prompts": [
        {
            "id": "haiku",
            "label": "Haiku about a topic",
            "raw": "Write a haiku about {{topic}}.",
        },
        {
            "id": "pitch",
            "label": "One-sentence pitch",
            "raw": "Pitch {{topic}} to me in a single sentence.",
        },
    ],
    "providers": ["stub:base", "stub:creative"],
    "defaultTest": {
        "assert": [
            {"type": "not-contains", "values": ["sorry", "cannot", "as an AI"]},
            {"type": "latency", "threshold": 2000},
        ]
    },
    "tests": [
        {
            "description": "haiku mentions the topic",
            "vars": {"topic": "oceans"},
            "prompt": "haiku",
            "assert": [
                {"type": "icontains", "value": "ocean"},
                {"type": "word-count", "max": 30},
            ],
        },
        {
            "description": "pitch is short and on-topic",
            "vars": {"topic": "oceans"},
            "prompt": "pitch",
            "assert": [
                {"type": "icontains", "value": "ocean"},
                {"type": "word-count", "min": 5, "max": 25},
            ],
        },
    ],
}


def cmd_init(args: argparse.Namespace) -> int:
    path = args.path
    try:
        with open(path, "x", encoding="utf-8") as fh:
            json.dump(EXAMPLE_CONFIG, fh, indent=2)
            fh.write("\n")
    except FileExistsError:
        print(f"error: {path} already exists", file=sys.stderr)
        return 1
    print(f"wrote example eval config to {path}")
    print("run it with:  pfeval run " + path)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result = run_eval(config)
    print(report_mod.terminal(result))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(report_mod.to_json(result) + "\n")
        print(f"JSON results written to {args.json_out}")
    if args.html_out:
        with open(args.html_out, "w", encoding="utf-8") as fh:
            fh.write(report_mod.to_html(result))
        print(f"HTML report written to {args.html_out}")
    return 0 if result.failed == 0 else 2


def cmd_list(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"prompts ({len(config.prompts)}):")
    for p in config.prompts:
        vars_ = ", ".join(templating.variables_in(p.raw)) or "none"
        print(f"  {p.id:<14} {p.label}   [vars: {vars_}]")
    print(f"providers ({len(config.providers)}):")
    for p in config.providers:
        print(f"  {p.id}")
    print(f"tests ({len(config.tests)}):")
    for t in config.tests:
        scope = []
        if t.prompts:
            scope.append("prompts=" + ",".join(t.prompts))
        if t.providers:
            scope.append("providers=" + ",".join(t.providers))
        print(f"  {t.description}" + (f"  [{'; '.join(scope)}]" if scope else ""))
    from promptfoo_lite.runner import plan

    print(f"planned runs: {len(plan(config))}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfeval",
        description="promptfoo-lite: tiny LLM prompt-evaluation harness (zero deps)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="scaffold an example eval config")
    p_init.add_argument("path", nargs="?", default="promptfooconfig.json")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="run an eval config")
    p_run.add_argument("config", help="path to the eval JSON config")
    p_run.add_argument("--json", dest="json_out", help="write JSON results here")
    p_run.add_argument("--html", dest="html_out", help="write an HTML report here")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="inspect an eval config")
    p_list.add_argument("config", help="path to the eval JSON config")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
