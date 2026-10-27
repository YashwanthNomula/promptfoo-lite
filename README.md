# promptfoo-lite

A tiny LLM prompt-evaluation harness in pure Python — **zero dependencies**.
Define prompts, test cases, and assertions in one JSON file, run them against
any provider, and get a graded report in your terminal (or JSON / HTML).

The killer feature: a **built-in deterministic stub model** (`stub:<seed>`),
so you can demo and develop evals with no API key. Swap in a real model later
via the OpenAI-compatible provider (`openai:<model>`, needs `OPENAI_API_KEY`).

## Quickstart

```bash
python -m promptfoo_lite init myeval.json   # scaffold an example config
python -m promptfoo_lite run myeval.json    # run it
```

That's it. `pip install -e .` also gives you the `pfeval` command.

## Real demo transcript

The bundled `examples/haiku_eval.json` pits two prompt variants against two
stub "models" across 3 test cases (8 runs). The eval catches something real:
the pitch prompt makes the model drift off-topic — exactly what evals are for.

```
$ python -m promptfoo_lite run examples/haiku_eval.json

promptfoo-lite eval report
============================================================

[✓] case 1: haiku mentions the topic (prompt: haiku, provider: stub:base)
    prompt → Write a haiku about oceans.
    output → Here's what you should know about oceans: the short answer is yes, with…
    ✓ contains (case-insensitive) all expected strings
    ✓ 30 words (allowed 0..30)
    score 1.00 · 8ms · 35 tokens · $0.000000

[✓] case 2: haiku mentions the topic (prompt: haiku, provider: stub:creative)
    prompt → Write a haiku about oceans.
    output → Here's what you should know about oceans: there are a few angles worth …
    ✓ contains (case-insensitive) all expected strings
    ✓ 30 words (allowed 0..30)
    score 1.00 · 7ms · 35 tokens · $0.000000

[✗] case 3: pitch is short and on-topic (prompt: pitch, provider: stub:base)
    prompt → Pitch oceans to me in a single sentence.
    output → In summary about sentence: the core idea is surprisingly simple, and th…
    ✗ missing (case-insensitive) 'ocean'
    ✓ 25 words (allowed 5..25)
    score 0.50 · 34ms · 34 tokens · $0.000000

…

------------------------------------------------------------
passed 6/8 (75.0%) · avg score 0.81 · 0.0s
latency mean 22ms · p50 27ms · p95 33ms
283 tokens · est. cost $0.0000

by prompt:
  ████████████████████ 100.0%  haiku        Haiku about a topic
  ██████████░░░░░░░░░░  50.0%  pitch        One-sentence pitch

by provider:
  ███████████████░░░░░  75.0%  stub:base       141 tokens · est. $0.0000
  ███████████████░░░░░  75.0%  stub:creative   142 tokens · est. $0.0000
```

Try the support-bot example too — it ships with one deliberately failing case:

```
$ python -m promptfoo_lite run examples/support_bot_eval.json
```

## Writing an eval

```json
{
  "prompts": [
    {"id": "haiku", "label": "Haiku about a topic", "raw": "Write a haiku about {{topic}}."}
  ],
  "providers": ["stub:base", "stub:creative", "openai:gpt-4o-mini"],
  "defaultTest": {
    "assert": [
      {"type": "not-contains", "values": ["sorry", "cannot", "as an AI"]},
      {"type": "latency", "threshold": 2000}
    ]
  },
  "tests": [
    {
      "description": "haiku mentions the topic",
      "vars": {"topic": "oceans"},
      "prompt": "haiku",
      "assert": [
        {"type": "icontains", "value": "ocean"},
        {"type": "word-count", "max": 30}
      ]
    }
  ]
}
```

- **prompts**: strings or `{id, label, raw}` objects. `{{var}}` placeholders are
  filled from each test's `vars` (dotted names like `{{user.name}}` work).
- **providers**: `stub[:seed]` (deterministic, no key), `echo` (returns the
  prompt verbatim — great for testing the harness itself), `openai[:model]`.
- **tests**: each runs against every prompt × provider unless scoped with
  `"prompt"` / `"provider"`. Every test inherits `defaultTest.assert`.
- **assertions**: `contains`, `icontains`, `not-contains`, `equals`, `regex`,
  `not-regex` (fails if the pattern matches — use it to forbid leaked
  secrets or boilerplate in outputs),
  `starts-with`, `ends-with`, `word-count` (`min`/`max`), `char-count`
  (`min`/`max`), `similarity` (word-overlap vs a reference, `threshold`),
  `latency` (max ms), `cost` (max USD per case). Each takes an optional
  `weight`; a test's score is the weighted average, and a test passes only if
  **all** its assertions pass (plus an optional per-test `threshold` on the
  score).

## Cost and latency stats

Every provider reports per-case **latency**, **token usage**
(prompt/completion split where the API provides it), and an **estimated USD
cost**:

- `openai:*` parses real `usage` from the API response and prices it against a
  per-model rate table (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`, `gpt-4.1`,
  `o3-mini`, `gpt-3.5-turbo`; unknown models fall back to a blended
  $0.60/1M-token estimate). These are ballpark public rates — check current
  pricing before quoting them.
- `stub:*` / `echo` synthesize token counts and report `$0.00`, so every
  report has the same shape with no API key needed.

All three reports surface aggregate stats: mean / p50 / p95 / max latency,
total tokens, and estimated cost overall plus per provider. The JSON report
carries the full breakdown under `summary.latency_ms`, `summary.tokens`, and
`summary.cost_usd`, and each case gets `cost_usd`, `prompt_tokens`, and
`completion_tokens` — handy for tracking eval spend in CI over time.

```json
{"type": "cost", "threshold": 0.005}
```

The `cost` assertion fails any case whose estimated cost exceeds the limit
(in USD) — pair it with `latency` in `defaultTest` to guard both budgets.

## Token-budget gate

Cap what an eval is allowed to spend per run. Set a budget in the config
or on the command line:

```json
{
  "prompts": ["Write a haiku about {{topic}}."],
  "providers": ["openai:gpt-4o-mini"],
  "budget": 0.50,
  "tests": [...]
}
```

```bash
pfeval run eval.json --budget 0.50
```

After the run, the estimated total cost (from `EvalResult.stats()`) is
compared against the budget:

- **at or under budget** → passes; the report shows
  `BUDGET OK: $0.1234 of $0.5000 budget`
- **over budget** → gate failure: the report shows
  `BUDGET EXCEEDED: $0.6123 > $0.5000` and `pfeval run` exits `2`
  (the same code as failed cases and diff regressions — gate on it in CI)

The `--budget` flag overrides the config's `"budget"` key. Both are
validated fail-closed: a negative, non-numeric, or infinite budget is
rejected (bad flag → exit `1`; bad config → `error:` + exit `1`) instead
of silently disabling the gate.

The budget verdict also ships in the machine-readable reports:
`summary.budget` (`{budget_usd, cost_usd, exceeded, status}`) in the JSON
report, and a green/red budget banner in the HTML report.

## CLI

```
pfeval init [path]                  scaffold an example config
pfeval run config.json [--json out.json] [--html report.html] [--budget DOLLARS] [--filter TEXT]
pfeval list config.json             show prompts, providers, tests, planned runs
pfeval diff old.json new.json [--json diff.json]
                                    compare two eval JSON reports case by case
```

`pfeval run` exits `0` when everything passes, `2` when any case fails or
the token budget is exceeded (see *Token-budget gate* below), and `1` on
bad input — drop it straight into CI. `--html` writes a standalone report
page (no external assets).

`--filter TEXT` runs only the cases whose test description, prompt
id/label, or provider id/label contains `TEXT` (case-insensitive) — the
fast loop for debugging one failing case without paying for the whole
suite. The report header shows `filter 'haiku': 2 of 8 planned runs`, and
the JSON summary records the filter plus the unfiltered plan size. A
filter that matches nothing is an error (exit `1`), so a typo never
silently reports "0/0 passed".

## Regression diffing

`pfeval diff` compares two `run --json` reports and answers the question
CI cares about: *what changed between these two runs?*

```bash
pfeval run eval.json --json old.json
# ... tweak the prompt ...
pfeval run eval.json --json new.json
pfeval diff old.json new.json
```

Cases are matched by (test, prompt, provider). Each matched case reports
pass/fail change plus score, latency (ms and %), token, and cost deltas;
added/removed cases are listed separately rather than counted as
regressions. The summary shows the pass-rate delta in percentage points
and counts of newly failing vs. fixed cases.

`pfeval diff` exits `2` when any case newly fails (regression — gate on it
in CI), `0` when nothing regressed, and `1` on bad input (not an eval
JSON, unreadable file, or zero overlapping cases).

## Layout

```
promptfoo_lite/
  templating.py   {{var}} rendering
  config.py       JSON config loading + validation
  providers.py    stub / echo / OpenAI providers
  assertions.py   the 13 assertion graders
  runner.py       concurrent eval execution + scoring + latency/token/cost stats
  budget.py       token-budget gate: parse/validate a USD budget, compare run cost
  report.py       terminal / JSON / HTML reports
  diff.py         run-to-run regression diff (diff command)
  cli.py          pfeval command
examples/         haiku_eval.json, support_bot_eval.json
tests/            121 tests, all green
```

## Tests

```bash
python -m pytest tests/ -q
```
