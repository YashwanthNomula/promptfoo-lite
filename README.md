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
    score 1.00 · 8ms · 30 tokens

[✓] case 2: haiku mentions the topic (prompt: haiku, provider: stub:creative)
    prompt → Write a haiku about oceans.
    output → Here's what you should know about oceans: there are a few angles worth …
    ✓ contains (case-insensitive) all expected strings
    ✓ 30 words (allowed 0..30)
    score 1.00 · 7ms · 30 tokens

[✗] case 3: pitch is short and on-topic (prompt: pitch, provider: stub:base)
    prompt → Pitch oceans to me in a single sentence.
    output → In summary about sentence: the core idea is surprisingly simple, and th…
    ✗ missing (case-insensitive) 'ocean'
    ✓ 25 words (allowed 5..25)
    score 0.50 · 34ms · 26 tokens

…

------------------------------------------------------------
passed 6/8 (75.0%) · avg score 0.81 · 0.0s

by prompt:
  ████████████████████ 100.0%  haiku        Haiku about a topic
  ██████████░░░░░░░░░░  50.0%  pitch        One-sentence pitch

by provider:
  ███████████████░░░░░  75.0%  stub:base
  ███████████████░░░░░  75.0%  stub:creative
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
  `starts-with`, `ends-with`, `word-count` (`min`/`max`), `char-count`
  (`min`/`max`), `similarity` (word-overlap vs a reference, `threshold`),
  `latency` (max ms). Each takes an optional `weight`; a test's score is the
  weighted average, and a test passes only if **all** its assertions pass
  (plus an optional per-test `threshold` on the score).

## CLI

```
pfeval init [path]                  scaffold an example config
pfeval run config.json [--json out.json] [--html report.html]
pfeval list config.json             show prompts, providers, tests, planned runs
```

`pfeval run` exits `0` when everything passes, `2` when any case fails —
drop it straight into CI. `--html` writes a standalone report page
(no external assets).

## Layout

```
promptfoo_lite/
  templating.py   {{var}} rendering
  config.py       JSON config loading + validation
  providers.py    stub / echo / OpenAI providers
  assertions.py   the 12 assertion graders
  runner.py       concurrent eval execution + scoring
  report.py       terminal / JSON / HTML reports
  cli.py          pfeval command
examples/         haiku_eval.json, support_bot_eval.json
tests/            47 tests, all green
```

## Tests

```bash
python -m pytest tests/ -q
```
