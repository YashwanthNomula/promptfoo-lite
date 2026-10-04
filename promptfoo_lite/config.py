"""Eval configuration loading and validation (JSON, stdlib only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


class ConfigError(ValueError):
    """Raised when an eval config file is invalid."""


@dataclass
class Prompt:
    id: str
    label: str
    raw: str


@dataclass
class ProviderSpec:
    id: str
    label: str


@dataclass
class AssertionSpec:
    type: str
    value: object = None
    values: list = field(default_factory=list)
    threshold: float | None = None
    min: int | None = None
    max: int | None = None
    weight: float = 1.0


@dataclass
class TestCase:
    description: str
    vars: dict
    prompts: list[str]  # prompt ids; empty = all
    providers: list[str]  # provider ids; empty = all
    assert_: list[AssertionSpec]
    threshold: float | None = None  # minimum score to pass this case


@dataclass
class EvalConfig:
    prompts: list[Prompt]
    providers: list[ProviderSpec]
    tests: list[TestCase]
    default_threshold: float = 1.0
    max_concurrency: int = 8


def _as_prompt(raw: object, index: int) -> Prompt:
    if isinstance(raw, str):
        return Prompt(id=f"prompt-{index}", label=raw[:60], raw=raw)
    if isinstance(raw, dict) and "raw" in raw:
        return Prompt(
            id=str(raw.get("id", f"prompt-{index}")),
            label=str(raw.get("label", str(raw["raw"])[:60])),
            raw=str(raw["raw"]),
        )
    raise ConfigError(f"prompts[{index}] must be a string or {{\"raw\": ...}}")


def _as_provider(raw: object, index: int) -> ProviderSpec:
    if isinstance(raw, str):
        return ProviderSpec(id=raw, label=raw)
    if isinstance(raw, dict) and "id" in raw:
        return ProviderSpec(id=str(raw["id"]), label=str(raw.get("label", raw["id"])))
    raise ConfigError(f"providers[{index}] must be a string or {{\"id\": ...}}")


def _as_assertion(raw: object, where: str) -> AssertionSpec:
    if not isinstance(raw, dict) or "type" not in raw:
        raise ConfigError(f"{where}: each assert must be an object with a \"type\"")
    return AssertionSpec(
        type=str(raw["type"]),
        value=raw.get("value"),
        values=list(raw.get("values", [])),
        threshold=raw.get("threshold"),
        min=raw.get("min"),
        max=raw.get("max"),
        weight=float(raw.get("weight", 1.0)),
    )


def _selector(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ConfigError("prompt/provider selectors must be a string or list of strings")


def load_config(path: str) -> EvalConfig:
    """Load and validate an eval config from a JSON file."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc})")

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be an object")

    prompts_raw = data.get("prompts")
    if not prompts_raw:
        raise ConfigError(f"{path}: \"prompts\" must be a non-empty list")
    prompts = [_as_prompt(p, i) for i, p in enumerate(prompts_raw)]

    providers_raw = data.get("providers", ["stub:base"])
    if not providers_raw:
        raise ConfigError(f"{path}: \"providers\" must be a non-empty list")
    providers = [_as_provider(p, i) for i, p in enumerate(providers_raw)]

    default_test = data.get("defaultTest", {}) or {}
    if not isinstance(default_test, dict):
        raise ConfigError(f"{path}: \"defaultTest\" must be an object")
    default_asserts = [
        _as_assertion(a, "defaultTest.assert") for a in default_test.get("assert", [])
    ]

    tests_raw = data.get("tests", [])
    tests: list[TestCase] = []
    for i, t in enumerate(tests_raw):
        where = f"tests[{i}]"
        if not isinstance(t, dict):
            raise ConfigError(f"{where} must be an object")
        asserts = [_as_assertion(a, f"{where}.assert") for a in t.get("assert", [])]
        tests.append(
            TestCase(
                description=str(
                    t.get("description", f"case {i + 1}")
                ),
                vars=dict(t.get("vars", {})),
                prompts=_selector(t.get("prompt")),
                providers=_selector(t.get("provider")),
                assert_=asserts or default_asserts,
                threshold=t.get("threshold", default_test.get("threshold")),
            )
        )
    if not tests:
        raise ConfigError(f"{path}: \"tests\" must be a non-empty list")

    prompt_ids = {p.id for p in prompts}
    provider_ids = {p.id for p in providers}
    for t in tests:
        for pid in t.prompts:
            if pid not in prompt_ids:
                raise ConfigError(f"test {t.description!r}: unknown prompt {pid!r}")
        for pid in t.providers:
            if pid not in provider_ids:
                raise ConfigError(f"test {t.description!r}: unknown provider {pid!r}")

    return EvalConfig(
        prompts=prompts,
        providers=providers,
        tests=tests,
        default_threshold=float(data.get("defaultThreshold", 1.0)),
        max_concurrency=int(data.get("maxConcurrency", 8)),
    )
