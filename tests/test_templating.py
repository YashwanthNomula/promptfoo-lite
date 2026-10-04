"""Tests for templating."""

import pytest

from promptfoo_lite import templating


def test_render_simple():
    assert templating.render("Hello {{name}}!", {"name": "Ada"}) == "Hello Ada!"


def test_render_multiple_and_spacing():
    out = templating.render("{{ a }} and {{b}}", {"a": "x", "b": "y"})
    assert out == "x and y"


def test_render_dotted():
    out = templating.render("Hi {{user.name}}", {"user": {"name": "Ada"}})
    assert out == "Hi Ada"


def test_render_numbers_and_bools():
    out = templating.render("n={{n}} b={{b}}", {"n": 42, "b": True})
    assert out == "n=42 b=True"


def test_render_missing_raises():
    with pytest.raises(templating.TemplateError, match="missing variable"):
        templating.render("Hello {{name}}", {})


def test_render_bad_value_raises():
    with pytest.raises(templating.TemplateError):
        templating.render("Hi {{x}}", {"x": ["not", "a", "scalar"]})


def test_variables_in():
    assert templating.variables_in("a {{x}} b {{y.z}}") == ["x", "y.z"]
