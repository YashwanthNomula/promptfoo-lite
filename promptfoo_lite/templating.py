"""``{{var}}`` template rendering for prompts."""

from __future__ import annotations

import re

_VAR_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.]+)\s*\}\}")


class TemplateError(ValueError):
    """Raised when a prompt template cannot be rendered."""


def render(template: str, vars: dict) -> str:
    """Render *template*, substituting ``{{var}}`` placeholders from *vars*.

    Dotted names (``{{user.name}}``) walk nested dicts. Raises TemplateError
    on missing variables or non-string values.
    """

    def _replace(match: re.Match) -> str:
        name = match.group(1)
        value = vars
        for part in name.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise TemplateError(f"missing variable: {name!r}")
        if not isinstance(value, (str, int, float, bool)):
            raise TemplateError(
                f"variable {name!r} must be a string, number or bool, "
                f"got {type(value).__name__}"
            )
        return str(value)

    return _VAR_RE.sub(_replace, template)


def variables_in(template: str) -> list[str]:
    """Return the placeholder names used in *template*, in order of appearance."""
    return _VAR_RE.findall(template)
