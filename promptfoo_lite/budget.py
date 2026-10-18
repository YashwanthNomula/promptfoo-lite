"""Token-budget gate: validate a dollar budget and compare a run's
estimated cost against it.

A budget is a non-negative dollar amount (int, float, or numeric string).
Validation is fail-closed: anything that is not a finite, non-negative
number raises :class:`BudgetError` instead of silently defaulting to a
permissive value.

Boundary rule (documented here and in the README): a run passes the gate
when its estimated cost is **at or under** the budget — only a cost
strictly greater than the budget breaches it.
"""

from __future__ import annotations

import math


class BudgetError(ValueError):
    """Raised when a budget value is invalid."""


def parse_budget(value: object) -> float:
    """Validate a budget in USD and return it as a float.

    Accepts ints, floats, and numeric strings. Rejects booleans, NaN,
    infinities, negative numbers, and non-numeric values with
    :class:`BudgetError` (fail-closed).
    """
    if isinstance(value, bool):
        raise BudgetError(
            f"invalid budget {value!r}: must be a non-negative number"
        )
    if isinstance(value, (int, float)):
        amount = float(value)
    elif isinstance(value, str):
        try:
            amount = float(value.strip())
        except ValueError:
            raise BudgetError(
                f"invalid budget {value!r}: must be a non-negative number"
            )
    else:
        raise BudgetError(
            f"invalid budget {value!r}: must be a non-negative number"
        )
    if not math.isfinite(amount) or amount < 0:
        raise BudgetError(
            f"invalid budget {value!r}: must be a non-negative number"
        )
    return amount


def check_budget(cost_usd: float, budget_usd: float) -> dict:
    """Compare a run's estimated total cost against a budget.

    Returns a verdict dict with the budget, the cost, an ``exceeded`` flag,
    and a human-readable ``status``. A cost exactly equal to the budget
    passes (``exceeded`` is only set when cost is strictly greater).
    """
    exceeded = cost_usd > budget_usd
    return {
        "budget_usd": budget_usd,
        "cost_usd": cost_usd,
        "exceeded": exceeded,
        "status": "exceeded" if exceeded else "ok",
    }
