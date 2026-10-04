"""promptfoo-lite: a tiny LLM prompt-evaluation harness with zero dependencies."""

__version__ = "1.0.0"

from promptfoo_lite.config import EvalConfig, load_config
from promptfoo_lite.runner import EvalResult, TestCaseResult, run_eval

__all__ = [
    "__version__",
    "EvalConfig",
    "EvalResult",
    "TestCaseResult",
    "load_config",
    "run_eval",
]
