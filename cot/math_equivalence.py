"""Backward compatibility shim for math equivalence utilities."""

from math_utils.parse import extract_ans, get_answer, last_boxed_only_string, remove_boxed, strip_string
from math_utils.compare import is_equiv

__all__ = [
    "extract_ans",
    "get_answer",
    "last_boxed_only_string",
    "remove_boxed",
    "strip_string",
    "is_equiv",
]

