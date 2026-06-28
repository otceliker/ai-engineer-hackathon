"""Minimal vendored subset of bfcl_eval.constants.enums.

Only the Language enum is needed by the AST checker. Vendored to avoid pulling
the full bfcl_eval package (which imports every provider SDK at import time).
"""
from enum import Enum


class Language(Enum):
    """Language controls the type checking for AST checker."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
