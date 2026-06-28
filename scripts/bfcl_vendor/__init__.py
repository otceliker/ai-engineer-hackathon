"""Vendored BFCL V4 AST checker (from bfcl-eval==2026.3.23).

Only the Python single-turn AST evaluation path is vendored here, with imports
rewritten to be self-contained. The upstream `bfcl_eval` package imports every
model-provider SDK (anthropic, openai, cohere, ...) at import time via
`constants.model_config`, which we deliberately avoid.

Source files copied verbatim except for import rewrites:
  - ast_checker.py        (convert_func_name made identity — no OpenAI dot-quirk)
  - type_mappings.py      (verbatim)
  - java_type_converter.py / js_type_converter.py (import path only)
  - enums.py              (Language enum only)
"""
from .ast_checker import ast_checker
from .enums import Language

__all__ = ["ast_checker", "Language"]
