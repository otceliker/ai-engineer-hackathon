#!/usr/bin/env python3
"""Patch bfcl-eval's system_prompt_pre_processing_chat_model to inject a lessons-memory block.

Env-gated and append-only:
  - LESSONS_FILE unset/empty  -> behaviour identical to stock bfcl (clean A/B baseline arm).
  - LESSONS_FILE=<path to json {"lessons": [...]}> -> appends a "Lessons learned" block AFTER
    bfcl's required format instructions, so the [func(params)] contract is never disturbed.

Idempotent: re-running is a no-op if the marker is already present. Prints the patched file path.

Usage (on neptune, inside .venv-bfcl):
    .venv-bfcl/bin/python scripts/patch_bfcl_lessons.py
    # to revert:
    .venv-bfcl/bin/python scripts/patch_bfcl_lessons.py --revert
"""
import argparse
import importlib.util
import os
import sys

MARKER = "# --- lessons-memory injection (our patch) ---"

ANCHOR = """    else:
        prompts.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    return prompts"""

INJECTION = """    else:
        prompts.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    # --- lessons-memory injection (our patch) ---
    import os as _os
    _lf = _os.environ.get("LESSONS_FILE")
    if _lf and _os.path.exists(_lf):
        import json as _json
        _d = _json.load(open(_lf))
        _lessons = _d.get("lessons", _d) if isinstance(_d, dict) else _d
        _lessons = [l for l in _lessons if isinstance(l, str)]
        if _lessons:
            _block = "\\n\\nLessons learned from past mistakes:\\n" + "\\n".join(
                f"- {l}" for l in _lessons
            )
            prompts[0]["content"] = prompts[0]["content"] + _block
    # --- end lessons-memory injection ---

    return prompts"""


def target_file():
    import glob
    # Prefer an explicit glob relative to cwd (avoids import-path shadowing from the repo dir).
    hits = glob.glob(".venv-bfcl/lib/python*/site-packages/bfcl_eval/model_handler/utils.py")
    hits += glob.glob(os.path.expanduser(
        "~/ai-engineer-hackathon/.venv-bfcl/lib/python*/site-packages/bfcl_eval/model_handler/utils.py"))
    if hits:
        return os.path.abspath(hits[0])
    spec = importlib.util.find_spec("bfcl_eval.model_handler.utils")
    if spec is None or not spec.origin:
        sys.exit("Could not locate bfcl_eval.model_handler.utils — run inside .venv-bfcl.")
    return spec.origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = target_file()
    src = open(path).read()

    if args.revert:
        if MARKER not in src:
            print(f"No patch present in {path}; nothing to revert.")
            return
        src = src.replace(INJECTION, ANCHOR)
        open(path, "w").write(src)
        print(f"Reverted: {path}")
        return

    if MARKER in src:
        print(f"Already patched (idempotent no-op): {path}")
        return
    if "import os" not in src.split("\n\n")[0] and "import os" not in src[:2000]:
        # ensure os is importable at module scope (it usually is); the injection imports json locally
        pass
    if ANCHOR not in src:
        sys.exit("Anchor block not found — bfcl version changed; inspect the function manually.")
    src = src.replace(ANCHOR, INJECTION)
    open(path, "w").write(src)
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
