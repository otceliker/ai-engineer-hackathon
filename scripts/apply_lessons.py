#!/usr/bin/env python3
"""Apply a fixed lessons playbook straight to a model on a split and compare to its baseline.

A one-shot transfer / ablation test (no loop, no gate): does a playbook learned elsewhere
(e.g. distilled by a strong proposer off the 1.5B's failures) help a *different* agent
(e.g. the 7B)? Loads the model ONCE, evaluates base prompt vs base+lessons on the split,
reports overall + per-category + paired McNemar.

Usage:
    python scripts/apply_lessons.py --model models/Qwen__Qwen2.5-7B-Instruct \
        --lessons results/headroom_lessons_deepseek-v4-pro.json --split stream_test
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bfcl_eval import BfclAgent, render_system_prompt  # noqa: E402

DEFAULT_SPLITS = os.path.join(ROOT, "data", "bfcl", "splits.json")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def load_lesson_texts(path):
    d = json.load(open(path))
    if isinstance(d, dict) and "lessons" in d:
        return [l["text"] for l in d["lessons"]]
    if isinstance(d, list):
        return [x if isinstance(x, str) else x.get("text", "") for x in d]
    raise ValueError(f"Unrecognized lessons file shape: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lessons", required=True, help="LessonBook json or list-of-strings json")
    ap.add_argument("--split", default="stream_test")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args()

    records = json.load(open(args.splits_file))["splits"][args.split]
    if args.limit:
        records = records[:args.limit]
    texts = load_lesson_texts(args.lessons)
    model_name = os.path.basename(args.model.rstrip("/"))
    print(f"Model: {model_name} | split: {args.split} n={len(records)} | lessons: {len(texts)}", flush=True)

    agent = BfclAgent(args.model, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                      temperature=0.0, max_tokens=args.max_tokens)
    base = agent.evaluate(records, render_system_prompt(None))
    patched = agent.evaluate(records, render_system_prompt(texts))

    # paired McNemar
    b = sum(1 for i in base["per_id"] if not base["per_id"][i] and patched["per_id"][i])
    c = sum(1 for i in base["per_id"] if base["per_id"][i] and not patched["per_id"][i])
    p = mcnemar(b, c)

    print(f"\n=== {model_name} | lessons from {os.path.basename(args.lessons)} ===")
    print(f"Baseline : {base['accuracy']:.1%} ({base['n_correct']}/{base['n']})")
    print(f"+Lessons : {patched['accuracy']:.1%} ({patched['n_correct']}/{patched['n']})")
    print(f"Delta    : {patched['accuracy']-base['accuracy']:+.1%}  (fixes b={b}, breaks c={c}, McNemar p={p:.4f})")
    print("\nPer-category (base -> +lessons):")
    for cat in sorted(patched["by_category"]):
        b0, n0 = base["by_category"][cat]; b1, n1 = patched["by_category"][cat]
        d = b1 / n1 - b0 / n0
        flag = "  <--" if abs(d) > 0.001 else ""
        print(f"  {cat:24s}: {b0/n0:5.1%} -> {b1/n1:5.1%} ({d:+.1%}){flag}")


if __name__ == "__main__":
    main()
