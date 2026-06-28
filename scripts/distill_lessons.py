#!/usr/bin/env python3
"""Offline batch lesson distillation (no streaming, no gate).

Run the agent over the ENTIRE training split, hand a strong proposer ALL the failures
at once, have it write N distinct non-redundant lessons, apply them to the agent on the
test split, and measure the lift. This is the full-information CEILING of the method —
the contrast to the online streaming+gated loop (which only sees one mini-batch per round).

Usage:
    PROPOSER_API_KEY=$(cat ~/.do_token) python scripts/distill_lessons.py \
      --model models/Qwen__Qwen2.5-7B-Instruct --proposer-model deepseek-v4-pro \
      --splits-file data/bfcl/splits_full.json --n-lessons 10 --max-model-len 8192
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bfcl_eval import BfclAgent, render_system_prompt, parse_tool_calls  # noqa: E402
from scripts.proposer import format_failure, _first_json  # noqa: E402

DEFAULT_SPLITS = os.path.join(ROOT, "data", "bfcl", "splits.json")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


DISTILL_SYSTEM = (
    "You improve a function-calling agent by writing a SHORT playbook of general lessons that "
    "will be appended to its system prompt. You are given the agent's FAILED tasks across an "
    "entire training set (the request, available tools, what the agent emitted, and the correct "
    "behavior).\n\n"
    "Write EXACTLY {n} lessons that, taken together, would most improve the agent.\n"
    "Requirements:\n"
    "- Each lesson is general and reusable (a behavior pattern, never a single task's answer).\n"
    "- Imperative and short (one sentence).\n"
    "- The {n} lessons must be DISTINCT and NON-OVERLAPPING — do not restate the same idea in "
    "different words. Cover the widest range of the observed failure modes.\n"
    "- Order them by how many failures they would fix (most impactful first).\n\n"
    'Output ONLY JSON: {{"lessons": ["...", "..."]}} with exactly {n} strings.'
)


def distill(proposer_model, failures, n_lessons, max_failures, rng):
    import requests
    api_key = os.environ.get("PROPOSER_API_KEY") or os.environ.get("DO_TOKEN")
    base_url = os.environ.get("PROPOSER_BASE_URL", "https://inference.do-ai.run/v1").rstrip("/")
    if not api_key:
        sys.exit("Need PROPOSER_API_KEY (or DO_TOKEN) for the proposer.")

    shown = failures
    truncated = 0
    if max_failures and len(failures) > max_failures:
        # stratified-ish: shuffle then take, so all categories are represented
        shown = failures[:]
        rng.shuffle(shown)
        truncated = len(failures) - max_failures
        shown = shown[:max_failures]
    blocks = "\n".join(format_failure(*f) for f in shown)
    note = f"\n({truncated} additional failures omitted for length)" if truncated else ""
    user = (f"The agent failed the following {len(failures)} training tasks "
            f"(showing {len(shown)}):\n\n{blocks}{note}\n\nWrite exactly {n_lessons} lessons as JSON.")

    payload = {
        "model": proposer_model, "max_tokens": 4096, "temperature": 0.3,
        "messages": [
            {"role": "system", "content": DISTILL_SYSTEM.format(n=n_lessons)},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    delay = 2
    for attempt in range(1, 6):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=180)
        except requests.RequestException as e:
            last = f"net error: {e}"
        else:
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"]
                obj = _first_json(text)
                lessons = obj.get("lessons", obj) if isinstance(obj, dict) else obj
                return [l for l in lessons if isinstance(l, str)][:n_lessons]
            if r.status_code not in (429, 500, 502, 503, 504):
                sys.exit(f"Proposer API {r.status_code}: {r.text[:300]}")
            last = f"HTTP {r.status_code}"
        if attempt < 5:
            print(f"  [proposer] {last}; backoff {delay}s", flush=True); time.sleep(delay); delay = min(delay * 2, 60)
    sys.exit(f"Proposer failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="agent model path")
    ap.add_argument("--proposer-model", default="deepseek-v4-pro")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS)
    ap.add_argument("--train-split", default="stream_train")
    ap.add_argument("--test-split", default="stream_test")
    ap.add_argument("--n-lessons", type=int, default=10)
    ap.add_argument("--max-failures", type=int, default=250, help="cap failures shown to proposer (0=all)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    splits = json.load(open(args.splits_file))["splits"]
    train, test = splits[args.train_split], splits[args.test_split]
    model_name = os.path.basename(args.model.rstrip("/"))
    print(f"Agent: {model_name} | proposer: {args.proposer_model} | train={len(train)} test={len(test)} | N={args.n_lessons}", flush=True)

    agent = BfclAgent(args.model, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                      temperature=0.0, max_tokens=args.max_tokens)

    # 1. agent failures across the ENTIRE training set
    print("Evaluating agent on full training set ...", flush=True)
    train_res = agent.evaluate(train, render_system_prompt(None))
    failures = [(r, train_res["raw"][r["id"]], parse_tool_calls(train_res["raw"][r["id"]]))
                for r in train if not train_res["per_id"][r["id"]]]
    print(f"  train acc {train_res['accuracy']:.1%}; {len(failures)} failures -> proposer", flush=True)

    # 2. distill N lessons from all failures at once
    lessons = distill(args.proposer_model, failures, args.n_lessons, args.max_failures, rng)
    print(f"\nDistilled {len(lessons)} lessons:")
    for i, l in enumerate(lessons, 1):
        print(f"  {i}. {l}")

    # 3. apply to test, paired comparison
    print("\nEvaluating base vs +lessons on test ...", flush=True)
    base = agent.evaluate(test, render_system_prompt(None))
    patched = agent.evaluate(test, render_system_prompt(lessons))
    b = sum(1 for i in base["per_id"] if not base["per_id"][i] and patched["per_id"][i])
    c = sum(1 for i in base["per_id"] if base["per_id"][i] and not patched["per_id"][i])
    p = mcnemar(b, c)

    out = args.out or os.path.join(ROOT, "results", f"distill_{model_name}_{args.proposer_model}.json")
    json.dump({"lessons": lessons, "agent": model_name, "proposer": args.proposer_model,
               "train_acc": train_res["accuracy"], "n_failures": len(failures),
               "base_acc": base["accuracy"], "patched_acc": patched["accuracy"],
               "b": b, "c": c, "mcnemar_p": p,
               "baseline_by_category": base["by_category"], "by_category": patched["by_category"]},
              open(out, "w"), indent=2)

    print(f"\n=== {model_name} | offline distill ({args.n_lessons} lessons from {len(failures)} train failures) ===")
    print(f"Baseline : {base['accuracy']:.1%} ({base['n_correct']}/{base['n']})")
    print(f"+Lessons : {patched['accuracy']:.1%} ({patched['n_correct']}/{patched['n']})")
    print(f"Delta    : {patched['accuracy']-base['accuracy']:+.1%}  (fixes b={b}, breaks c={c}, McNemar p={p:.4f})")
    print("\nPer-category (base -> +lessons):")
    for cat in sorted(patched["by_category"]):
        b0, n0 = base["by_category"][cat]; b1, n1 = patched["by_category"][cat]
        d = b1 / n1 - b0 / n0
        print(f"  {cat:24s}: {b0/n0:5.1%} -> {b1/n1:5.1%} ({d:+.1%})" + ("  <--" if abs(d) > 0.001 else ""))
    print(f"\nlessons -> {out}")


if __name__ == "__main__":
    main()
