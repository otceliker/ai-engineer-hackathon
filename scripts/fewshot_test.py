#!/usr/bin/env python3
"""Few-shot-from-recovered-failures test.

Take N tasks the agent originally FAILED, rejection-sample the agent (temp>0, up to K tries)
until it produces an AST-passing trace, then feed those self-generated correct traces as
compact few-shot examples in the system prompt. Measure base vs +few-shot on the test split.

Contrast to lessons: concrete worked traces in the model's own format, not abstract rules.

Usage:
    python scripts/fewshot_test.py --model models/Qwen__Qwen2.5-7B-Instruct \
      --splits-file data/bfcl/splits_full.json --n-shots 10 --max-model-len 8192
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bfcl_eval import BfclAgent, render_system_prompt, parse_tool_calls, score_record, BASE_SYSTEM_PROMPT  # noqa: E402

DEFAULT_SPLITS = os.path.join(ROOT, "data", "bfcl", "splits.json")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def user_query(record):
    msgs = [m["content"] for m in record["question"][0] if m.get("role") == "user"]
    return " ".join(msgs).strip()


def correct_trace(agent, record, k, temp):
    """Rejection-sample up to k tries; return the first AST-passing output text, else None."""
    from vllm import SamplingParams
    prompt = agent.build_prompt(record, render_system_prompt(None))
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=512, n=k)
    out = agent.llm.generate([prompt], sp)
    for cand in out[0].outputs:
        if score_record(record, parse_tool_calls(cand.text)):
            return cand.text.strip()
    return None


def build_fewshot_block(shots):
    lines = ["Here are examples of correct responses to similar requests. Follow the same style:"]
    for i, (rec, trace) in enumerate(shots, 1):
        tools = ", ".join(f.get("name", "?") for f in rec["function"]) or "(none)"
        resp = trace if "<tool_call>" in trace else "(no function call — no available tool fits the request)"
        lines.append(f"\nExample {i}:\nTools available: {tools}\nRequest: {user_query(rec)[:300]}\nCorrect response: {resp[:300]}")
    return BASE_SYSTEM_PROMPT + "\n\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS)
    ap.add_argument("--train-split", default="stream_train")
    ap.add_argument("--test-split", default="stream_test")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--scan", type=int, default=200, help="how many train tasks to scan for recoverable failures")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits = json.load(open(args.splits_file))["splits"]
    train, test = splits[args.train_split][:args.scan], splits[args.test_split]
    model_name = os.path.basename(args.model.rstrip("/"))
    print(f"Agent: {model_name} | scan {len(train)} train | test {len(test)} | n_shots={args.n_shots} k={args.k_samples} temp={args.temp}", flush=True)

    agent = BfclAgent(args.model, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                      temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    # 1. find failures (greedy base)
    base_train = agent.evaluate(train, render_system_prompt(None))
    failed = [r for r in train if not base_train["per_id"][r["id"]]]
    print(f"  {len(failed)} failures in scan; rejection-sampling for {args.n_shots} recovered traces ...", flush=True)

    # 2. rejection-sample correct traces
    shots = []
    tried = recovered = 0
    for rec in failed:
        if len(shots) >= args.n_shots:
            break
        tried += 1
        tr = correct_trace(agent, rec, args.k_samples, args.temp)
        if tr is not None:
            shots.append((rec, tr)); recovered += 1
    print(f"  recovered {len(shots)}/{tried} attempted (k={args.k_samples})", flush=True)
    if not shots:
        sys.exit("No recoverable failures found.")

    fewshot_sys = build_fewshot_block(shots)
    print(f"\nFew-shot block: {len(shots)} examples, {len(fewshot_sys)} chars")
    for i, (rec, tr) in enumerate(shots, 1):
        print(f"  ex{i} [{rec['category']}]: {user_query(rec)[:70]} -> {(tr[:70] if '<tool_call>' in tr else '(no call)')}")

    # 3. base vs +few-shot on test
    print("\nEvaluating base vs +few-shot on test ...", flush=True)
    base = agent.evaluate(test, render_system_prompt(None))
    fs = agent.evaluate(test, fewshot_sys)
    b = sum(1 for i in base["per_id"] if not base["per_id"][i] and fs["per_id"][i])
    c = sum(1 for i in base["per_id"] if base["per_id"][i] and not fs["per_id"][i])
    p = mcnemar(b, c)

    out = args.out or os.path.join(ROOT, "results", f"fewshot_{model_name}.json")
    json.dump({"agent": model_name, "n_shots": len(shots), "k": args.k_samples, "temp": args.temp,
               "base_acc": base["accuracy"], "fewshot_acc": fs["accuracy"], "b": b, "c": c, "mcnemar_p": p,
               "shots": [{"id": r["id"], "category": r["category"], "query": user_query(r), "trace": t} for r, t in shots],
               "baseline_by_category": base["by_category"], "by_category": fs["by_category"]},
              open(out, "w"), indent=2)

    print(f"\n=== {model_name} | few-shot ({len(shots)} recovered-failure exemplars) ===")
    print(f"Baseline  : {base['accuracy']:.1%} ({base['n_correct']}/{base['n']})")
    print(f"+Few-shot : {fs['accuracy']:.1%} ({fs['n_correct']}/{fs['n']})")
    print(f"Delta     : {fs['accuracy']-base['accuracy']:+.1%}  (fixes b={b}, breaks c={c}, McNemar p={p:.4f})")
    print("\nPer-category (base -> +few-shot):")
    for cat in sorted(fs["by_category"]):
        b0, n0 = base["by_category"][cat]; b1, n1 = fs["by_category"][cat]
        d = b1 / n1 - b0 / n0
        print(f"  {cat:24s}: {b0/n0:5.1%} -> {b1/n1:5.1%} ({d:+.1%})" + ("  <--" if abs(d) > 0.001 else ""))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
