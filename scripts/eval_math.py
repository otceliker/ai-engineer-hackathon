#!/usr/bin/env python3
"""Quick MATH benchmark eval for a local safetensors model, via vLLM + math-verify.

Loads the Hendrycks MATH test split (parquet), generates solutions with vLLM,
grades the final \\boxed{} answer against gold with math-verify, and reports
accuracy overall / by level / by subject. Writes per-example results to JSONL.

Examples:
    # smoke test on 10 problems
    python scripts/eval_math.py --model models/Qwen__Qwen2.5-Math-1.5B-Instruct --limit 10

    # full 500-problem eval, reasoning model needs longer generations
    python scripts/eval_math.py \
        --model models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B \
        --max-tokens 8192 --temperature 0.6
"""
import argparse
import glob
import json
import os
import sys
import time

try:
    import pyarrow.parquet as pq
    from vllm import LLM, SamplingParams
    from math_verify import parse, verify
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Install: uv pip install vllm 'math-verify[antlr4_13_2]'")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEST = glob.glob(os.path.join(ROOT, "data", "**", "*test*.parquet"), recursive=True)
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_rows(path, limit, levels, subjects):
    t = pq.read_table(path).to_pylist()
    if levels:
        t = [r for r in t if int(r["level"]) in levels]
    if subjects:
        subs = {s.lower() for s in subjects}
        t = [r for r in t if r["subject"].lower() in subs]
    return t[:limit] if limit else t


def grade(generated, gold):
    """True if generated answer matches gold under math-verify. Robust to parse errors."""
    try:
        g = parse(gold)
        p = parse(generated)
        # verify(gold, pred) — order matters; gold first.
        return bool(verify(g, p))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to safetensors model dir")
    ap.add_argument("--data", default=DEFAULT_TEST[0] if DEFAULT_TEST else None,
                    help="MATH test parquet (auto-detected by default)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of problems (0 = all)")
    ap.add_argument("--levels", type=int, nargs="*", help="filter to these difficulty levels (1-5)")
    ap.add_argument("--subjects", nargs="*", help="filter to these subjects")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="context length; 0 = auto-derive from model config")
    ap.add_argument("--gpu-mem", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile/cudagraphs (default: enforce_eager for reliability)")
    ap.add_argument("--out", default=None, help="results JSONL path")
    args = ap.parse_args()

    if not args.data or not os.path.exists(args.data):
        sys.exit(f"MATH test parquet not found ({args.data}). Pass --data explicitly.")

    rows = load_rows(args.data, args.limit, args.levels, args.subjects)
    if not rows:
        sys.exit("No rows after filtering.")
    print(f"Loaded {len(rows)} problems from {args.data}", flush=True)

    llm = LLM(model=args.model, max_model_len=args.max_model_len or None,
              gpu_memory_utilization=args.gpu_mem, dtype="bfloat16",
              enforce_eager=not args.compile)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens)

    conversations = [[{"role": "user", "content": r["problem"] + "\n\n" + INSTRUCTION}] for r in rows]
    t0 = time.time()
    outputs = llm.chat(conversations, sp)
    elapsed = time.time() - t0

    model_name = os.path.basename(args.model.rstrip("/"))
    out_path = args.out or os.path.join(ROOT, "results", f"{model_name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_correct = 0
    by_level, by_subject = {}, {}
    with open(out_path, "w") as f:
        for r, o in zip(rows, outputs):
            text = o.outputs[0].text
            ok = grade(text, r["answer"])
            n_correct += ok
            lvl, sub = int(r["level"]), r["subject"]
            by_level.setdefault(lvl, [0, 0]); by_subject.setdefault(sub, [0, 0])
            by_level[lvl][0] += ok; by_level[lvl][1] += 1
            by_subject[sub][0] += ok; by_subject[sub][1] += 1
            f.write(json.dumps({
                "unique_id": r["unique_id"], "level": lvl, "subject": sub,
                "gold": r["answer"], "correct": ok,
                "n_gen_tokens": len(o.outputs[0].token_ids), "generated": text,
            }) + "\n")

    acc = n_correct / len(rows)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {n_correct}/{len(rows)} = {acc:.1%}")
    print(f"Wall time: {elapsed:.1f}s ({elapsed/len(rows):.2f}s/problem)")
    print("\nBy level:")
    for lvl in sorted(by_level):
        c, n = by_level[lvl]; print(f"  L{lvl}: {c}/{n} = {c/n:.1%}")
    print("\nBy subject:")
    for sub in sorted(by_subject):
        c, n = by_subject[sub]; print(f"  {sub}: {c}/{n} = {c/n:.1%}")
    print(f"\nPer-example results -> {out_path}")


if __name__ == "__main__":
    main()
