#!/usr/bin/env python3
"""Held-out self-consistency: the clean, comparable version of the pass@k finding.

On the SAME held-out set the LoRA/retrieval runs used, sample K times (temp 0.8) and
take the majority-vote answer; compare to the greedy pass@1 baseline (reused from the
run's base eval) with a paired McNemar. This is the apples-to-apples claim that a
training-free selection method beats the flat self-training.

Usage:
  python scripts/self_consistency.py --base models/Qwen__Qwen2.5-1.5B-Instruct \
      --pool-run full_Prealgebra --k 8
"""
import argparse, json, math, os, re, signal, subprocess, sys
from collections import Counter, defaultdict

try:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from math_verify import parse, verify
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(ROOT, "data", "nlile__hendrycks-MATH-benchmark", "data",
                             "train-00000-of-00001.parquet")
PY = sys.executable
SUBENV = {**os.environ, "PATH": os.path.dirname(PY) + ":" + os.environ.get("PATH", ""),
          "VLLM_USE_FLASHINFER_SAMPLER": "0"}
SYSTEM = "Solve the problem step by step and put your final answer within \\boxed{}."


def kill_gpu_procs():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    for t in out.stdout.split():
        if t.strip().isdigit():
            try:
                os.kill(int(t), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def grade(text, gold):
    try:
        return bool(verify(parse(gold), parse(text)))
    except Exception:
        return False


def extract_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{"); depth, out = 1, []
    for ch in t[i:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def majority(samples, gold):
    ans = [extract_boxed(t) for t in samples]
    counts = Counter(a for a in ans if a)
    if not counts:
        return False
    modal = counts.most_common(1)[0][0]
    return grade(next(t for t, a in zip(samples, ans) if a == modal), gold)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool-run", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    run = os.path.join(ROOT, "runs", args.pool_run)
    man = json.load(open(os.path.join(run, "manifest.json")))
    tok = AutoTokenizer.from_pretrained(args.base)
    allrows = {r["unique_id"]: r for r in pq.read_table(TRAIN_PARQUET).to_pylist()
               if r["subject"] == man["category"]}
    held = [allrows[i] for i in man["held_ids"]]
    gold = {r["unique_id"]: r["answer"] for r in held}

    # greedy pass@1 baseline (reuse base eval)
    greedy = {}
    for l in open(os.path.join(run, "base", "eval_raw.jsonl")):
        d = json.loads(l); greedy[d["id"]] = grade(d["text"], gold[d["id"]])

    # sample K on held-out
    out_dir = os.path.join(run, "selfconsistency"); os.makedirs(out_dir, exist_ok=True)
    pf = os.path.join(out_dir, "prompts.jsonl"); of = os.path.join(out_dir, "samples.jsonl")
    with open(pf, "w") as f:
        for r in held:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["problem"]}]
            f.write(json.dumps({"id": r["unique_id"],
                                "prompt": tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}) + "\n")
    p = subprocess.Popen([PY, os.path.join(ROOT, "scripts", "star_gen.py"), "--base", args.base,
                          "--prompts", pf, "--out", of, "--n", str(args.k), "--temp", str(args.temp),
                          "--max-tokens", str(args.max_tokens), "--max-model-len", "2048"],
                         env=SUBENV, cwd=ROOT, start_new_session=True)
    ret = p.wait(); kill_gpu_procs()
    if ret != 0:
        sys.exit("generation failed")

    samples = defaultdict(list)
    for l in open(of):
        d = json.loads(l); samples[d["id"]].append(d["text"])
    ids = [r["unique_id"] for r in held]
    maj = {i: majority(samples[i], gold[i]) for i in ids}
    per_sample = sum(grade(t, gold[i]) for i in ids for t in samples[i]) / (len(ids) * args.k)
    passk = sum(any(grade(t, gold[i]) for t in samples[i]) for i in ids) / len(ids)
    g_acc = sum(greedy.values()) / len(ids)
    m_acc = sum(maj.values()) / len(ids)
    b = sum(1 for i in ids if greedy[i] and not maj[i])
    c = sum(1 for i in ids if not greedy[i] and maj[i])
    print(f"\n=== Held-out self-consistency ({man['category']}, N={len(ids)}, K={args.k}) ===")
    print(f"  greedy pass@1 (baseline):        {g_acc:.1%}")
    print(f"  sampled per-sample acc (T={args.temp}): {per_sample:.1%}")
    print(f"  majority-vote@{args.k}:                {m_acc:.1%}  ({(m_acc-g_acc)*100:+.1f}pp vs greedy)")
    print(f"  pass@{args.k} ceiling:                 {passk:.1%}")
    print(f"  paired McNemar maj vs greedy: greedy-better={b} maj-better={c} p={mcnemar(b, c):.4f}")


if __name__ == "__main__":
    main()
