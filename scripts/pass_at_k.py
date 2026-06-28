#!/usr/bin/env python3
"""pass@1 vs pass@k from a round's saved harvest attempts (K samples/problem).

Reuses the attempts.jsonl already written by star_loop's harvest (temp 0.8). Grades
every sample with Math-Verify, then reports per-sample accuracy (≈pass@1) and the
pass@k coverage curve (fraction of problems solved within the first k samples).

A large pass@k − pass@1 gap = the model HAS the capability; it's a selection problem
(best-of-N / self-consistency / verifier), not a capability problem.

Usage:
  python scripts/pass_at_k.py --run runs/full_Prealgebra --round 0
"""
import argparse, json, os, re, sys
from collections import defaultdict, Counter

try:
    import pyarrow.parquet as pq
    from math_verify import parse, verify
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(ROOT, "data", "nlile__hendrycks-MATH-benchmark", "data",
                             "train-00000-of-00001.parquet")


def grade(text, gold):
    try:
        return bool(verify(parse(gold), parse(text)))
    except Exception:
        return False


def extract_boxed(t):
    i = t.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, out = 1, []
    for ch in t[i:]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out).strip()


def majority_vote(samples, gold):
    """Self-consistency: pick the modal boxed answer, grade a representative of it."""
    ans = {s: extract_boxed(t) for s, t in samples.items()}
    counts = Counter(a for a in ans.values() if a)
    if not counts:
        return False
    modal = counts.most_common(1)[0][0]
    rep = next(t for s, t in samples.items() if ans[s] == modal)
    return grade(rep, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()
    run = args.run if os.path.isabs(args.run) else os.path.join(ROOT, args.run)
    man = json.load(open(os.path.join(run, "manifest.json")))
    category = man["category"]
    gold = {r["unique_id"]: r["answer"] for r in pq.read_table(TRAIN_PARQUET).to_pylist()
            if r["subject"] == category}

    att = defaultdict(dict)   # id -> {sample_idx: text}
    for l in open(os.path.join(run, f"round{args.round}", "attempts.jsonl")):
        d = json.loads(l); att[d["id"]][d["sample_idx"]] = d["text"]

    K = max(max(s) for s in att.values()) + 1
    per_problem = {}          # id -> list[bool] over samples
    for pid, samples in att.items():
        per_problem[pid] = [grade(samples.get(s, ""), gold.get(pid, "")) for s in range(K)]

    n = len(per_problem)
    per_sample_acc = sum(sum(v) for v in per_problem.values()) / (n * K)   # ≈ pass@1
    print(f"{category}: {n} problems, K={K} samples each")
    print(f"  per-sample accuracy (pass@1 est): {per_sample_acc:.1%}")
    print("  pass@k coverage (fraction solved within first k samples):")
    for k in range(1, K + 1):
        cov = sum(any(v[:k]) for v in per_problem.values()) / n
        print(f"    pass@{k}: {cov:.1%}")
    gap = sum(any(v) for v in per_problem.values()) / n - per_sample_acc
    print(f"  pass@{K} − pass@1 gap: {gap*100:+.1f}pp (latent headroom for selection methods)")
    maj = sum(majority_vote(att[pid], gold.get(pid, "")) for pid in per_problem) / n
    print(f"  majority-vote@{K} (self-consistency, TRAINING-FREE): {maj:.1%}  "
          f"({(maj-per_sample_acc)*100:+.1f}pp vs pass@1)")


if __name__ == "__main__":
    main()
