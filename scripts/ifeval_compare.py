#!/usr/bin/env python3
"""Generic IFEval paired comparison on the held-out test split: `compare BASE_ARM CMP_ARM...`.

Reads each arm's official score dir, restricts to test split, prints prompt-level strict & loose
accuracy and paired McNemar of each CMP_ARM vs BASE_ARM. Works for any arm names.
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IFE = os.path.join(ROOT, "data", "ifeval")
RES = os.path.join(ROOT, "results", "ifeval")


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def p2k():
    return {json.loads(l)["prompt"]: json.loads(l)["key"] for l in open(os.path.join(IFE, "input_data.jsonl"))}


def load(name, kind, m):
    path = os.path.join(RES, f"score_{name}", f"eval_results_{kind}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    for l in open(path):
        r = json.loads(l)
        k = m.get(r["prompt"])
        if k is not None:
            out[k] = bool(r["follow_all_instructions"])
    return out


def main():
    args = sys.argv[1:]
    base_arm = args[0] if args else "base"
    cmp_arms = args[1:] if len(args) > 1 else ["refine_self"]
    keys = json.load(open(os.path.join(IFE, "test_keys.json")))
    m = p2k()
    bs, bl = load(base_arm, "strict", m), load(base_arm, "loose", m)

    def acc(d):
        ks = [k for k in keys if k in d]
        return (sum(d[k] for k in ks) / len(ks)) if ks else float("nan")

    print(f"\n===== IFEval TEST (held-out) — base arm: {base_arm} =====")
    if bs:
        print(f"{base_arm:14} strict={acc(bs):6.1%}  loose={acc(bl):6.1%}")
    for a in cmp_arms:
        s, l = load(a, "strict", m), load(a, "loose", m)
        if s is None:
            print(f"{a:14} (missing)")
            continue
        ks = [k for k in keys if k in bs and k in s]
        b = sum(1 for k in ks if not bs[k] and s[k])
        c = sum(1 for k in ks if bs[k] and not s[k])
        print(f"{a:14} strict={acc(s):6.1%}  loose={acc(l):6.1%}   vs {base_arm}: +{b}/-{c}  McNemar p={mcnemar(b,c):.4f}")


if __name__ == "__main__":
    main()
