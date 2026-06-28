#!/usr/bin/env python3
"""Summarize the IFEval dose-response sweep on the HELD-OUT test split.

For each arm (base/L1/L2/L4) reads its official score dir (eval_results_strict.jsonl +
eval_results_loose.jsonl), maps response prompts back to keys, restricts to the test split,
and reports prompt-level strict & loose accuracy plus paired McNemar vs base.
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


def prompt_to_key():
    m = {}
    for l in open(os.path.join(IFE, "input_data.jsonl")):
        r = json.loads(l)
        m[r["prompt"]] = r["key"]
    return m


def load_arm(name, kind, p2k):
    """Return {key: follow_all_instructions(bool)} for an arm's strict|loose results."""
    path = os.path.join(RES, f"score_{name}", f"eval_results_{kind}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    for l in open(path):
        r = json.loads(l)
        k = p2k.get(r["prompt"])
        if k is not None:
            out[k] = bool(r["follow_all_instructions"])
    return out


def acc(d, keys):
    ks = [k for k in keys if k in d]
    return (sum(d[k] for k in ks) / len(ks), len(ks)) if ks else (float("nan"), 0)


def paired(base, arm, keys):
    ks = [k for k in keys if k in base and k in arm]
    b = sum(1 for k in ks if not base[k] and arm[k])
    c = sum(1 for k in ks if base[k] and not arm[k])
    return b, c, mcnemar(b, c)


def main():
    test_keys = json.load(open(os.path.join(IFE, "test_keys.json")))
    train_keys = json.load(open(os.path.join(IFE, "train_keys.json")))
    p2k = prompt_to_key()
    arms = ["base", "L1", "L2", "L4"]

    for split_name, keys in [("TEST (held-out)", test_keys), ("FULL", test_keys + train_keys)]:
        print(f"\n================ IFEval prompt-level accuracy — {split_name} ================")
        print(f"{'arm':6} {'n_lessons':9} {'strict':>9} {'loose':>9}   {'strict vs base (b/c, p)':>28}")
        base_s = load_arm("base", "strict", p2k)
        base_l = load_arm("base", "loose", p2k)
        nmap = {"base": 0, "L1": 1, "L2": 2, "L4": 4}
        for a in arms:
            s = load_arm(a, "strict", p2k)
            l = load_arm(a, "loose", p2k)
            if s is None:
                print(f"{a:6} {nmap[a]:<9} {'(missing)':>9}")
                continue
            sa, n = acc(s, keys)
            la, _ = acc(l, keys)
            if a == "base" or base_s is None:
                cmp = "—"
            else:
                b, c, p = paired(base_s, s, keys)
                cmp = f"+{b}/-{c}  p={p:.4f}"
            print(f"{a:6} {nmap[a]:<9} {sa:>8.1%} {la:>8.1%}   {cmp:>28}   (n={n})")
    print("\n(Independent items + greedy decoding -> paired McNemar is valid here, unlike memory_kv.)")


if __name__ == "__main__":
    main()
