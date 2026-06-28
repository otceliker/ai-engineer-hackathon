#!/usr/bin/env python3
"""Diagnostic: WHY don't lessons help on BBH? Inspect base-failed test cases WITH routed lessons.

Reuses the cached base (has full base output text) + cached activations. Reclusters (K=10), writes
self-lessons per cluster (v2_repro style), routes each sampled base-failed test task to its cluster,
generates WITH the routed lessons, and prints: task, gold, injected lessons, base answer/output,
lessoned answer/output — so we can classify the failure (useless / not-followed / orthogonal / overrode
correct reasoning / changed-but-still-wrong).
"""
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bbh_rsi import Model, normalize, extract_answer, kmeans, solve_msgs  # noqa: E402

CACHE = os.path.join(ROOT, "results", "bbh", "cache")
TAG = "Qwen__Qwen2.5-7B-Instruct"
N_SAMPLE = 12


def main():
    base = json.load(open(os.path.join(CACHE, f"base_{TAG}.json")))
    z = np.load(os.path.join(CACHE, f"emb_{TAG}.npz"), allow_pickle=True)
    Xf, Et = z["Xf"], z["Et"]
    fail_keys, test_keys = list(z["fail_keys"]), list(z["test_keys"])
    rows = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "data", "bbh", "bbh.json")))["rows"]}
    fails = [rows[k] for k in fail_keys]

    M = Model(os.path.join(ROOT, "models", TAG), 0.6, 8, 16)
    mu = Xf.mean(0)
    proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)
    lab, cent = kmeans(proc(Xf), 10)

    def propose(members, shift):
        m = members[shift % len(members):] + members[:shift % len(members)]
        ex = "\n\n".join(f"PROBLEM: {e['input'][:350]}\nCORRECT: {e['target']}" for e in m[:6])
        out = M.generate([[{"role": "user", "content":
              "Problems a model got WRONG (with correct answers):\n\n" + ex +
              "\n\nWrite ONE short general reusable strategy (1-2 sentences). Output only the strategy."}]], 128)[0]
        return out.strip().strip('"').replace("\n", " ")[:240]

    print("building cluster lessons...", flush=True)
    cluster_lessons = []
    for k in range(10):
        mem = [fails[i] for i in range(len(fails)) if lab[i] == k]
        ls = []
        for s in range(4):
            c = propose(mem, s * 3 + 1)
            if c and c not in ls:
                ls.append(c)
        cluster_lessons.append(ls)

    test = [rows[k] for k in test_keys]
    Xt = proc(Et)
    nearest = (Xt @ cent.T).argmax(1)
    failed = [(i, r) for i, r in enumerate(test) if not base[r["key"]]["correct"]]
    sample = random.Random(0).sample(failed, min(N_SAMPLE, len(failed)))

    block = lambda lst: "\n".join(f"- {l}" for l in lst)
    gens = M.generate([solve_msgs(r["input"], block(cluster_lessons[nearest[i]])) for i, r in sample], 512)

    recs = []
    for (i, r), g in zip(sample, gens):
        bt = base[r["key"]]["text"]
        rec = {"task": r["task"], "gold": r["target"], "cluster": int(nearest[i]),
               "lessons": cluster_lessons[nearest[i]],
               "base_ans": normalize(extract_answer(bt)), "base_correct": base[r["key"]]["correct"],
               "lessoned_ans": normalize(extract_answer(g)),
               "lessoned_correct": normalize(extract_answer(g)) == normalize(r["target"]),
               "input": r["input"][:400], "base_out_tail": bt[-450:], "lessoned_out_tail": g[-450:]}
        recs.append(rec)
        print("\n" + "=" * 90)
        print(f"[{rec['task']}] cluster {rec['cluster']} | gold={rec['gold']}")
        print(f"INPUT: {rec['input'][:260]}")
        print(f"LESSONS INJECTED:\n  " + "\n  ".join(rec['lessons']))
        print(f"BASE ans={rec['base_ans']!r} ({'✓' if rec['base_correct'] else '✗'}) | "
              f"LESSONED ans={rec['lessoned_ans']!r} ({'✓' if rec['lessoned_correct'] else '✗'}) | "
              f"changed={rec['base_ans']!=rec['lessoned_ans']}")
        print(f"LESSONED OUT (tail): ...{rec['lessoned_out_tail']}")

    json.dump(recs, open(os.path.join(ROOT, "results", "bbh", "diag.json"), "w"), indent=1)
    chg = sum(1 for r in recs if r["base_ans"] != r["lessoned_ans"])
    fixed = sum(1 for r in recs if not r["base_correct"] and r["lessoned_correct"])
    print(f"\n=== SUMMARY: {len(recs)} base-failed cases | answer changed by lessons: {chg} | "
          f"fixed: {fixed} | still wrong: {len(recs)-fixed} ===")


if __name__ == "__main__":
    main()
