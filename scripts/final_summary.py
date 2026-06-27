#!/usr/bin/env python3
"""Final three-way summary for the STaR experiment: overlay accuracy-vs-round for
plain RFT / rationalization / retrieval (shared base + held-out), plus pairwise
paired McNemar at the final round. Writes the overlay PNG to docs/assets/."""
import csv, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
SOURCES = [
    ("plain RFT (LoRA)", "full_Prealgebra/metrics.csv", "full_Prealgebra/round3/eval.jsonl"),
    ("+rationalization", "rat_Prealgebra/metrics.csv", "rat_Prealgebra/round3/eval.jsonl"),
    ("retrieval (BM25)", "full_Prealgebra/arms/retrieval/metrics.csv",
     "full_Prealgebra/arms/retrieval/round3_graded.jsonl"),
]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def curve(rel):
    rows = list(csv.DictReader(open(os.path.join(RUNS, rel))))
    base = [float(r["heldout_acc"]) for r in rows if int(r["round"]) < 0]
    rds = [r for r in rows if int(r["round"]) >= 0]
    return (base[0] if base else None), [int(r["round"]) for r in rds], \
           [float(r["heldout_acc"]) * 100 for r in rds]


def evald(rel):
    return {json.loads(l)["id"]: json.loads(l)["correct"] for l in open(os.path.join(RUNS, rel))}


base = None
plt.figure(figsize=(7, 4.5))
for lab, mcsv, _ in SOURCES:
    b, R, acc = curve(mcsv); base = b or base
    plt.plot(R, acc, "o-", label=lab)
plt.axhline(base * 100, ls="--", color="gray", label=f"base ({base*100:.1f}%)")
plt.xlabel("round"); plt.ylabel("held-out accuracy (%)"); plt.xticks(R)
plt.title("Same pool, three consumers — Prealgebra / Qwen2.5-1.5B (N=150)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
out = os.path.join(ROOT, "docs/assets/star_prealgebra/7_all_arms.png")
plt.savefig(out, dpi=130)
print(f"[ok] overlay -> {out}\n")

evals = {lab: evald(ev) for lab, _, ev in SOURCES}
labs = list(evals)
print("=== final-round paired McNemar (same 150 held-out) ===")
for x in range(len(labs)):
    for y in range(x + 1, len(labs)):
        A, B = evals[labs[x]], evals[labs[y]]
        ids = [i for i in A if i in B]
        b = sum(1 for i in ids if A[i] and not B[i])
        c = sum(1 for i in ids if not A[i] and B[i])
        print(f"  {labs[x]} ({sum(A[i] for i in ids)}/{len(ids)}) vs "
              f"{labs[y]} ({sum(B[i] for i in ids)}/{len(ids)}): "
              f"{b} vs {c}  McNemar p={mcnemar(b, c):.3f}")
