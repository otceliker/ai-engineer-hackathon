#!/usr/bin/env python3
"""Compare STaR run variants: overlay accuracy-vs-round + paired McNemar between arms
at the final round (same held-out problems). Reusable for plain/rationalization/arms.

Usage:
  python scripts/star_compare.py --runs full_Prealgebra rat_Prealgebra \
      --labels "plain RFT" "+rationalization" --out docs/assets/star_prealgebra/6_plain_vs_rat.png
"""
import argparse, csv, json, math, os, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def curve(run):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "runs", run, "metrics.csv"))))
    base = [float(r["heldout_acc"]) for r in rows if int(r["round"]) < 0][0]
    rds = [r for r in rows if int(r["round"]) >= 0]
    return base, [int(r["round"]) for r in rds], [float(r["heldout_acc"]) * 100 for r in rds]


def final_eval(run):
    rds = [int(r["round"]) for r in csv.DictReader(open(os.path.join(ROOT, "runs", run, "metrics.csv")))
           if int(r["round"]) >= 0]
    last = max(rds)
    return {json.loads(l)["id"]: json.loads(l)["correct"]
            for l in open(os.path.join(ROOT, "runs", run, f"round{last}", "eval.jsonl"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plt.figure(figsize=(6.5, 4))
    base = None
    for run, lab in zip(args.runs, args.labels):
        b, R, acc = curve(run)
        base = b
        plt.plot(R, acc, "o-", label=lab)
    plt.axhline(base * 100, ls="--", color="gray", label=f"base ({base*100:.1f}%)")
    plt.xlabel("round"); plt.ylabel("held-out acc (%)"); plt.xticks(R)
    plt.title("STaR variants (Prealgebra, N=150)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    plt.savefig(os.path.join(ROOT, args.out), dpi=130)
    print(f"[ok] overlay -> {args.out}")

    # pairwise final-round paired McNemar
    evals = {lab: final_eval(run) for run, lab in zip(args.runs, args.labels)}
    labs = list(evals)
    print("\n=== final-round head-to-head (paired McNemar) ===")
    for x in range(len(labs)):
        for y in range(x + 1, len(labs)):
            A, B = evals[labs[x]], evals[labs[y]]
            ids = [i for i in A if i in B]
            b = sum(1 for i in ids if A[i] and not B[i])
            c = sum(1 for i in ids if not A[i] and B[i])
            accA = sum(A[i] for i in ids); accB = sum(B[i] for i in ids)
            print(f"  {labs[x]} ({accA}/{len(ids)}) vs {labs[y]} ({accB}/{len(ids)}): "
                  f"{labs[x]}-better={b} {labs[y]}-better={c} McNemar p={mcnemar(b, c):.3f}")


if __name__ == "__main__":
    main()
