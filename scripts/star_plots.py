#!/usr/bin/env python3
"""Graphs for a STaR self-improvement run. Reads metrics.csv (+ per-round
train_loss.json) from a run dir, writes PNGs to <run>/plots/ alongside the source
CSV so they regenerate. Headless (Agg)."""
import argparse, csv, json, os, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Install: uv pip install matplotlib")


def load(run):
    rows = list(csv.DictReader(open(os.path.join(run, "metrics.csv"))))
    base = next((r for r in rows if int(r["round"]) < 0), None)
    rds = [r for r in rows if int(r["round"]) >= 0]
    return base, rds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with metrics.csv")
    args = ap.parse_args()
    base, rds = load(args.run)
    pdir = os.path.join(args.run, "plots"); os.makedirs(pdir, exist_ok=True)
    R = [int(r["round"]) for r in rds]
    base_acc = float(base["heldout_acc"]) * 100 if base else None

    # 1) held-out accuracy vs round
    plt.figure(figsize=(6, 4))
    plt.plot(R, [float(r["heldout_acc"]) * 100 for r in rds], "o-", label="LoRA self-train")
    if base_acc is not None:
        plt.axhline(base_acc, ls="--", color="gray", label=f"base ({base_acc:.1f}%)")
    plt.xlabel("round"); plt.ylabel("held-out accuracy (%)")
    plt.title("Held-out accuracy vs round"); plt.legend(); plt.grid(alpha=.3); plt.xticks(R)
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "1_accuracy.png"), dpi=130); plt.close()

    # 2) frontier expansion vs round (held-out problems solved that base could not)
    plt.figure(figsize=(6, 4))
    plt.bar(R, [int(r["heldout_new_vs_base"]) for r in rds], color="#2a7")
    plt.xlabel("round"); plt.ylabel("# held-out solved that base couldn't")
    plt.title("Frontier expansion vs base (not just resampling)")
    plt.grid(alpha=.3, axis="y"); plt.xticks(R)
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "2_frontier.png"), dpi=130); plt.close()

    # 3) traces harvested + cumulative pool vs round
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(R, [int(r["n_harvested"]) for r in rds], alpha=.6, color="#69c", label="new traces")
    ax1.set_xlabel("round"); ax1.set_ylabel("new correct traces"); ax1.set_xticks(R)
    ax2 = ax1.twinx()
    ax2.plot(R, [int(r["cum_pool"]) for r in rds], "k.-", label="cumulative pool")
    ax2.set_ylabel("cumulative pool size")
    ax1.set_title("Harvest per round + cumulative pool")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "3_harvest.png"), dpi=130); plt.close()

    # 4) per-stage wall-clock per round
    plt.figure(figsize=(6, 4))
    bottom = [0] * len(R)
    for stage, col in [("generate", "t_gen"), ("grade", "t_grade"),
                       ("train", "t_train"), ("eval", "t_eval")]:
        vals = [float(r[col]) for r in rds]
        plt.bar(R, vals, bottom=bottom, label=stage)
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.xlabel("round"); plt.ylabel("wall-clock (s)")
    plt.title("Per-stage time per round"); plt.legend(); plt.xticks(R)
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "4_timing.png"), dpi=130); plt.close()

    # 5) training loss curves overlaid by round (if available)
    plt.figure(figsize=(6, 4))
    any_loss = False
    for t in R:
        f = os.path.join(args.run, f"round{t}", "adapter", "train_loss.json")
        if os.path.exists(f):
            losses = json.load(open(f)).get("losses", [])
            if losses:
                plt.plot(range(len(losses)), losses, label=f"round {t}"); any_loss = True
    if any_loss:
        plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training loss by round")
        plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(pdir, "5_trainloss.png"), dpi=130)
    plt.close()

    print(f"[ok] plots -> {pdir}")


if __name__ == "__main__":
    main()
