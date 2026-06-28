#!/usr/bin/env python3
"""BBH RSI v2 — per-cluster GROWING, GATED lesson LISTS (one self-contained 7B, single HF load).

Each task-representation cluster gets its OWN lesson LIST, built by forward selection on a held-out dev
slice of that cluster's failures (append a candidate only if it improves dev solve-rate; the gate doubles
as dedup). Routing injects ONLY the matched cluster's list. Arms vs base: global / routed (/ oracle).

NEW: per-arm checkpointing (writes <out>_progress.jsonl as each arm finishes) + viz export
(<out>_viz.json: 3D PCA of failure & test activations, clusters+lessons, per-test base/routed outputs)
for the exploration website.

Reuses helpers from bbh_rsi.py. Run with .venv (torch/transformers). Reads data/bbh/bbh.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bbh_rsi import Model, normalize, extract_answer, kmeans, mcnemar, solve_msgs  # noqa: E402


def block(lst):
    return "\n".join(f"- {l}" for l in lst) if lst else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bbh", "bbh.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--emb-bs", type=int, default=16)
    ap.add_argument("--n-cand", type=int, default=4)
    ap.add_argument("--max-lessons", type=int, default=4)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--dev-cap", type=int, default=6)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bbh", "bbh_rsi_v2.json"))
    args = ap.parse_args()

    d = json.load(open(args.data))
    rows = {r["key"]: r for r in d["rows"]}
    train = [rows[k] for k in d["train_keys"]]
    test = [rows[k] for k in d["test_keys"]]
    print(f"[v2] model={os.path.basename(args.model)} train={len(train)} test={len(test)} K={args.k}", flush=True)
    M = Model(args.model, args.layer_frac, args.gen_bs, args.emb_bs)

    def run(rowset, lbk):  # bool dict — cheap, for dev gating
        gens = M.generate([solve_msgs(r["input"], lbk.get(r["key"])) for r in rowset], args.max_new)
        return {r["key"]: normalize(extract_answer(g)) == normalize(r["target"]) for r, g in zip(rowset, gens)}

    def run_full(rowset, lbk):  # {correct,text} — for final arms + viz
        gens = M.generate([solve_msgs(r["input"], lbk.get(r["key"])) for r in rowset], args.max_new)
        return {r["key"]: {"correct": normalize(extract_answer(g)) == normalize(r["target"]), "text": g}
                for r, g in zip(rowset, gens)}

    def acc_bool(res, rowset):
        return sum(res[r["key"]] for r in rowset) / len(rowset) if rowset else 0.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    progress = args.out.replace(".json", "_progress.jsonl")
    open(progress, "w").close()

    def checkpoint(name, full):
        a = sum(v["correct"] for v in full.values()) / len(full)
        b = sum(1 for r in test if not base_te[r["key"]]["correct"] and full[r["key"]]["correct"])
        c = sum(1 for r in test if base_te[r["key"]]["correct"] and not full[r["key"]]["correct"])
        rec = {"arm": name, "acc": a, "b": b, "c": c, "p": mcnemar(b, c)}
        with open(progress, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [checkpoint] {name}: {a:.1%}  +{b}/-{c} p={mcnemar(b,c):.4f}", flush=True)
        return rec

    # base
    print("[v2-1] base train+test", flush=True)
    base_tr = run(train, {})
    base_te = run_full(test, {})
    checkpoint("base", base_te)

    # cluster train failures (task representation)
    fails = [r for r in train if not base_tr[r["key"]]]
    print(f"[v2-2] embed+cluster {len(fails)} failures (K={args.k})", flush=True)
    Xf = M.embed([solve_msgs(r["input"], None) for r in fails])
    mu = Xf.mean(0)
    proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)
    lab, cent = kmeans(proc(Xf), min(args.k, len(fails)))
    K = cent.shape[0]

    def propose(members, shift):
        m = members[shift % len(members):] + members[:shift % len(members)]
        ex = "\n\n".join(f"PROBLEM: {e['input'][:350]}\nCORRECT: {e['target']}" for e in m[:6])
        msg = [{"role": "user", "content":
                "Here are problems a model got WRONG (shown with the correct answer):\n\n" + ex +
                "\n\nWrite ONE short, general, reusable strategy (1-2 imperative sentences) to solve problems "
                "like these correctly. General method, NOT a specific answer. Output only the strategy."}]
        return M.generate([msg], 128)[0].strip().replace("\n", " ")

    def build_list(members):
        if len(members) < 2:
            return []
        nd = max(1, min(args.dev_cap, int(len(members) * args.dev_frac)))
        dev, mine = members[:nd], members[nd:] or members
        cands = []
        for s in range(args.n_cand):
            c = propose(mine, s * 3 + 1)
            if c and c not in cands:
                cands.append(c)
        lst, cur = [], acc_bool(run(dev, {r["key"]: None for r in dev}), dev)
        while cands and len(lst) < args.max_lessons:
            best, best_rate = None, cur
            for c in cands:
                rate = acc_bool(run(dev, {r["key"]: block(lst + [c]) for r in dev}), dev)
                if rate > best_rate:
                    best, best_rate = c, rate
            if best is None:
                break
            lst.append(best); cands.remove(best); cur = best_rate
        return lst

    def name_cluster(lessons, members):
        ex = "; ".join(m["input"][:90].replace("\n", " ") for m in members[:3])
        body = ("\n".join(f"- {l}" for l in lessons) if lessons else "(no lesson written)")
        msg = [{"role": "user", "content":
                "These problems share a failure mode. Strategies written for it:\n" + body +
                f"\n\nExample problems: {ex}\n\nGive a SHORT category name (3-6 words) for this kind of "
                "problem. Output only the name, no quotes or punctuation."}]
        return M.generate([msg], 24)[0].strip().strip('"').replace("\n", " ")[:60]

    print("[v2-3] build gated per-cluster lesson lists + names", flush=True)
    cluster_lists, cluster_names = [], []
    for k in range(K):
        members = [fails[i] for i in range(len(fails)) if lab[i] == k]
        lst = build_list(members)
        cluster_lists.append(lst)
        nm = name_cluster(lst, members)
        cluster_names.append(nm)
        print(f"  cluster {k} (n={len(members)}) [{nm}]: {len(lst)} lesson(s)" +
              (f" | e.g. {lst[0][:80]}" if lst else ""), flush=True)

    task_lists = {}
    if args.oracle:
        print("[v2-3b] build gated per-task (oracle) lists", flush=True)
        for t in sorted({r["task"] for r in fails}):
            task_lists[t] = build_list([r for r in fails if r["task"] == t])

    # route test
    print("[v2-4] embed+route test", flush=True)
    Et = M.embed([solve_msgs(r["input"], None) for r in test])   # raw, kept for PCA
    Xt = proc(Et)
    nearest = (Xt @ cent.T).argmax(1)
    routed = {r["key"]: block(cluster_lists[nearest[i]]) for i, r in enumerate(test)}
    global_block = block([l for lst in cluster_lists for l in lst][:20]) or None
    glob = {r["key"]: global_block for r in test}

    print("[v2-5] arms", flush=True)
    arms_full = {"base": base_te}
    arms_full["global"] = run_full(test, glob); checkpoint("global", arms_full["global"])
    arms_full["routed"] = run_full(test, routed); checkpoint("routed", arms_full["routed"])
    if args.oracle:
        oc = {r["key"]: block(task_lists.get(r["task"], [])) for r in test}
        arms_full["oracle"] = run_full(test, oc); checkpoint("oracle", arms_full["oracle"])

    # summary
    summary = {"model": os.path.basename(args.model), "K": K, "n_test": len(test), "n_train_fail": len(fails),
               "lessons_per_cluster": [len(x) for x in cluster_lists], "cluster_names": cluster_names,
               "cluster_lists": cluster_lists, "arms": {}}
    print("\n===== BBH RSI v2 (per-cluster gated lists, held-out test) =====")
    for name, full in arms_full.items():
        a = sum(v["correct"] for v in full.values()) / len(test)
        b = sum(1 for r in test if not base_te[r["key"]]["correct"] and full[r["key"]]["correct"])
        c = sum(1 for r in test if base_te[r["key"]]["correct"] and not full[r["key"]]["correct"])
        summary["arms"][name] = {"acc": a, "b": b, "c": c, "p": mcnemar(b, c)}
        print(f"  {name:8} {a:6.1%}   " + ("—" if name == "base" else f"+{b}/-{c}  p={mcnemar(b,c):.4f}"))
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")

    # viz export (3D PCA of activations + per-test detail) — wrapped: never lose the main results
    try:
        m3 = Xf.mean(0)
        _, _, Vt = np.linalg.svd(Xf - m3, full_matrices=False)
        comps = Vt[:3]
        f3 = (Xf - m3) @ comps.T
        t3 = (Et - m3) @ comps.T
        cents3 = [(f3[lab == k].mean(0).tolist() if (lab == k).any() else [0, 0, 0]) for k in range(K)]
        viz = {
            "model": summary["model"], "K": K, "arms": summary["arms"],
            "clusters": [{"id": k, "name": cluster_names[k], "size": int((lab == k).sum()),
                          "lessons": cluster_lists[k], "centroid": cents3[k]} for k in range(K)],
            "train_points": [{"xyz": f3[i].tolist(), "cluster": int(lab[i]), "task": fails[i]["task"],
                              "prompt": fails[i]["input"][:200]} for i in range(len(fails))],
            "test_items": [{"key": r["key"], "task": r["task"], "prompt": r["input"], "gold": r["target"],
                            "xyz": t3[i].tolist(), "routed_cluster": int(nearest[i]),
                            "routed_cluster_name": cluster_names[nearest[i]],
                            "routed_lessons": cluster_lists[nearest[i]],
                            "base_output": base_te[r["key"]]["text"], "base_correct": base_te[r["key"]]["correct"],
                            "routed_output": arms_full["routed"][r["key"]]["text"],
                            "routed_correct": arms_full["routed"][r["key"]]["correct"]} for i, r in enumerate(test)],
        }
        vpath = args.out.replace(".json", "_viz.json")
        json.dump(viz, open(vpath, "w"))
        print(f"-> viz {vpath}")
    except Exception as e:
        print(f"viz export failed (results still saved): {e}", flush=True)


if __name__ == "__main__":
    main()
