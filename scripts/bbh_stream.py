#!/usr/bin/env python3
"""ONLINE streaming RSI on BBH ("ultimate vision"): cold-start clusters, lessons grow as tasks stream,
with a PAIRED no-lessons control so the rising curve isn't a confound.

One self-contained 7B (HF). Reuses bbh_lab's disk cache (base + activations) so the CONTROL stream and
any no-lesson task are free (cached base); only LESSONED tasks (and lesson-writing) hit the model.

Algorithm: stream the test set (shuffled). For each task: route to nearest existing cluster (cosine >
threshold) → its lesson list → solve (with lessons → fresh gen; no lessons → cached base). Control =
cached base always. If the lessoned attempt FAILS, add the task's activation to the cluster pool
(assign to nearest cluster or spawn a new one). Every BATCH tasks, (re)write gated lessons for clusters
that gained members. Track cumulative + windowed moving-average accuracy for both streams.

Output: results/bbh/stream.json (curves, cluster growth, final lessons) for the streaming webview.
Run AFTER bbh_lab has created the cache (results/bbh/cache/{base_,emb_}<tag>.{json,npz}).
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bbh_rsi import Model, normalize, extract_answer, solve_msgs  # noqa: E402

CACHE = os.path.join(ROOT, "results", "bbh", "cache")


def block(lst):
    return "\n".join(f"- {l}" for l in lst) if lst else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bbh", "bbh.json"))
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.82, help="cosine to join a cluster, else spawn")
    ap.add_argument("--max-lessons", type=int, default=3)
    ap.add_argument("--n-cand", type=int, default=3)
    ap.add_argument("--dev-cap", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bbh", "stream.json"))
    args = ap.parse_args()

    tag = os.path.basename(args.model.rstrip("/"))
    base = json.load(open(os.path.join(CACHE, f"base_{tag}.json")))
    z = np.load(os.path.join(CACHE, f"emb_{tag}.npz"), allow_pickle=True)
    Et, test_keys = z["Et"], list(z["test_keys"])
    rows = {r["key"]: r for r in json.load(open(args.data))["rows"]}
    eidx = {k: i for i, k in enumerate(test_keys)}
    norm = lambda v: v / (np.linalg.norm(v) + 1e-8)
    En = {k: norm(Et[eidx[k]]) for k in test_keys}

    rng = np.random.RandomState(args.seed)
    stream = list(test_keys); rng.shuffle(stream)
    M = Model(args.model, args.layer_frac, args.gen_bs, 16)

    def solve_one(key, lessons):
        g = M.generate([solve_msgs(rows[key]["input"], lessons)], args.max_new)[0]
        return normalize(extract_answer(g)) == normalize(rows[key]["target"])

    def propose(members, shift):
        m = members[shift % len(members):] + members[:shift % len(members)]
        ex = "\n\n".join(f"PROBLEM: {rows[k]['input'][:300]}\nCORRECT: {rows[k]['target']}" for k in m[:5])
        out = M.generate([[{"role": "user", "content":
            "Problems a model got WRONG (with correct answers):\n\n" + ex +
            "\n\nWrite ONE short general reusable strategy (1-2 sentences). Output only the strategy."}]], 128)[0]
        return out.strip().strip('"').replace("\n", " ")[:240]

    def rewrite(cl):
        """Gated forward-selection of lessons for a cluster from its member failures."""
        members = cl["members"]
        if len(members) < 2:
            return
        nd = max(1, min(args.dev_cap, int(len(members) * 0.3)))
        dev = members[:nd]
        cands = []
        for s in range(args.n_cand):
            c = propose(members, s * 2 + 1)
            if c and c not in cands:
                cands.append(c)
        lst = []
        cur = sum(base[k]["correct"] for k in dev) / nd  # dev baseline = cached base
        # re-evaluate dev under candidate lessons (fresh gen)
        while cands and len(lst) < args.max_lessons:
            best, br = None, cur
            for c in cands:
                ok = sum(solve_one(k, block(lst + [c])) for k in dev) / nd
                if ok > br:
                    best, br = c, ok
            if best is None:
                break
            lst.append(best); cands.remove(best); cur = br
        cl["lessons"] = lst

    clusters = []   # each: {centroid(np), n, members[keys], lessons[], dirty}
    def nearest(en):
        if not clusters:
            return -1, -1.0
        sims = [float(en @ c["centroid"]) for c in clusters]
        j = int(np.argmax(sims))
        return j, sims[j]

    ctrl_c = les_c = 0
    log = []
    for t, key in enumerate(stream):
        en = En[key]
        j, sim = nearest(en)
        lessons = block(clusters[j]["lessons"]) if (j >= 0 and sim > args.threshold and clusters[j]["lessons"]) else None
        les_ok = solve_one(key, lessons) if lessons else base[key]["correct"]
        ctrl_ok = base[key]["correct"]
        ctrl_c += ctrl_ok; les_c += les_ok
        # learn from lessoned failures
        if not les_ok:
            j2, sim2 = nearest(en)
            if j2 >= 0 and sim2 > args.threshold:
                c = clusters[j2]
                c["centroid"] = norm(c["centroid"] * c["n"] + en); c["n"] += 1
                c["members"].append(key); c["dirty"] = True
            else:
                clusters.append({"centroid": en.copy(), "n": 1, "members": [key], "lessons": [], "dirty": True})
        log.append({"t": t, "key": key, "task": rows[key]["task"], "routed_cluster": (j if lessons else -1),
                    "had_lessons": bool(lessons), "lessoned_correct": bool(les_ok), "control_correct": bool(ctrl_ok),
                    "cum_lessoned": les_c / (t + 1), "cum_control": ctrl_c / (t + 1), "n_clusters": len(clusters)})
        if (t + 1) % args.batch == 0:
            for c in clusters:
                if c.get("dirty"):
                    rewrite(c); c["dirty"] = False
            print(f"  [stream] {t+1}/{len(stream)} | lessoned {les_c/(t+1):.1%} vs control {ctrl_c/(t+1):.1%} "
                  f"| clusters={len(clusters)}", flush=True)

    out = {"model": tag, "n": len(stream), "batch": args.batch, "threshold": args.threshold,
           "final_lessoned": les_c / len(stream), "final_control": ctrl_c / len(stream),
           "steps": log,
           "clusters": [{"id": i, "size": c["n"], "lessons": c["lessons"]} for i, c in enumerate(clusters)]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"\nSTREAM done: lessoned {out['final_lessoned']:.1%} vs control {out['final_control']:.1%} "
          f"| {len(clusters)} clusters -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
