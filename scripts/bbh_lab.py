#!/usr/bin/env python3
"""Fast BBH RSI lab — one HF model load, PERSISTENT cache of base+activations, sweep many configs.

The base run (train+test solve) and the failure/test activations depend ONLY on the model, so we compute
them ONCE and cache to disk (results/bbh/cache/), reused across every config and every future invocation.
Each config then costs only: write lessons -> routed arm -> score vs cached base. Global/oracle arms are
dropped for sweeps (v1 already showed global hurts + gave the oracle ceiling).

Configs (JSON list) knobs: k, max_lessons, n_cand, cot, lesson_max_new, temp, gate, dev_cap, topk,
proposer ("self" = the 7B, or a DO model id like "deepseek-v4-pro"/"glm-5.2").

Usage:
  python scripts/bbh_lab.py --model models/Qwen__Qwen2.5-7B-Instruct --configs scripts/bbh_lab_configs.json \
      --test-limit 0 --out results/bbh/lab.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bbh_rsi import Model, normalize, extract_answer, kmeans, mcnemar, solve_msgs  # noqa: E402

CACHE = os.path.join(ROOT, "results", "bbh", "cache")


def block(lst):
    return "\n".join(f"- {l}" for l in lst) if lst else None


def do_propose(model, examples_text, n):
    """External proposer via DigitalOcean (returns up to n candidate lessons)."""
    import requests
    key = open(os.path.expanduser("~/.do_token")).read().strip()
    sysmsg = (f"Write EXACTLY {n} short, general, reusable strategies (1-2 imperative sentences each) to "
              f"solve problems like the failures shown. General methods, not specific answers. "
              f'Output ONLY JSON: {{"lessons":["...", ...]}}.')
    payload = {"model": model, "max_tokens": 1024, "temperature": 0.4,
               "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": examples_text}]}
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    d = 2
    for _ in range(5):
        try:
            r = requests.post("https://inference.do-ai.run/v1/chat/completions", json=payload, headers=h, timeout=180)
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"] or ""
                s = t.find("{")
                obj = json.loads(t[s:t.rfind("}") + 1])
                return [x for x in obj.get("lessons", []) if isinstance(x, str)][:n]
        except Exception:
            pass
        time.sleep(d); d = min(d * 2, 30)
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bbh", "bbh.json"))
    ap.add_argument("--configs", required=True)
    ap.add_argument("--test-limit", type=int, default=0, help="per-task test cap for fast sweeps (0=full)")
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-bs", type=int, default=8)
    ap.add_argument("--emb-bs", type=int, default=16)
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--no-think", action="store_true", help="disable Qwen3 thinking traces (faster)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bbh", "lab.jsonl"))
    args = ap.parse_args()

    d = json.load(open(args.data))
    rows = {r["key"]: r for r in d["rows"]}
    train = [rows[k] for k in d["train_keys"]]
    test_full = [rows[k] for k in d["test_keys"]]
    tag = os.path.basename(args.model.rstrip("/"))
    os.makedirs(CACHE, exist_ok=True)
    base_path = os.path.join(CACHE, f"base_{tag}.json")
    emb_path = os.path.join(CACHE, f"emb_{tag}.npz")

    M = Model(args.model, args.layer_frac, args.gen_bs, args.emb_bs, no_think=args.no_think)

    def run(rowset, lbk, temp=0.0):
        gens = M.generate([solve_msgs(r["input"], lbk.get(r["key"])) for r in rowset], args.max_new, temp)
        return {r["key"]: {"correct": normalize(extract_answer(g)) == normalize(r["target"]), "text": g}
                for r, g in zip(rowset, gens)}

    from collections import Counter
    def run_vote(rowset, lbk, n, temp):
        msgs = []
        for r in rowset:
            msgs += [solve_msgs(r["input"], lbk.get(r["key"]))] * n   # n samples per item
        gens = M.generate(msgs, args.max_new, temp)
        res = {}
        for j, r in enumerate(rowset):
            ans = [normalize(extract_answer(g)) for g in gens[j * n:(j + 1) * n]]
            maj = Counter(ans).most_common(1)[0][0]
            res[r["key"]] = {"correct": maj == normalize(r["target"]), "text": ""}
        return res

    # ---- cached base (train+test, model-only) ----
    if os.path.exists(base_path) and not args.refresh_cache:
        base = json.load(open(base_path))
        print(f"[cache] base loaded ({len(base)} keys)", flush=True)
    else:
        print("[compute] base train+test (one-time, cached)", flush=True)
        base = {}
        base.update(run(train, {}))
        base.update(run(test_full, {}))
        json.dump(base, open(base_path, "w"))
    base_tr_acc = sum(base[r["key"]]["correct"] for r in train) / len(train)
    base_te_acc = sum(base[r["key"]]["correct"] for r in test_full) / len(test_full)
    print(f"  base: train {base_tr_acc:.1%} | test {base_te_acc:.1%}", flush=True)
    fails = [r for r in train if not base[r["key"]]["correct"]]

    # ---- cached activations ----
    if os.path.exists(emb_path) and not args.refresh_cache:
        z = np.load(emb_path, allow_pickle=True)
        Xf, Et = z["Xf"], z["Et"]
        fail_keys = list(z["fail_keys"]); test_keys = list(z["test_keys"])
        fails = [rows[k] for k in fail_keys]
        print(f"[cache] embeddings loaded (Xf {Xf.shape}, Et {Et.shape})", flush=True)
    else:
        print("[compute] activations for failures + test (one-time, cached)", flush=True)
        Xf = M.embed([solve_msgs(r["input"], None) for r in fails])
        Et = M.embed([solve_msgs(r["input"], None) for r in test_full])
        fail_keys = [r["key"] for r in fails]; test_keys = [r["key"] for r in test_full]
        np.savez(emb_path, Xf=Xf, Et=Et, fail_keys=fail_keys, test_keys=test_keys)

    mu = Xf.mean(0)
    proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)
    Xfn, Etn = proc(Xf), proc(Et)

    # ---- test subset for fast sweeps ----
    if args.test_limit:
        per = {}
        test = []
        for r in test_full:
            per[r["task"]] = per.get(r["task"], 0) + 1
            if per[r["task"]] <= args.test_limit:
                test.append(r)
        idx = {k: i for i, k in enumerate(test_keys)}
        Etn_use = Etn[[idx[r["key"]] for r in test]]
    else:
        test, Etn_use = test_full, Etn
    print(f"[sweep] test size {len(test)}", flush=True)

    def propose_examples(members):
        return "\n\n".join(f"PROBLEM: {m['input'][:350]}\nCORRECT: {m['target']}" for m in members[:6])

    def self_propose(members, cfg, shift):
        m = members[shift % len(members):] + members[:shift % len(members)]
        ex = propose_examples(m)
        instr = ("First think step by step about the shared failure mode, then on a new line write "
                 "'STRATEGY:' followed by ONE general reusable strategy (1-2 sentences)."
                 if cfg.get("cot") else
                 "Write ONE short general reusable strategy (1-2 sentences). Output only the strategy.")
        out = M.generate([[{"role": "user", "content":
              "Problems a model got WRONG (with correct answers):\n\n" + ex + "\n\n" + instr}]],
              cfg.get("lesson_max_new", 128), cfg.get("temp", 0.0))[0]
        if cfg.get("cot") and "STRATEGY:" in out:
            out = out.split("STRATEGY:")[-1]
        return out.strip().strip('"').replace("\n", " ")[:240]

    def candidates(members, cfg):
        n = cfg.get("n_cand", 4)
        prop = cfg.get("proposer", "self")
        if prop != "self":
            cands = do_propose(prop, "Problems a model got WRONG (with correct answers):\n\n"
                               + propose_examples(members), n)
        else:
            cands = [self_propose(members, cfg, s * 3 + 1) for s in range(n)]
        seen, uniq = set(), []
        for c in cands:
            if c and c not in seen:
                seen.add(c); uniq.append(c)
        return uniq

    def build_list(members, cfg):
        if len(members) < 2:
            return []
        cands = candidates(members, cfg)
        if not cfg.get("gate", True):
            return cands[:cfg.get("max_lessons", 4)]
        nd = max(1, min(cfg.get("dev_cap", 4), int(len(members) * 0.3)))
        dev = members[:nd]
        lst, cur = [], sum(run(dev, {r["key"]: None for r in dev})[r["key"]]["correct"] for r in dev) / nd
        while cands and len(lst) < cfg.get("max_lessons", 4):
            best, br = None, cur
            for c in cands:
                rr = run(dev, {r["key"]: block(lst + [c]) for r in dev})
                rate = sum(rr[r["key"]]["correct"] for r in dev) / nd
                if rate > br:
                    best, br = c, rate
            if best is None:
                break
            lst.append(best); cands.remove(best); cur = br
        return lst

    configs = json.load(open(args.configs))
    results = []
    for cfg in configs:
        name = cfg["name"]
        print(f"\n[config] {name}: {cfg}", flush=True)
        if cfg.get("no_lessons"):
            lists, routed = [], {r["key"]: None for r in test}   # consensus-only control (no lessons)
        else:
            K = min(cfg.get("k", 10), len(fails))
            lab, cent = kmeans(Xfn, K)
            lists = [build_list([fails[i] for i in range(len(fails)) if lab[i] == k], cfg) for k in range(K)]
            topk = cfg.get("topk", 1)
            sims = Etn_use @ cent.T
            routed = {}
            for i, r in enumerate(test):
                order = np.argsort(-sims[i])[:topk]
                routed[r["key"]] = block([l for k in order for l in lists[k]]) or None
        nvote = cfg.get("vote", 1)
        res = run_vote(test, routed, nvote, cfg.get("vote_temp", 0.7)) if nvote > 1 else run(test, routed)
        a = sum(res[r["key"]]["correct"] for r in test) / len(test)
        b = sum(1 for r in test if not base[r["key"]]["correct"] and res[r["key"]]["correct"])
        c = sum(1 for r in test if base[r["key"]]["correct"] and not res[r["key"]]["correct"])
        base_sub = sum(base[r["key"]]["correct"] for r in test) / len(test)
        rec = {"config": name, "acc": a, "base_acc": base_sub, "delta": a - base_sub,
               "b": b, "c": c, "p": mcnemar(b, c), "lessons_per_cluster": [len(x) for x in lists],
               "pk": {r["key"]: int(res[r["key"]]["correct"]) for r in test}, "params": cfg}
        results.append(rec)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  -> {name}: {a:.1%} (base {base_sub:.1%}, Δ{a-base_sub:+.1%}, +{b}/-{c}, p={mcnemar(b,c):.4f})", flush=True)

    print("\n===== BBH lab sweep (routed vs cached GREEDY base, test n=%d) =====" % len(test))
    for r in sorted(results, key=lambda x: -x["delta"]):
        print(f"  {r['config']:22} {r['acc']:6.1%}  Δ{r['delta']:+.1%}  +{r['b']}/-{r['c']}  p={r['p']:.4f}")

    # ===== RSI SIGNAL: lessons' MARGINAL lift at MATCHED inference compute =====
    # Consensus alone is just test-time compute. The self-improvement claim = lessons beating the
    # SAME-compute consensus baseline (routed_voteK vs base_voteK), NOT vs greedy base.
    bykey = {r["config"]: r["pk"] for r in results}
    greedy_base = {r["key"]: int(base[r["key"]]["correct"]) for r in test}
    pairs = [("greedy@1", greedy_base, bykey.get("v2_repro")),
             ("maj@3", bykey.get("base_vote3"), bykey.get("routed_vote3")),
             ("maj@5", bykey.get("base_vote5"), bykey.get("routed_vote5")),
             ("maj@9", bykey.get("base_vote9"), bykey.get("routed_vote9"))]
    print("\n===== ★ RSI SIGNAL: lessons' marginal lift at MATCHED compute (routed vs consensus baseline) =====")
    for label, bpk, rpk in pairs:
        if not bpk or not rpk:
            continue
        ks = [r["key"] for r in test if r["key"] in bpk and r["key"] in rpk]
        ba = sum(bpk[k] for k in ks) / len(ks); ra = sum(rpk[k] for k in ks) / len(ks)
        b = sum(1 for k in ks if not bpk[k] and rpk[k]); c = sum(1 for k in ks if bpk[k] and not rpk[k])
        flag = "  <-- RSI" if (ra - ba) > 0 and mcnemar(b, c) < 0.1 else ""
        print(f"  {label:9} consensus-base {ba:5.1%} -> +lessons {ra:5.1%}   Δ{ra-ba:+.1%}  +{b}/-{c}  p={mcnemar(b,c):.4f}{flag}")


if __name__ == "__main__":
    main()
