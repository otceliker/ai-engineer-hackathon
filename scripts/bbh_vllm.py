#!/usr/bin/env python3
"""Consistent 3-shot BBH RSI pipeline, vLLM-accelerated. Phased so vLLM (gen) and HF (activations)
never hold the GPU simultaneously.

  --phase basegen      : vLLM — 3-shot solve ALL train + test (no lessons). Score train -> real failures.
  --phase embed        : HF  — activations for train-failures + test; k-means K clusters (GPU free).
  --phase lessons_test : vLLM — failure-aware lessons from REAL 3-shot failures + 3-shot test lessoned.
  --phase score        : CPU — strict + format-agnostic base vs lessoned; write official.json/.jsonl.

Everything cached (cache/vllm_gen.json keyed base|key / les|key; vllm_fail.json; vllm_clusters.npz;
vllm_lessons.json) so phases resume and re-scoring is free. Same Qwen2.5-7B as solver + embedder + writer.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "results", "bbh", "cache")
COT_DIR = os.path.join(ROOT, "data", "bbh", "cot-prompts")
VGEN = os.path.join(CACHE, "vllm_gen.json")
VFAIL = os.path.join(CACHE, "vllm_fail.json")
VCLUST = os.path.join(CACHE, "vllm_clusters.npz")
VLESS = os.path.join(CACHE, "vllm_lessons.json")
SUMMARY = os.path.join(ROOT, "results", "bbh", "official.json")
PEREX = os.path.join(ROOT, "results", "bbh", "official.jsonl")
TAG = "Qwen__Qwen2.5-7B-Instruct"
URL = "http://localhost:8001/v1/chat/completions"
VMODEL = "qwen"
MAX_NEW = 2048
K = 20
N_CAND = 3
PROPOSE_TOKENS = 320
WORKERS = 48

CUES = [r"answer is\s*:?\s*(.+)", r"answer:\s*(.+)", r"final (?:result|answer)[^.\n]*?\bis\s+(.+)", r"\bis\s*:?\s*(.+)$"]


def extract_strict(t):
    m = list(re.finditer(r"answer is\s*:?\s*(.+)", t, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    ls = [l for l in t.strip().splitlines() if l.strip()]
    return ls[-1].strip() if ls else ""


def extract_robust(t):
    for p in CUES:
        m = list(re.finditer(p, t, re.IGNORECASE | re.MULTILINE))
        if m:
            return m[-1].group(1).strip()
    ls = [l for l in t.strip().splitlines() if l.strip()]
    return ls[-1].strip() if ls else ""


def _nf(s):
    return re.sub(r"\s+", " ", s.strip().lower().strip(".\"'`* )("))


def _match(post, gold):
    g = gold.strip()
    if re.fullmatch(r"\([A-Za-z]\)", g):
        gl = g[1].lower()
        mm = re.search(r"\(([A-Za-z])\)", post) or re.match(r"([A-Za-z])\b", post)
        return bool(mm) and mm.group(1).lower() == gl
    if g.lower() in ("yes", "no"):
        mm = re.match(r"\s*(yes|no)\b", post, re.IGNORECASE)
        return bool(mm) and mm.group(1).lower() == g.lower()
    return _nf(post) == _nf(g) or _nf(g) == _nf(post.split(",")[-1])


def score_strict(text, gold):
    return _match(extract_strict(text).rstrip(".").strip(), gold)


def score_robust(text, gold):
    return _match(extract_robust(text).rstrip(".").strip(), gold)


def atomic(path, obj, raw=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(obj if raw else json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def vchat(messages, max_tokens):
    payload = {"model": VMODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    delay = 2
    for _ in range(5):
        try:
            r = requests.post(URL, json=payload, timeout=300)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 2, 30)
    return None


def cot_map():
    return {f: open(os.path.join(COT_DIR, f)).read() for f in os.listdir(COT_DIR) if f.endswith(".txt")
            } if False else {t: open(os.path.join(COT_DIR, f"{t}.txt")).read()
                             for t in [x[:-4] for x in os.listdir(COT_DIR) if x.endswith(".txt")]}


def user_prompt(cot, inp):
    return cot.strip() + "\n\nQ: " + inp.strip() + "\nA: Let's think step by step."


def run_parallel(jobs, build, max_tokens, gen_cache, label):
    """jobs: list of (ckey, key); build(key)->messages. Fill gen_cache[ckey]=text, cache periodically."""
    todo = [(ck, k) for ck, k in jobs if ck not in gen_cache]
    print(f"[{label}] {len(jobs)} jobs, {len(todo)} to generate", flush=True)
    done = 0

    def work(item):
        ck, k = item
        return ck, vchat(build(k), max_tokens)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for ck, out in ex.map(work, todo):
            if out is not None:
                gen_cache[ck] = out
            done += 1
            if done % 50 == 0:
                atomic(VGEN, gen_cache)
                print(f"  [{label}] {done}/{len(todo)}", flush=True)
    atomic(VGEN, gen_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    A = ap.parse_args()
    data = json.load(open(os.path.join(ROOT, "data", "bbh", "bbh.json")))
    rows = {r["key"]: r for r in data["rows"]}
    train_keys, test_keys = data["train_keys"], data["test_keys"]
    cot = cot_map()
    TASKS = sorted({rows[k]["task"] for k in test_keys})

    if A.phase == "basegen":
        gen = load(VGEN, {})
        jobs = [(f"base|{k}", k) for k in train_keys + test_keys]
        run_parallel(jobs, lambda k: [{"role": "system", "content": "You are a careful problem solver."},
                                      {"role": "user", "content": user_prompt(cot[rows[k]["task"]], rows[k]["input"])}],
                     MAX_NEW, gen, "basegen")
        fails = [k for k in train_keys if f"base|{k}" in gen and not score_strict(gen[f"base|{k}"], rows[k]["target"])]
        atomic(VFAIL, fails)
        tr_acc = sum(score_strict(gen[f"base|{k}"], rows[k]["target"]) for k in train_keys if f"base|{k}" in gen) / max(1, len(train_keys))
        print(f"3-shot TRAIN base acc={tr_acc:.3f} | real failures={len(fails)}/{len(train_keys)}", flush=True)

    elif A.phase == "embed":
        import numpy as np
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from bbh_rsi import Model, kmeans, solve_msgs
        fails = load(VFAIL, [])
        M = Model(os.path.join(ROOT, "models", TAG), 0.6, 8, 16)
        Xf = M.embed([solve_msgs(rows[k]["input"], None) for k in fails])
        Et = M.embed([solve_msgs(rows[k]["input"], None) for k in test_keys])
        mu = Xf.mean(0)
        proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)
        lab, cent = kmeans(proc(Xf), K)
        np.savez(VCLUST, fails=np.array(fails), lab=lab, cent=cent, mu=mu,
                 Et=Et, test_keys=np.array(test_keys))
        print(f"embedded {len(fails)} failures + {len(test_keys)} test; K={K} clusters", flush=True)

    elif A.phase == "lessons_test":
        import numpy as np
        gen = load(VGEN, {})
        z = np.load(VCLUST, allow_pickle=True)
        fails = list(z["fails"]); lab = z["lab"]; cent = z["cent"]; mu = z["mu"]
        Et = z["Et"]; tk = list(z["test_keys"])
        proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)

        def fail_block(k):
            wrong = gen.get(f"base|{k}", "").strip()
            snip = wrong if len(wrong) <= 480 else wrong[:240] + " […] " + wrong[-240:]
            return (f"PROBLEM: {rows[k]['input'][:300]}\nYOUR WRONG ANSWER: {extract_strict(wrong)[:100]}\n"
                    f"YOUR REASONING: {snip}\nCORRECT ANSWER: {rows[k]['target']}")

        if os.path.exists(VLESS):
            lessons = json.load(open(VLESS))
        else:
            lessons = []
            for c in range(K):
                mem = [fails[i] for i in range(len(fails)) if lab[i] == c]
                ls = []
                for s in range(N_CAND):
                    if not mem:
                        break
                    m = mem[(s * 3 + 1) % len(mem):] + mem[:(s * 3 + 1) % len(mem)]
                    ex = "\n\n---\n\n".join(fail_block(k) for k in m[:5])
                    out = vchat([{"role": "user", "content":
                          "Below are problems you previously answered INCORRECTLY — your own wrong answer "
                          "and reasoning, plus the correct answer:\n\n" + ex +
                          "\n\nDiagnose the common mistake and write ONE short, general, reusable strategy "
                          "(1-2 sentences) to prevent it. Output ONLY the strategy."}], PROPOSE_TOKENS)
                    if out:
                        c2 = out.strip().strip('"').replace("\n", " ")[:280]
                        if c2 and c2 not in ls:
                            ls.append(c2)
                lessons.append(ls)
            atomic(VLESS, lessons)
            print("built failure-aware lessons (K=%d)" % K, flush=True)
        Xt = proc(Et)
        nearest = (Xt @ cent.T).argmax(1)
        block = lambda L: "\n".join(f"- {x}" for x in L)
        jobs = []
        for i, k in enumerate(tk):
            jobs.append((f"les|{k}", k))
        rt = {k: int(nearest[i]) for i, k in enumerate(tk)}

        def build(k):
            L = lessons[rt[k]] or ["Think carefully and verify each step."]
            return [{"role": "system", "content": "You are a careful problem solver.\n\nLessons learned from past mistakes:\n" + block(L)},
                    {"role": "user", "content": user_prompt(cot[rows[k]["task"]], rows[k]["input"])}]
        run_parallel(jobs, build, MAX_NEW, gen, "test-lessoned")

    elif A.phase == "score":
        gen = load(VGEN, {})
        recs, ts = [], {t: {"b": [0, 0, 0], "l": [0, 0, 0]} for t in TASKS}  # strict_ok, robust_ok, n
        for k in test_keys:
            r = rows[k]; t = r["task"]
            rec = {"key": k, "task": t, "gold": r["target"]}
            for arm, ck in (("b", f"base|{k}"), ("l", f"les|{k}")):
                if ck in gen:
                    s = score_strict(gen[ck], r["target"]); rb = score_robust(gen[ck], r["target"])
                    rec[arm + "_strict"] = s; rec[arm + "_robust"] = rb
                    rec[arm + "_ans"] = extract_robust(gen[ck])[:120]
                    ts[t][arm][0] += s; ts[t][arm][1] += rb; ts[t][arm][2] += 1
            recs.append(rec)
        atomic(PEREX, "\n".join(json.dumps(x) for x in recs), raw=True)
        # webview summary uses robust (the fair metric)
        tasks_out = []
        tot = {"b": [0, 0, 0], "l": [0, 0, 0]}
        for t in TASKS:
            b, l = ts[t]["b"], ts[t]["l"]
            tasks_out.append({"task": t, "base_acc": (b[1] / b[2] if b[2] else None), "base_n": b[2],
                              "les_acc": (l[1] / l[2] if l[2] else None), "les_n": l[2]})
            for a in ("b", "l"):
                for j in range(3):
                    tot[a][j] += ts[t][a][j]
        ov = {"base": (tot["b"][1] / tot["b"][2] if tot["b"][2] else None), "base_n": tot["b"][2],
              "les": (tot["l"][1] / tot["l"][2] if tot["l"][2] else None), "les_n": tot["l"][2]}
        if ov["base"] is not None and ov["les"] is not None:
            ov["delta"] = ov["les"] - ov["base"]
        atomic(SUMMARY, {"model": TAG, "variant": "vllm-3shot-failk20",
                         "protocol": "FULL 3-shot consistent: 3-shot train base->failures->failure-aware lessons K=20->3-shot test (robust scoring)",
                         "max_new": MAX_NEW, "updated": time.strftime("%H:%M:%S"),
                         "done": tot["b"][2] + tot["l"][2], "total": 2 * len(test_keys),
                         "tasks": tasks_out, "overall": ov})
        # console: both metrics
        for metric, idx in (("STRICT", 0), ("ROBUST", 1)):
            B = tot["b"][idx] / tot["b"][2] if tot["b"][2] else 0
            L = tot["l"][idx] / tot["l"][2] if tot["l"][2] else 0
            print(f"{metric:7} OVERALL base={B:.3f} les={L:.3f} delta={L-B:+.3f} (n={tot['b'][2]}/{tot['l'][2]})", flush=True)
        print("\nper-task (robust):", flush=True)
        for to in tasks_out:
            if to["base_acc"] is not None and to["les_acc"] is not None:
                print(f"  {to['task']:40} base={to['base_acc']:.3f} les={to['les_acc']:.3f} d={to['les_acc']-to['base_acc']:+.3f}", flush=True)


if __name__ == "__main__":
    main()
