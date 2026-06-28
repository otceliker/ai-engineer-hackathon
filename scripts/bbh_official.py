#!/usr/bin/env python3
"""Official BBH protocol (3-shot CoT, suzgunmirac/BIG-Bench-Hard) — base vs lessoned, LIVE + CACHED.

Official protocol (3-shot prompts pin gold format) + proper matching (MC letter / yes-no / free-form),
MAX_NEW=2048 so lesson-induced CoT isn't truncated.

Lesson-writer ("failure-aware"): for each k-means cluster of the model's OWN train failures, we show the
model its OWN WRONG output (answer + reasoning tail) alongside the correct answer, and ask it to diagnose
the mistake pattern and write a reusable strategy. Routed to test prompts by activation similarity.

Durable/live: base generations cached as "base|key" (shared across variants); lessoned as
"les|<variant>|key" (variant = failk<K>); cluster lessons cached per variant. After every chunk we
re-score all cached outputs and atomically rewrite results/bbh/official.json (+ official.jsonl).

Usage: bbh_official.py [--k 20] [--per-task 25] [--tasks all|t1,t2,...] [--propose-tokens 320]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bbh_rsi import Model, kmeans  # noqa: E402

TAG = "Qwen__Qwen2.5-7B-Instruct"
CACHE = os.path.join(ROOT, "results", "bbh", "cache")
COT_DIR = os.path.join(ROOT, "data", "bbh", "cot-prompts")
GEN_CACHE = os.path.join(CACHE, "official_gen.json")
SUMMARY = os.path.join(ROOT, "results", "bbh", "official.json")
PEREX = os.path.join(ROOT, "results", "bbh", "official.jsonl")
ALL_TASKS = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
             "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
             "logical_deduction_five_objects", "logical_deduction_seven_objects",
             "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
             "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
             "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
             "temporal_sequences", "tracking_shuffled_objects_five_objects",
             "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
             "web_of_lies", "word_sorting"]
MAX_NEW = 2048
N_CAND = 3


def extract(text):
    m = list(re.finditer(r"answer is\s*:?\s*(.+)", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).strip()
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1].strip() if lines else ""


def _nf(s):
    s = s.strip().lower().strip(".\"'`* ")
    return re.sub(r"\s+", " ", s)


def score(pred_text, gold):
    post = extract(pred_text).rstrip(".").strip()
    g = gold.strip()
    if re.fullmatch(r"\([A-Za-z]\)", g):
        gl = g[1].lower()
        mm = re.search(r"\(([A-Za-z])\)", post)
        if mm:
            return mm.group(1).lower() == gl
        mm = re.match(r"([A-Za-z])\b", post)
        return bool(mm) and mm.group(1).lower() == gl
    if g.lower() in ("yes", "no"):
        mm = re.match(r"\s*(yes|no)\b", post, re.IGNORECASE)
        return bool(mm) and mm.group(1).lower() == g.lower()
    return _nf(post) == _nf(g)


def atomic_write(path, obj, raw=False):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(obj if raw else json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--per-task", type=int, default=25)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--propose-tokens", type=int, default=320)
    A = ap.parse_args()
    K = A.k
    TASKS = ALL_TASKS if A.tasks == "all" else A.tasks.split(",")
    variant = f"failk{K}"
    less_cache_path = os.path.join(CACHE, f"official_lessons_{variant}.json")

    cot = {t: open(os.path.join(COT_DIR, f"{t}.txt")).read() for t in TASKS}
    base = json.load(open(os.path.join(CACHE, f"base_{TAG}.json")))
    z = np.load(os.path.join(CACHE, f"emb_{TAG}.npz"), allow_pickle=True)
    Xf, Et = z["Xf"], z["Et"]
    fail_keys, test_keys = list(z["fail_keys"]), list(z["test_keys"])
    rows = {r["key"]: r for r in json.load(open(os.path.join(ROOT, "data", "bbh", "bbh.json")))["rows"]}
    fails = [rows[k] for k in fail_keys]

    gen_cache = json.load(open(GEN_CACHE)) if os.path.exists(GEN_CACHE) else {}
    print(f"variant={variant} K={K} tasks={len(TASKS)} per_task={A.per_task} | "
          f"{len(gen_cache)} cached gens", flush=True)

    M = Model(os.path.join(ROOT, "models", TAG), 0.6, 8, 16)
    mu = Xf.mean(0)
    proc = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-8)
    lab, cent = kmeans(proc(Xf), K)

    def fail_block(e):
        wrong = base[e["key"]]["text"].strip()
        snip = wrong if len(wrong) <= 480 else wrong[:240] + " […] " + wrong[-240:]
        return (f"PROBLEM: {e['input'][:300]}\n"
                f"YOUR WRONG ANSWER: {extract(wrong)[:100]}\n"
                f"YOUR REASONING: {snip}\n"
                f"CORRECT ANSWER: {e['target']}")

    def propose(members, shift):
        m = members[shift % len(members):] + members[:shift % len(members)]
        ex = "\n\n---\n\n".join(fail_block(e) for e in m[:5])
        prompt = ("Below are problems you previously answered INCORRECTLY — your own wrong answer and "
                  "reasoning, plus the correct answer:\n\n" + ex +
                  "\n\nDiagnose the common mistake you made and write ONE short, general, reusable "
                  "strategy (1-2 sentences) that would prevent it. Output ONLY the strategy.")
        out = M.generate([[{"role": "user", "content": prompt}]], A.propose_tokens)[0]
        return out.strip().strip('"').replace("\n", " ")[:280]

    if os.path.exists(less_cache_path):
        cluster_lessons = json.load(open(less_cache_path))
        print(f"loaded cached lessons ({variant})", flush=True)
    else:
        print(f"building failure-aware cluster lessons (K={K}, {A.propose_tokens} tok)...", flush=True)
        cluster_lessons = []
        for k in range(K):
            mem = [fails[i] for i in range(len(fails)) if lab[i] == k]
            ls = []
            if mem:
                for s in range(N_CAND):
                    c = propose(mem, s * 3 + 1)
                    if c and c not in ls:
                        ls.append(c)
            cluster_lessons.append(ls)
        atomic_write(less_cache_path, cluster_lessons)
    block = lambda lst: "\n".join(f"- {l}" for l in lst)

    Xt = proc(Et)
    tk_idx = {k: i for i, k in enumerate(test_keys)}
    subset_by_task = {t: [k for k in test_keys if rows[k]["task"] == t][:A.per_task] for t in TASKS}

    def ckey(arm, k):
        return f"base|{k}" if arm == "base" else f"les|{variant}|{k}"

    def build_msg(arm, k):
        r = rows[k]
        user = cot[r["task"]].strip() + "\n\nQ: " + r["input"].strip() + "\nA: Let's think step by step."
        if arm == "les":
            j = int((Xt[tk_idx[k]:tk_idx[k] + 1] @ cent.T).argmax())
            les = cluster_lessons[j] or ["Think carefully and check each step."]
            sysc = "You are a careful problem solver.\n\nLessons learned from past mistakes:\n" + block(les)
        else:
            sysc = "You are a careful problem solver."
        return [{"role": "system", "content": sysc}, {"role": "user", "content": user}]

    jobs = []
    for t in TASKS:
        for k in subset_by_task[t]:
            jobs += [("base", k), ("les", k)]

    def write_summary():
        recs, tsum = [], {t: {"base": [0, 0], "les": [0, 0]} for t in TASKS}
        for t in TASKS:
            for k in subset_by_task[t]:
                r = rows[k]
                rec = {"key": k, "task": t, "gold": r["target"]}
                for arm in ("base", "les"):
                    if ckey(arm, k) in gen_cache:
                        out = gen_cache[ckey(arm, k)]
                        ok = score(out, r["target"])
                        rec[arm + "_ok"] = ok
                        rec[arm + "_ans"] = extract(out)[:120]
                        rec[arm + "_tail"] = out[-300:]
                        tsum[t][arm][0] += int(ok)
                        tsum[t][arm][1] += 1
                recs.append(rec)
        atomic_write(PEREX, "\n".join(json.dumps(r) for r in recs), raw=True)
        tasks_out, tot = [], {"base": [0, 0], "les": [0, 0]}
        for t in TASKS:
            b, l = tsum[t]["base"], tsum[t]["les"]
            tasks_out.append({"task": t, "base_acc": (b[0] / b[1] if b[1] else None), "base_n": b[1],
                              "les_acc": (l[0] / l[1] if l[1] else None), "les_n": l[1]})
            for arm in ("base", "les"):
                tot[arm][0] += tsum[t][arm][0]
                tot[arm][1] += tsum[t][arm][1]
        ov = {"base": (tot["base"][0] / tot["base"][1] if tot["base"][1] else None), "base_n": tot["base"][1],
              "les": (tot["les"][0] / tot["les"][1] if tot["les"][1] else None), "les_n": tot["les"][1]}
        if ov["base"] is not None and ov["les"] is not None:
            ov["delta"] = ov["les"] - ov["base"]
        summary = {"model": TAG, "variant": variant,
                   "protocol": f"official 3-shot CoT, failure-aware lessons K={K}",
                   "max_new": MAX_NEW, "updated": time.strftime("%H:%M:%S"),
                   "done": tot["base"][1] + tot["les"][1], "total": len(jobs),
                   "tasks": tasks_out, "overall": ov}
        atomic_write(SUMMARY, summary)
        return summary

    write_summary()
    todo = [(a, k) for (a, k) in jobs if ckey(a, k) not in gen_cache]
    print(f"{len(jobs)} jobs, {len(todo)} to generate (2048 tok)", flush=True)
    CH = M.gen_bs
    for i in range(0, len(todo), CH):
        chunk = todo[i:i + CH]
        outs = M.generate([build_msg(a, k) for (a, k) in chunk], MAX_NEW)
        for (a, k), o in zip(chunk, outs):
            gen_cache[ckey(a, k)] = o
        atomic_write(GEN_CACHE, gen_cache)
        s = write_summary()
        ob, ol = s["overall"]["base"], s["overall"]["les"]
        print(f"  [{s['done']}/{s['total']}] base={None if ob is None else round(ob,3)} "
              f"les={None if ol is None else round(ol,3)}", flush=True)

    s = write_summary()
    print("\n" + "=" * 70)
    for t in s["tasks"]:
        if t["base_acc"] is not None:
            print(f"{t['task']:42} base={t['base_acc']:.2f} les={t['les_acc']:.2f} (n={t['base_n']})")
    o = s["overall"]
    print("-" * 70)
    print(f"{'OVERALL':42} base={o['base']:.3f} les={o['les']:.3f} delta {o.get('delta',0):+.3f} (n={o['base_n']})")


if __name__ == "__main__":
    main()
