#!/usr/bin/env python3
"""Fully self-contained BBH RSI: ONE Qwen2.5-7B (HF, single load) is agent + proposer + embedder.

Pipeline (one process, one model load):
  1. agent  : zero-shot CoT solve BBH train+test, exact-match score
  2. embed  : 7B last-token hidden state of each TRAIN-FAILURE prompt  (the model's "read" of the task)
  3. cluster: k-means on those activations -> K failure-mode clusters (label-free; no gold task used)
  4. propose: 7B writes ONE general lesson per cluster from its examples
  5. route  : embed each TEST prompt, assign nearest cluster centroid (cosine) -> its lesson  (NO gold label)
  6. arms   : re-solve TEST with base / global-pile(all lessons) / routed-top1 / oracle(gold task lesson)
  7. compare: per-arm accuracy + paired McNemar vs base

Run with the torch/transformers venv (.venv). Reads data/bbh/bbh.json (from bbh_prep.py).
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "You are a careful problem solver."
COT = "\n\nThink step by step. Then end your reply with 'So the answer is X.' where X is the final answer."


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def normalize(s):
    s = (s or "").strip().lower()
    s = s.strip("\"'` ")
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def extract_answer(text):
    m = list(re.finditer(r"answer is\s*:?\s*(.+?)(?:[.\n]|$)", text, re.IGNORECASE))
    if m:
        return m[-1].group(1)
    # fallback: last non-empty line
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def kmeans(X, K, iters=60, seed=0):
    rng = np.random.RandomState(seed)
    c = X[rng.choice(len(X), size=min(K, len(X)), replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newc = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(len(c))])
        if np.allclose(newc, c):
            break
        c = newc
    return lab, c


class Model:
    def __init__(self, path, layer_frac, gen_bs, emb_bs, no_think=False):
        self.no_think = no_think
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16, device_map="cuda")
        self.m.eval()
        self.layer_frac = layer_frac
        self.gen_bs, self.emb_bs = gen_bs, emb_bs

    def _ids(self, texts):
        kw = {"tokenize": False, "add_generation_prompt": True}
        if self.no_think:
            kw["enable_thinking"] = False   # Qwen3: skip <think> traces (faster, clean extraction)
        try:
            strs = [self.tok.apply_chat_template(t, **kw) for t in texts]
        except TypeError:                     # models whose template lacks enable_thinking (e.g. Qwen2.5)
            kw.pop("enable_thinking", None)
            strs = [self.tok.apply_chat_template(t, **kw) for t in texts]
        return self.tok(strs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")

    @torch.no_grad()
    def generate(self, msg_lists, max_new=512, temperature=0.0):
        out = []
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        for i in range(0, len(msg_lists), self.gen_bs):
            enc = self._ids(msg_lists[i:i + self.gen_bs])
            g = self.m.generate(**enc, **kw)
            new = g[:, enc["input_ids"].shape[1]:]
            out += self.tok.batch_decode(new, skip_special_tokens=True)
            print(f"    gen {min(i+self.gen_bs,len(msg_lists))}/{len(msg_lists)}", flush=True)
        return out

    @torch.no_grad()
    def embed(self, msg_lists):
        vecs = []
        for i in range(0, len(msg_lists), self.emb_bs):
            enc = self._ids(msg_lists[i:i + self.emb_bs])
            hs = self.m(**enc, output_hidden_states=True).hidden_states
            layer = int(len(hs) * self.layer_frac)
            vecs.append(hs[layer][:, -1, :].float().cpu().numpy())
        return np.concatenate(vecs, 0)


def solve_msgs(task_input, lesson):
    sys_c = SYS + (f"\n\nLessons learned from past mistakes:\n{lesson}" if lesson else "")
    return [{"role": "system", "content": sys_c}, {"role": "user", "content": task_input + COT}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bbh", "bbh.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--emb-bs", type=int, default=32)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bbh", "bbh_rsi.json"))
    args = ap.parse_args()

    d = json.load(open(args.data))
    rows = {r["key"]: r for r in d["rows"]}
    train = [rows[k] for k in d["train_keys"]]
    test = [rows[k] for k in d["test_keys"]]
    print(f"model={os.path.basename(args.model)} train={len(train)} test={len(test)} K={args.k}", flush=True)
    M = Model(args.model, args.layer_frac, args.gen_bs, args.emb_bs)

    def run(rowset, lessons_by_key):
        gens = M.generate([solve_msgs(r["input"], lessons_by_key.get(r["key"])) for r in rowset], args.max_new)
        res = {}
        for r, g in zip(rowset, gens):
            res[r["key"]] = normalize(extract_answer(g)) == normalize(r["target"])
        return res

    def acc(res, rowset):
        return sum(res[r["key"]] for r in rowset) / len(rowset)

    # 1. base
    print("[1] base train+test", flush=True)
    base_tr = run(train, {})
    base_te = run(test, {})
    print(f"  base: train {acc(base_tr,train):.1%} | test {acc(base_te,test):.1%}", flush=True)

    # 2. embed train failures
    fails = [r for r in train if not base_tr[r["key"]]]
    print(f"[2] embed {len(fails)} train failures", flush=True)
    Xf = M.embed([solve_msgs(r["input"], None) for r in fails])
    mu = Xf.mean(0)
    def proc(X):
        Y = X - mu
        return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    Xfn = proc(Xf)

    # 3. cluster
    K = min(args.k, len(fails))
    print(f"[3] k-means K={K}", flush=True)
    lab, cent = kmeans(Xfn, K)

    # 4. propose one lesson per cluster
    print("[4] propose lessons", flush=True)
    def propose_msg(examples):
        ex = "\n\n".join(
            f"PROBLEM: {e['input'][:350]}\nMODEL ANSWERED: {e['pred']} (WRONG)\nCORRECT: {e['target']}"
            for e in examples)
        return [{"role": "user", "content":
                 "Here are problems a model got WRONG (its answer vs the correct answer):\n\n" + ex +
                 "\n\nWrite ONE short, general, reusable strategy (1-2 imperative sentences) that would help "
                 "solve problems like these correctly. It must be a general method, NOT the answer to any "
                 "specific problem. Output only the strategy."}]
    cluster_lessons = []
    for k in range(K):
        members = [fails[i] for i in range(len(fails)) if lab[i] == k][:6]
        exs = [{"input": m["input"], "pred": "(incorrect)", "target": m["target"]} for m in members]
        lesson = M.generate([propose_msg(exs)], 128)[0].strip().replace("\n", " ")
        cluster_lessons.append(lesson)
        print(f"  cluster {k} (n={int((lab==k).sum())}): {lesson[:110]}", flush=True)

    # oracle: one lesson per gold task from that task's failures
    print("[4b] propose oracle per-task lessons", flush=True)
    tasks = sorted({r["task"] for r in fails})
    task_lessons = {}
    for t in tasks:
        members = [r for r in fails if r["task"] == t][:6]
        exs = [{"input": m["input"], "pred": "(incorrect)", "target": m["target"]} for m in members]
        task_lessons[t] = M.generate([propose_msg(exs)], 128)[0].strip().replace("\n", " ")

    # 5. route test by nearest centroid (cosine on processed embeddings)
    print("[5] embed+route test", flush=True)
    Xt = proc(M.embed([solve_msgs(r["input"], None) for r in test]))
    sims = Xt @ cent.T
    nearest = sims.argmax(1)
    routed_lessons = {r["key"]: cluster_lessons[nearest[i]] for i, r in enumerate(test)}
    global_block = "\n".join(f"- {l}" for l in cluster_lessons)
    global_lessons = {r["key"]: global_block for r in test}
    oracle_lessons = {r["key"]: task_lessons.get(r["task"], "") for r in test}

    # 6. arms on test
    print("[6] arms: global / routed / oracle", flush=True)
    arms = {"base": base_te,
            "global": run(test, global_lessons),
            "routed": run(test, routed_lessons),
            "oracle": run(test, oracle_lessons)}

    # 7. compare
    summary = {"model": os.path.basename(args.model), "K": K, "n_test": len(test),
               "n_train_fail": len(fails), "cluster_lessons": cluster_lessons, "arms": {}}
    print("\n===== BBH RSI (held-out test) =====")
    for name, res in arms.items():
        a = acc(res, test)
        b = sum(1 for r in test if not base_te[r["key"]] and res[r["key"]])
        c = sum(1 for r in test if base_te[r["key"]] and not res[r["key"]])
        p = mcnemar(b, c)
        summary["arms"][name] = {"acc": a, "b": b, "c": c, "p": p}
        cmp = "—" if name == "base" else f"+{b}/-{c}  p={p:.4f}"
        print(f"  {name:8} {a:6.1%}   {cmp}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
