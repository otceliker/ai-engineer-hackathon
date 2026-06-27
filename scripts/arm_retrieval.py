#!/usr/bin/env python3
"""Arm A — retrieval-only baseline (no weight updates), matched to the LoRA loop.

Consumes the SAME per-round correct-trace pool the LoRA run saved, and the SAME
held-out (from that run's manifest). For each held-out problem, BM25-retrieve the
top-k most similar solved (problem, trace) pairs, prepend them as worked examples,
generate greedy, grade. The store grows each round exactly as the LoRA pool did.

Logs retrieval similarity per held-out problem so a retrieval "win" can be checked
against near-duplicates (kNN-over-solutions) rather than genuine analogy transfer.

Usage:
  python scripts/arm_retrieval.py --base models/Qwen__Qwen2.5-1.5B-Instruct \
      --pool-run full_Prealgebra --k 3
"""
import argparse, csv, json, math, os, re, signal, subprocess, sys, time
from collections import defaultdict

try:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from math_verify import parse, verify
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(ROOT, "data", "nlile__hendrycks-MATH-benchmark", "data",
                             "train-00000-of-00001.parquet")
PY = sys.executable
VENV_BIN = os.path.dirname(PY)
SYSTEM = "Solve the problem step by step and put your final answer within \\boxed{}."
SUBENV = {**os.environ, "PATH": VENV_BIN + ":" + os.environ.get("PATH", ""),
          "VLLM_USE_FLASHINFER_SAMPLER": "0"}


# ---- GPU subprocess plumbing (same discipline as star_loop) -----------------
def kill_gpu_procs():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    for t in out.stdout.split():
        if t.strip().isdigit():
            try:
                os.kill(int(t), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def ensure_gpu_free(min_free_gib, timeout=180):
    deadline = time.time() + timeout
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        free = int(out.stdout.strip().split("\n")[0]) / 1024.0
        if free >= min_free_gib or time.time() > deadline:
            return free
        time.sleep(3)


def gen(base, prompts_rows, out, max_tokens=512, max_model_len=4096):
    pf = out + ".prompts.jsonl"
    with open(pf, "w") as f:
        for r in prompts_rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    ensure_gpu_free(19)
    p = subprocess.Popen([PY, os.path.join(ROOT, "scripts", "star_gen.py"), "--base", base,
                          "--prompts", pf, "--out", out, "--n", "1", "--temp", "0.0",
                          "--max-tokens", str(max_tokens), "--max-model-len", str(max_model_len)],
                         env=SUBENV, cwd=ROOT, start_new_session=True)
    ret = p.wait()
    kill_gpu_procs()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, "star_gen")
    by_id = {}
    for line in open(out):
        d = json.loads(line); by_id[d["id"]] = d["text"]
    return by_id


# ---- BM25 (zero-dependency lexical retrieval) -------------------------------
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.df = defaultdict(int)
        for d in self.docs:
            for w in set(d):
                self.df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in self.df.items()}
        self.tf = [defaultdict(int) for _ in self.docs]
        for i, d in enumerate(self.docs):
            for w in d:
                self.tf[i][w] += 1

    def top(self, query, k):
        q = tok(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q:
                if w in self.tf[i]:
                    f = self.tf[i][w]
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            scores.append(s)
        order = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i]) for i in order]


def grade(text, gold):
    try:
        return bool(verify(parse(gold), parse(text)))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pool-run", required=True, help="run dir whose pools + held-out to reuse")
    ap.add_argument("--k", type=int, default=3, help="retrieved worked examples per problem")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    run = os.path.join(ROOT, "runs", args.pool_run)
    man = json.load(open(os.path.join(run, "manifest.json")))
    held_ids, category = set(man["held_ids"]), man["category"]
    tokn = AutoTokenizer.from_pretrained(args.base)
    allrows = {r["unique_id"]: r for r in pq.read_table(TRAIN_PARQUET).to_pylist()
               if r["subject"] == category}
    held = [allrows[i] for i in man["held_ids"]]
    gold = {r["unique_id"]: r["answer"] for r in held}

    rounds = args.rounds or man["rounds"]
    out_dir = os.path.join(run, "arms", "retrieval"); os.makedirs(out_dir, exist_ok=True)
    mf = open(os.path.join(out_dir, "metrics.csv"), "w", newline="")
    w = csv.writer(mf); w.writerow(["round", "heldout_acc", "k", "mean_top1_bm25", "box_rate"])
    base_acc = float([r for r in csv.DictReader(open(os.path.join(run, "metrics.csv")))
                      if int(r["round"]) < 0][0]["heldout_acc"])
    w.writerow([-1, f"{base_acc:.4f}", 0, 0, ""])
    print(f"Arm A retrieval | category {category} | held {len(held)} | base {base_acc:.1%}", flush=True)

    for t in range(rounds):
        pool = [json.loads(l) for l in open(os.path.join(run, f"round{t}", "correct_pool.jsonl"))]
        # one representative trace per solved problem
        rep = {}
        for d in pool:
            rep.setdefault(d["id"], d["trace"])
        pids = [i for i in rep if i in allrows]
        corpus = [allrows[i]["problem"] for i in pids]
        bm = BM25(corpus)
        prompts, top1 = [], []
        retr_log = []
        for r in held:
            hits = bm.top(r["problem"], args.k)
            top1.append(hits[0][1] if hits else 0.0)
            msgs = [{"role": "system", "content": SYSTEM}]
            for idx, sc in hits:
                ex = allrows[pids[idx]]
                msgs += [{"role": "user", "content": ex["problem"]},
                         {"role": "assistant", "content": rep[pids[idx]]}]
            msgs.append({"role": "user", "content": r["problem"]})
            prompts.append({"id": r["unique_id"],
                            "prompt": tokn.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)})
            retr_log.append({"id": r["unique_id"], "retrieved": [pids[i] for i, _ in hits],
                             "bm25": [round(s, 2) for _, s in hits]})
        outs = gen(args.base, prompts, os.path.join(out_dir, f"round{t}_eval.jsonl"))
        cor = {i: grade(outs[i], gold[i]) for i in [r["unique_id"] for r in held]}
        acc = sum(cor.values()) / len(held)
        box = sum("\\boxed" in outs[i] for i in cor) / len(held)
        mean_top1 = sum(top1) / len(top1)
        json.dump(retr_log, open(os.path.join(out_dir, f"round{t}_retrieval.json"), "w"))
        with open(os.path.join(out_dir, f"round{t}_graded.jsonl"), "w") as f:
            for i in cor:
                f.write(json.dumps({"id": i, "correct": cor[i], "text": outs[i]}) + "\n")
        w.writerow([t, f"{acc:.4f}", args.k, f"{mean_top1:.2f}", f"{box:.3f}"]); mf.flush()
        print(f"[round {t}] retrieval acc {acc:.1%} | pool {len(pids)} solved-problems | "
              f"mean top-1 BM25 {mean_top1:.2f} | box {box:.0%}", flush=True)
    mf.close()
    print(f"\nSaved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
