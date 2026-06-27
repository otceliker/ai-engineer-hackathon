#!/usr/bin/env python3
"""Grader-driven LoRA self-improvement loop (STaR / rejection-sampling fine-tuning).

Plain loop, single category, frontier expansion instrumented from the start.
Generate with latest adapter (base at round 0), keep Math-Verify-correct traces,
train a LoRA FROM BASE on the cumulative pool, eval greedy on a fixed held-out set.

Design (locked + review changes folded in):
  - generate=sample, eval=greedy; train-from-base every round on the cumulative pool
  - held-out drawn from MATH train split with IDs DISJOINT from the train pool (#3:
    disjoint IDs is the real leakage constraint; larger N for power), never trained on
  - false-positive guard (#2): keep a trace only if Math-Verify-correct AND has a real
    \\boxed{} AND shows work (not an answer-only one-liner)
  - frontier vs BASE model (held-out problems the base could not solve), + train-side
    newly-solved per round (leading indicator); flip matrix + McNemar vs base
  - prompts are zero-shot-with-instruction (no few-shot scaffold) so eval is zero-shot
    and box-emission rate is a real signal (#5)
  - GPU work runs in subprocesses (star_gen vLLM / lora_train PEFT) so the GPU is fully
    released between generate and train (#4 teardown/rebuild)

Usage:
  python scripts/star_loop.py --base models/Qwen__Qwen2.5-1.5B-Instruct \
      --category Prealgebra --train-pool 400 --held 150 --rounds 4 --k 6
"""
import argparse, csv, json, math, os, random, signal, subprocess, sys, time
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


def kill_gpu_procs():
    """Reap any process still holding GPU memory. vLLM's EngineCore puts itself in
    its own session (survives killpg of the parent), so we target it by GPU usage.
    Safe here: llm-server is stopped, so nothing else legitimately uses the GPU
    between our steps."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    for tok_ in out.stdout.split():
        if tok_.strip().isdigit():
            try:
                os.kill(int(tok_), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def run_isolated(cmd):
    """Run a GPU subprocess in its own session; on exit reap any GPU-holding orphan."""
    p = subprocess.Popen(cmd, env=SUBENV, cwd=ROOT, start_new_session=True)
    ret = p.wait()
    kill_gpu_procs()                            # reap vLLM EngineCore orphan
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def ensure_gpu_free(min_free_gib, timeout=180):
    """vLLM's EngineCore subprocess can linger holding VRAM after the parent exits.
    Poll until enough is actually free before launching the next GPU step."""
    deadline = time.time() + timeout
    while True:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        free = int(out.stdout.strip().split("\n")[0]) / 1024.0
        if free >= min_free_gib or time.time() > deadline:
            if free < min_free_gib:
                print(f"[warn] only {free:.1f} GiB free after {timeout}s (wanted {min_free_gib})", flush=True)
            return free
        time.sleep(3)


def build_prompt(tok, problem):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def grade(text, gold):
    try:
        return bool(verify(parse(gold), parse(text)))
    except Exception:
        return False


def keep_trace(text, gold):
    """#2 false-positive guard: correct AND real box AND shows work."""
    return grade(text, gold) and "\\boxed" in text and len(text.split()) >= 8


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n))


def gen(base, adapter, prompts_rows, out, n, temp, max_tokens):
    pf = out + ".prompts.jsonl"
    with open(pf, "w") as f:
        for r in prompts_rows:
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"]}) + "\n")
    cmd = [PY, os.path.join(ROOT, "scripts", "star_gen.py"), "--base", base,
           "--prompts", pf, "--out", out, "--n", str(n), "--temp", str(temp),
           "--max-tokens", str(max_tokens)]
    if adapter:
        cmd += ["--adapter", adapter]
    ensure_gpu_free(19)                         # vLLM wants ~20 GiB (gpu_mem 0.85)
    run_isolated(cmd)
    by_id = defaultdict(list)
    for line in open(out):
        d = json.loads(line); by_id[d["id"]].append(d["text"])
    return by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--category", default="Prealgebra")
    ap.add_argument("--train-pool", type=int, default=400)
    ap.add_argument("--held", type=int, default=150)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=6, help="samples/problem when harvesting")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--cap", type=int, default=3, help="max traces kept per problem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    rows = [r for r in pq.read_table(TRAIN_PARQUET).to_pylist() if r["subject"] == args.category]
    random.Random(args.seed).shuffle(rows)
    pool_rows = rows[:args.train_pool]
    held_rows = rows[args.train_pool:args.train_pool + args.held]
    pool_ids = {r["unique_id"] for r in pool_rows}
    held_ids = {r["unique_id"] for r in held_rows}
    assert pool_ids.isdisjoint(held_ids), "train/held-out ID leakage!"

    def mk(rrows):
        return [{"id": r["unique_id"], "prompt": build_prompt(tok, r["problem"]),
                 "problem": r["problem"], "gold": r["answer"]} for r in rrows]
    pool, held = mk(pool_rows), mk(held_rows)
    gold = {r["id"]: r["gold"] for r in pool + held}

    run = os.path.join(ROOT, "runs", f"{args.tag or 'star'}_{args.category.replace(' ', '_')}")
    os.makedirs(run, exist_ok=True)
    json.dump({"base": args.base, "category": args.category, "train_pool": len(pool),
               "held": len(held), "rounds": args.rounds, "k": args.k, "temp": args.temp,
               "epochs": args.epochs, "cap": args.cap, "seed": args.seed,
               "pool_ids": sorted(pool_ids), "held_ids": sorted(held_ids)},
              open(os.path.join(run, "manifest.json"), "w"), indent=2)

    # --- base reference eval (no adapter) ---
    print("=== BASE held-out eval ===", flush=True)
    bdir = os.path.join(run, "base"); os.makedirs(bdir, exist_ok=True)
    bg = gen(args.base, None, held, os.path.join(bdir, "eval_raw.jsonl"), 1, 0.0, args.max_tokens)
    base_correct = {i: grade(bg[i][0], gold[i]) for i in [r["id"] for r in held]}
    base_acc = sum(base_correct.values()) / len(held)
    print(f"BASE held-out acc: {base_acc:.1%}", flush=True)

    accum = defaultdict(list)     # id -> list of kept trace texts
    solved_ever = set()           # train ids solved in any prior round (train-side frontier)
    metrics_path = os.path.join(run, "metrics.csv")
    mf = open(metrics_path, "w", newline="")
    writer = csv.writer(mf)
    writer.writerow(["round", "heldout_acc", "d_vs_base_pp", "n_harvested", "cum_pool",
                     "heldout_new_vs_base", "train_new_solved", "box_rate",
                     "mcnemar_p_vs_base", "t_gen", "t_grade", "t_train", "t_eval"])
    writer.writerow([-1, f"{base_acc:.4f}", 0, 0, 0, 0, 0,
                     sum("\\boxed" in bg[i][0] for i in base_correct) / len(held),
                     1.0, 0, 0, 0, 0])
    flip = {}

    prev_adapter = None
    for t in range(args.rounds):
        rd = os.path.join(run, f"round{t}"); os.makedirs(rd, exist_ok=True)
        # 1) harvest (base at t=0, else previous adapter)
        t0 = time.time()
        att = gen(args.base, prev_adapter, pool, os.path.join(rd, "attempts.jsonl"),
                  args.k, args.temp, args.max_tokens)
        t_gen = time.time() - t0
        # 2) grade + keep, accumulate (dedupe, cap), track frontier
        t0 = time.time()
        n_harvested = 0; new_solved = 0
        for r in pool:
            i = r["id"]
            kept = [tx for tx in att.get(i, []) if keep_trace(tx, gold[i])]
            for tx in kept:
                if tx not in accum[i] and len(accum[i]) < args.cap:
                    accum[i].append(tx); n_harvested += 1
            if kept and i not in solved_ever:
                solved_ever.add(i); new_solved += 1
        t_grade = time.time() - t0
        # write cumulative pool + training data
        pool_path = os.path.join(rd, "correct_pool.jsonl")
        train_path = os.path.join(rd, "train_data.jsonl")
        prompt_of = {r["id"]: r["prompt"] for r in pool}
        with open(pool_path, "w") as fp, open(train_path, "w") as ft:
            for i, traces in accum.items():
                for tx in traces:
                    fp.write(json.dumps({"id": i, "trace": tx}) + "\n")
                    ft.write(json.dumps({"prompt": prompt_of[i], "completion": tx}) + "\n")
        cum_pool = sum(len(v) for v in accum.values())
        # 3) train adapter_t FROM BASE on cumulative pool
        adapter = os.path.join(rd, "adapter")
        t0 = time.time()
        ensure_gpu_free(8)                      # harvest-gen vLLM group already reaped
        run_isolated([PY, os.path.join(ROOT, "scripts", "lora_train.py"),
                      "--base", args.base, "--data", train_path, "--out", adapter,
                      "--epochs", str(args.epochs), "--seed", str(args.seed)])
        t_train = time.time() - t0
        # 4) eval adapter_t on held-out (greedy)
        t0 = time.time()
        eg = gen(args.base, adapter, held, os.path.join(rd, "eval_raw.jsonl"), 1, 0.0, args.max_tokens)
        cor = {i: grade(eg[i][0], gold[i]) for i in [r["id"] for r in held]}
        box_rate = sum("\\boxed" in eg[i][0] for i in cor) / len(held)
        t_eval = time.time() - t0
        acc = sum(cor.values()) / len(held)
        new_vs_base = sum(1 for i in cor if cor[i] and not base_correct[i])
        b = sum(1 for i in cor if base_correct[i] and not cor[i])   # pass->fail vs base
        c = sum(1 for i in cor if not base_correct[i] and cor[i])   # fail->pass vs base
        p = mcnemar_p(b, c)
        flip[f"round{t}"] = {"fail_to_pass": c, "pass_to_fail": b, "mcnemar_p": p}
        with open(os.path.join(rd, "eval.jsonl"), "w") as f:
            for i in cor:
                f.write(json.dumps({"id": i, "correct": cor[i], "text": eg[i][0]}) + "\n")
        json.dump({"t_gen": t_gen, "t_grade": t_grade, "t_train": t_train, "t_eval": t_eval},
                  open(os.path.join(rd, "timing.json"), "w"))
        writer.writerow([t, f"{acc:.4f}", f"{(acc-base_acc)*100:.1f}", n_harvested, cum_pool,
                         new_vs_base, new_solved, f"{box_rate:.3f}", f"{p:.4f}",
                         f"{t_gen:.1f}", f"{t_grade:.1f}", f"{t_train:.1f}", f"{t_eval:.1f}"])
        mf.flush()
        print(f"[round {t}] held-out {acc:.1%} ({(acc-base_acc)*100:+.1f}pp vs base) | "
              f"frontier+{new_vs_base} | harvested {n_harvested} | pool {cum_pool} | "
              f"train-new {new_solved} | box {box_rate:.0%} | McNemar p={p:.3f}", flush=True)
        prev_adapter = adapter

    mf.close()
    json.dump(flip, open(os.path.join(run, "flip_matrix.json"), "w"), indent=2)
    print(f"\nSaved run -> {run}\n  metrics.csv, flip_matrix.json, per-round dirs", flush=True)


if __name__ == "__main__":
    main()
