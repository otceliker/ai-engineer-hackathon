#!/usr/bin/env python3
"""Online lessons-memory prompt-patching loop with a held-out gate.

The core experiment (docs/WIKI.md §10):
  1. A small agent streams `stream_train` tasks in mini-batches and is scored.
  2. A proposer reads the batch's FAILURES and proposes edits to a capped lessons
     playbook (the patch).
  3. GATE: the challenger playbook is accepted over the current champion only if it
     wins on a fresh `patch_dev` subsample by McNemar (p < threshold) AND is net
     positive. `patch_dev` is resampled each round to blunt overfitting.
  4. `stream_test` is strictly held out — measured only for the read-only accuracy
     curve (every N accepts), never for acceptance.

Headline claim: final champion vs the static base prompt on `stream_test`, paired McNemar.

Per-round records stream to a JSONL log (tail it for a live TUI). Example:
    VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$PWD/.venv/bin:$PATH" \
      python scripts/stream_loop.py --model models/Qwen__Qwen2.5-1.5B-Instruct \
        --proposer local --batch-size 25 --gate-size 120
"""
import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bfcl_eval import BfclAgent, render_system_prompt, parse_tool_calls, BASE_SYSTEM_PROMPT  # noqa: E402
from scripts.lessons import LessonBook  # noqa: E402
from scripts.proposer import make_proposer  # noqa: E402

DEFAULT_SPLITS = os.path.join(ROOT, "data", "bfcl", "splits.json")


def mcnemar(b, c):
    """Two-sided exact-binomial McNemar p-value (same as scripts/star_compare.py)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n)


def discordant(champ_ids, chal_ids):
    """Paired discordant counts over the same ids.
    b = challenger fixes (champ wrong, chal right); c = challenger breaks (champ right, chal wrong)."""
    b = sum(1 for i in champ_ids if not champ_ids[i] and chal_ids.get(i))
    c = sum(1 for i in champ_ids if champ_ids[i] and not chal_ids.get(i))
    return b, c


def build_failures(records, result):
    """From an evaluate() result, build (record, raw_output, parsed_calls) for each miss."""
    failures = []
    for r in records:
        if not result["per_id"][r["id"]]:
            raw = result["raw"][r["id"]]
            failures.append((r, raw, parse_tool_calls(raw)))
    return failures


def log_line(fh, obj):
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--proposer", choices=["local", "claude", "do"], default="local",
                    help="local=1.5B self-improve; claude=Anthropic SDK; do=OpenAI-compatible (DigitalOcean GenAI)")
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--gate-size", type=int, default=120, help="patch_dev subsample size per round")
    ap.add_argument("--mcnemar-p", type=float, default=0.1, help="accept iff McNemar p < this AND net positive")
    ap.add_argument("--no-gate", action="store_true",
                    help="HEADROOM PROBE: accept every applied edit (no gate). Measures the upper "
                         "bound of what lessons can do per-category, free of gate selection.")
    ap.add_argument("--max-rounds", type=int, default=0, help="cap rounds (0 = full stream_train)")
    ap.add_argument("--eval-every", type=int, default=5, help="measure stream_test every N accepted patches")
    ap.add_argument("--max-bullets", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    # generation
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    # outputs
    ap.add_argument("--out", default=None, help="round-log JSONL (default results/stream_loop_<proposer>.jsonl)")
    ap.add_argument("--lessons-out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.splits_file) as f:
        payload = json.load(f)
    splits = payload["splits"]
    train, dev, test = splits["stream_train"], splits["patch_dev"], splits["stream_test"]

    model_name = os.path.basename(args.model.rstrip("/"))
    out_path = args.out or os.path.join(ROOT, "results", f"stream_loop_{model_name}_{args.proposer}.jsonl")
    lessons_path = args.lessons_out or os.path.join(ROOT, "results", f"lessons_{model_name}_{args.proposer}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading agent ({model_name}) + {args.proposer} proposer ...", flush=True)
    agent = BfclAgent(args.model, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                      temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    proposer = make_proposer(args.proposer, agent=agent if args.proposer == "local" else None)

    champion = LessonBook(max_bullets=args.max_bullets, max_chars=args.max_chars)

    fh = open(out_path, "w")
    t_start = time.time()

    # --- baseline on strictly held-out stream_test (the number to beat) ---
    print("Measuring static-prompt baseline on stream_test ...", flush=True)
    base_res = agent.evaluate(test, render_system_prompt(None))
    baseline_per_id = dict(base_res["per_id"])
    print(f"  baseline stream_test = {base_res['accuracy']:.1%} ({base_res['n_correct']}/{base_res['n']})", flush=True)
    log_line(fh, {"event": "baseline", "split": "stream_test", "accuracy": base_res["accuracy"],
                  "n": base_res["n"], "n_correct": base_res["n_correct"],
                  "by_category": base_res["by_category"]})

    # --- stream loop ---
    n_batches = (len(train) + args.batch_size - 1) // args.batch_size
    if args.max_rounds:
        n_batches = min(n_batches, args.max_rounds)
    accepts = 0
    for rnd in range(n_batches):
        batch = train[rnd * args.batch_size:(rnd + 1) * args.batch_size]
        if not batch:
            break
        champ_sys = render_system_prompt(champion.texts())
        batch_res = agent.evaluate(batch, champ_sys)
        failures = build_failures(batch, batch_res)

        rec = {"event": "round", "round": rnd, "batch_n": len(batch),
               "batch_acc": batch_res["accuracy"], "n_failures": len(failures),
               "champion_size": len(champion.lessons), "champion_chars": champion.total_chars(),
               "accepted": False}

        if failures:
            edits = proposer.propose(failures, champion)
            rec["n_proposed"] = len(edits)
            if edits:
                challenger = champion.copy()
                applied, rejected = challenger.apply(edits)
                rec["n_applied"] = len(applied)
                rec["rejected"] = [r[1] for r in rejected]
                if applied and args.no_gate:
                    # HEADROOM PROBE: accept unconditionally, no gate eval.
                    champion = challenger
                    accepts += 1
                    rec.update({"accepted": True, "no_gate": True, "edits": edits,
                                "champion_lessons": champion.texts()})
                elif applied:
                    # GATE: fresh patch_dev subsample, paired champion-vs-challenger
                    sample = dev if len(dev) <= args.gate_size else rng.sample(dev, args.gate_size)
                    champ_dev = agent.evaluate(sample, champ_sys)
                    chal_dev = agent.evaluate(sample, render_system_prompt(challenger.texts()))
                    b, c = discordant(champ_dev["per_id"], chal_dev["per_id"])
                    p = mcnemar(b, c)
                    accept = (b > c) and (p < args.mcnemar_p)
                    rec.update({"gate_n": len(sample), "dev_champ_acc": champ_dev["accuracy"],
                                "dev_chal_acc": chal_dev["accuracy"], "fixes_b": b, "breaks_c": c,
                                "mcnemar_p": round(p, 4), "accepted": accept,
                                "edits": edits})
                    if accept:
                        champion = challenger
                        accepts += 1
                        rec["champion_lessons"] = champion.texts()

        # read-only stream_test curve every N accepts
        if rec["accepted"] and accepts % args.eval_every == 0:
            test_res = agent.evaluate(test, render_system_prompt(champion.texts()))
            rec["stream_test_acc"] = test_res["accuracy"]

        log_line(fh, rec)
        tag = "ACCEPT" if rec["accepted"] else ("noedit" if not failures or not rec.get("n_applied") else "reject")
        extra = ""
        if "mcnemar_p" in rec:
            extra = f" | dev {rec['dev_champ_acc']:.0%}->{rec['dev_chal_acc']:.0%} b={rec['fixes_b']} c={rec['breaks_c']} p={rec['mcnemar_p']}"
        st = f" | stream_test={rec['stream_test_acc']:.1%}" if "stream_test_acc" in rec else ""
        print(f"[r{rnd:03d}] batch_acc={rec['batch_acc']:.0%} fails={rec['n_failures']:2d} "
              f"{tag} lessons={len(champion.lessons)}{extra}{st}", flush=True)

    # --- final: champion vs baseline on full stream_test (the headline claim) ---
    print("\nFinal champion eval on stream_test ...", flush=True)
    final_res = agent.evaluate(test, render_system_prompt(champion.texts()))
    b, c = discordant(baseline_per_id, final_res["per_id"])
    p = mcnemar(b, c)
    delta = final_res["accuracy"] - base_res["accuracy"]
    summary = {"event": "final", "accepts": accepts, "rounds": n_batches, "no_gate": args.no_gate,
               "baseline_acc": base_res["accuracy"], "final_acc": final_res["accuracy"],
               "delta": delta, "fixes_b": b, "breaks_c": c, "mcnemar_p": round(p, 4),
               "champion_lessons": champion.texts(),
               "baseline_by_category": base_res["by_category"],
               "by_category": final_res["by_category"],
               "wall_s": round(time.time() - t_start, 1)}
    log_line(fh, summary)
    fh.close()
    champion.save(lessons_path)

    print(f"\n=== {model_name} | proposer={args.proposer} ===")
    print(f"Baseline  stream_test : {base_res['accuracy']:.1%} ({base_res['n_correct']}/{base_res['n']})")
    print(f"Patched   stream_test : {final_res['accuracy']:.1%} ({final_res['n_correct']}/{final_res['n']})")
    print(f"Delta                 : {delta:+.1%}  (fixes b={b}, breaks c={c}, McNemar p={p:.4f})")
    print(f"Accepts/rounds        : {accepts}/{n_batches}    final lessons={len(champion.lessons)}"
          + ("   [NO-GATE HEADROOM PROBE]" if args.no_gate else ""))
    print("\nPer-category (baseline -> patched):")
    bc, fc = base_res["by_category"], final_res["by_category"]
    for cat in sorted(fc):
        b0, n0 = bc[cat]; b1, n1 = fc[cat]
        d = b1 / n1 - b0 / n0
        print(f"  {cat:24s}: {b0/n0:5.1%} -> {b1/n1:5.1%}  ({d:+.1%})")
    print(f"\nChampion playbook ({len(champion.lessons)} lessons):")
    for l in champion.texts():
        print(f"  - {l}")
    print(f"\nRound log -> {out_path}\nLessons   -> {lessons_path}")


if __name__ == "__main__":
    main()
