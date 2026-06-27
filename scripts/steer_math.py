#!/usr/bin/env python3
"""Steering-vector capability GATE: can a difference-of-means activation vector,
added to the residual stream at inference, raise MATH accuracy on one skill category?

Implements the agreed design:
  #1 paired McNemar / flip-matrix instead of an N=60 +5pp threshold
  #2 shuffled-label control (the real null) + random-vector control
  #4 inject hook gated to decode steps only (no prompt steering)
  #6 non-reasoning model (Qwen2.5-1.5B-Instruct), problems from MATH *train* split
  #5 alpha grid extended downward (effect likely < coherence collapse)
  #7 \boxed{} emission rate logged separately from accuracy

Single coarse pass: real vector over (layers x alphas) on held-out, then controls
at the single best positive config. No two-stage confirmation unless a candidate appears.

Usage:
  python scripts/steer_math.py --model models/Qwen__Qwen2.5-1.5B-Instruct
"""
import argparse, json, math, os, random, re, sys
from collections import defaultdict

try:
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from math_verify import parse, verify
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PARQUET = os.path.join(ROOT, "data", "nlile__hendrycks-MATH-benchmark", "data",
                             "train-00000-of-00001.parquet")
TEST_JSONL = os.path.join(ROOT, "results", "Qwen__Qwen2.5-1.5B-Instruct.jsonl")

SYSTEM = "You are a math expert. Solve the problem step by step and put your final answer within \\boxed{}."
FEWSHOT = [
    ("What is $2 + 3 \\times 4$?",
     "Multiplication before addition: $3 \\times 4 = 12$, then $2 + 12 = 14$. The answer is $\\boxed{14}$."),
    ("Simplify the fraction $\\frac{6}{8}$.",
     "The greatest common divisor of 6 and 8 is 2, so $\\frac{6}{8} = \\frac{3}{4}$. The answer is $\\boxed{\\frac{3}{4}}$."),
]
ALPHAS = [-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]


# ----- data -----------------------------------------------------------------
def pick_category(override):
    if override:
        return override
    acc = defaultdict(lambda: [0, 0])
    for line in open(TEST_JSONL):
        r = json.loads(line); a = acc[r["subject"]]; a[0] += r["correct"]; a[1] += 1
    band = {s: c / n for s, (c, n) in acc.items() if 0.40 <= c / n <= 0.60}
    # closest to 0.5 within the Goldilocks band
    return min(band, key=lambda s: abs(band[s] - 0.5))


def load_split(category, n_train, n_held, seed):
    rows = [r for r in pq.read_table(TRAIN_PARQUET).to_pylist() if r["subject"] == category]
    random.Random(seed).shuffle(rows)
    return rows[:n_train], rows[n_train:n_train + n_held]


# ----- prompting / grading --------------------------------------------------
def build_prompt(tok, problem):
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": problem})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def is_correct(text, gold):
    try:
        return bool(verify(parse(gold), parse(text)))
    except Exception:
        return False


def has_box(text):
    return "\\boxed" in text


def degenerate(text):
    toks = text.split()
    tris = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    if len(tris) < 8:
        return False
    return (1 - len(set(tris)) / len(tris)) > 0.5


@torch.no_grad()
def generate(model, tok, prompts, max_new=512, bs=60):
    out = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            new = gen[j, enc["input_ids"].shape[1]:]
            out.append(tok.decode(new, skip_special_tokens=True))
    return out


# ----- vector construction --------------------------------------------------
@torch.no_grad()
def capture(model, tok, prompts, gens, layers):
    """Per-example: teacher-force prompt+generation, mean-pool residual over
    generated positions for each candidate layer. Returns acts[L] = [vec per ex]
    and the mean residual norm per layer."""
    acts = {L: [] for L in layers}
    norms = {L: [] for L in layers}
    for prompt, gen in zip(prompts, gens):
        pids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        gids = tok(gen, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        if gids.shape[1] == 0:
            for L in layers:
                acts[L].append(None)
            continue
        full = torch.cat([pids, gids], dim=1)
        hs = model(full, output_hidden_states=True).hidden_states  # tuple len n_layer+1
        p = pids.shape[1]
        for L in layers:
            h = hs[L + 1][0, p:, :].float()            # output of layer L, generated positions
            acts[L].append(h.mean(0).cpu())
            norms[L].append(h.norm(dim=-1).mean().item())
    mean_norm = {L: sum(norms[L]) / max(1, len(norms[L])) for L in layers}
    return acts, mean_norm


def diff_of_means(acts_L, labels):
    cor = [a for a, y in zip(acts_L, labels) if a is not None and y]
    inc = [a for a, y in zip(acts_L, labels) if a is not None and not y]
    if not cor or not inc:
        return None
    v = torch.stack(cor).mean(0) - torch.stack(inc).mean(0)
    return v / v.norm()


# ----- steering hook --------------------------------------------------------
def make_hook(vec, fired):
    def hook(mod, inp, out):
        is_tuple = isinstance(out, tuple)              # transformers 5.x layers return a bare tensor
        hs = out[0] if is_tuple else out
        if hs.shape[1] == 1:                            # #4: decode steps only, never prefill/prompt
            hs = hs + vec.to(hs.dtype).to(hs.device)
            fired[0] += 1
            return (hs,) + tuple(out[1:]) if is_tuple else hs
        return out
    return hook


def run_config(model, tok, layer, vec, prompts, golds):
    fired = [0]
    h = model.model.layers[layer].register_forward_hook(make_hook(vec, fired))
    try:
        gens = generate(model, tok, prompts)
    finally:
        h.remove()
    if float(vec.norm()) > 0 and fired[0] == 0:
        print("  [WARN] inject hook never fired — output shape mismatch", flush=True)
    correct = [is_correct(g, gd) for g, gd in zip(gens, golds)]
    boxed = sum(has_box(g) for g in gens) / len(gens)
    degen = sum(degenerate(g) for g in gens) / len(gens)
    return correct, boxed, degen


# ----- stats ----------------------------------------------------------------
def mcnemar_p(b, c):
    """exact two-sided McNemar on discordant counts b (pass->fail), c (fail->pass)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, p)


def flips(base, steered):
    b = sum(1 for x, y in zip(base, steered) if x and not y)   # pass -> fail
    c = sum(1 for x, y in zip(base, steered) if not x and y)   # fail -> pass
    return b, c


# ----- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--category", default=None, help="override Goldilocks auto-pick")
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-held", type=int, default=60)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="default: 3 mid-stack")
    ap.add_argument("--control-seeds", type=int, default=5)
    ap.add_argument("--win-pp", type=float, default=10.0, help="WIN bar in percentage points")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    category = pick_category(args.category)
    train, held = load_split(category, args.n_train, args.n_held, args.seed)
    print(f"Category: {category} | train={len(train)} held-out={len(held)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    n_layer = model.config.num_hidden_layers
    layers = args.layers or sorted({int(n_layer * f) for f in (0.45, 0.6, 0.7)})
    hidden = model.config.hidden_size
    print(f"Model: {n_layer} layers, hidden={hidden} | steering layers={layers}", flush=True)

    # 1) label train attempts, build vectors
    tprompts = [build_prompt(tok, r["problem"]) for r in train]
    tgens = generate(model, tok, tprompts)
    tlabels = [is_correct(g, r["answer"]) for g, r in zip(tgens, train)]
    print(f"Train cold acc: {sum(tlabels)}/{len(tlabels)} = {sum(tlabels)/len(tlabels):.1%}", flush=True)
    acts, mean_norm = capture(model, tok, tprompts, tgens, layers)
    real = {L: diff_of_means(acts[L], tlabels) for L in layers}

    # 2) baseline on held-out
    hprompts = [build_prompt(tok, r["problem"]) for r in held]
    hgolds = [r["answer"] for r in held]
    base_correct, base_boxed, base_degen = run_config(model, tok, layers[0],
                                                       torch.zeros(hidden), hprompts, hgolds)
    base_acc = sum(base_correct) / len(base_correct)
    print(f"\nBaseline held-out acc: {base_acc:.1%} (boxed {base_boxed:.0%})", flush=True)

    # 3) real-vector grid
    print("\n=== GRID (real difference-of-means vector) ===")
    print(f"{'layer':>5} {'alpha':>6} {'acc':>6} {'dAcc':>6} {'f>p':>4} {'p>f':>4} "
          f"{'McN_p':>7} {'boxed':>6} {'coh':>4}")
    rows = []
    for L in layers:
        if real[L] is None:
            print(f"  layer {L}: skipped (no correct/incorrect split)")
            continue
        for a in ALPHAS:
            vec = (a * mean_norm[L]) * real[L]
            cor, boxed, degen = run_config(model, tok, L, vec, hprompts, hgolds)
            acc = sum(cor) / len(cor)
            b, c = flips(base_correct, cor)
            p = mcnemar_p(b, c)
            coh = "ok" if degen < 0.1 else "BAD"
            rows.append(dict(layer=L, alpha=a, acc=acc, dpp=(acc - base_acc) * 100,
                             fp=c, pf=b, mcnemar=p, boxed=boxed, coh_ok=degen < 0.1))
            print(f"{L:>5} {a:>6.2f} {acc:>6.1%} {(acc-base_acc)*100:>+6.1f} {c:>4} {b:>4} "
                  f"{p:>7.3f} {boxed:>6.0%} {coh:>4}", flush=True)

    # 4) best positive config -> controls (winner's-curse-matched: each null picks
    #    its OWN best alpha at the same layer, so we compare real-max vs null-max)
    pos = [r for r in rows if r["alpha"] > 0 and r["coh_ok"]]
    best = max(pos, key=lambda r: r["dpp"]) if pos else None
    posalphas = [a for a in ALPHAS if a > 0]
    control = {"shuffled": [], "random": []}   # each entry = that null's best dAcc over the alpha sweep
    if best:
        L = best["layer"]
        print(f"\n=== CONTROLS (alpha-swept at best layer {L}; each null reports its own best) ===")

        def null_best(vec_unit):
            sweep = []
            for a in posalphas:
                cor, _, dg = run_config(model, tok, L, (a * mean_norm[L]) * vec_unit, hprompts, hgolds)
                if dg < 0.1:                      # only coherent configs can "win"
                    sweep.append((sum(cor) / len(cor) - base_acc) * 100)
            return max(sweep) if sweep else 0.0

        for k in range(args.control_seeds):
            lab = tlabels[:]; random.Random(1000 + k).shuffle(lab)   # shuffled-label null (#2)
            sv = diff_of_means(acts[L], lab)
            if sv is not None:
                control["shuffled"].append(null_best(sv))
            g = torch.Generator().manual_seed(2000 + k)              # random unit vector, matched norm
            rv = torch.randn(hidden, generator=g); rv = rv / rv.norm()
            control["random"].append(null_best(rv))
        for name, vals in control.items():
            if vals:
                print(f"  {name:9s} best-dAcc per seed: " + ", ".join(f"{v:+.1f}" for v in vals) +
                      f"  (null max {max(vals):+.1f}pp)")

    # 5) verdict (real-max must clear the matched null-max, the bar, AND McNemar)
    ctrl_all = control["shuffled"] + control["random"]
    ctrl_max = max(ctrl_all) if ctrl_all else 0.0
    verdict = "FLAT"
    if best:
        beats_ctrl = best["dpp"] > ctrl_max
        sig = best["mcnemar"] < 0.05
        big = best["dpp"] >= args.win_pp
        any_drop = all(r["dpp"] <= 0 for r in rows if r["alpha"] > 0)
        if big and sig and beats_ctrl and best["coh_ok"]:
            verdict = "WIN"
        elif any_drop:
            verdict = "DEGRADE"
        elif ctrl_max >= best["dpp"]:
            verdict = "FLAT"  # real indistinguishable from the winner's-curse-matched null

    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    if best:
        print(f"Best positive config: layer {best['layer']}, alpha {best['alpha']} | "
              f"dAcc {best['dpp']:+.1f}pp (fail>pass {best['fp']} vs pass>fail {best['pf']}, "
              f"McNemar p={best['mcnemar']:.3f}) | beats-controls={best['dpp']>ctrl_max} "
              f"(control max {ctrl_max:+.1f}pp)")
    if verdict == "FLAT":
        print("NOTE: FLAT here means 'no effect detectable at N=60' (power-limited), not "
              "'no effect exists' — McNemar needs ~a 9:2 fail>pass split to clear p<0.05, so a "
              "true small effect (~+6pp) would read FLAT. Permitted single follow-up if flat: "
              "try one earlier layer (~30% depth); then stop.")
    print("CAVEAT: even a WIN may reflect a 'be-more-careful/easy-problem' direction, "
          "not a pure math-skill primitive; the shuffled-label control is what licenses any causal read.")
    rec = {"WIN": "Activations are a viable skill primitive — lean in.",
           "FLAT": "Steering can't carry capability here — keep LoRAs as the workhorse; activations diagnostic-only.",
           "DEGRADE": "Direction entangled with general capability — abandon as a primitive."}[verdict]
    print(f"RECOMMENDATION: {rec}")

    out = os.path.join(ROOT, "results", f"steer_{category.replace(' ', '_')}.json")
    json.dump({"category": category, "baseline_acc": base_acc, "grid": rows,
               "controls": control, "verdict": verdict, "best": best,
               "mean_norm": mean_norm}, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
