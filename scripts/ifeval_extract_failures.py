#!/usr/bin/env python3
"""Extract TRAIN-split failures from a scored IFEval base run, for the proposer.

For each train prompt the base model failed, records the prompt, the model's response, and
the natural-language descriptions of the specific constraints it VIOLATED (from the official
checker's per-instruction follow list). This structured failure signal is what the strong
proposer reads to write lessons. (IFEval has no gold answers — the constraint spec IS the gold.)
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "ifeval_vendor"))
import instructions_registry as R  # noqa: E402

IFE = os.path.join(ROOT, "data", "ifeval")


def describe(iid, kwargs, prompt):
    """Return the natural-language description of one instruction."""
    inst = R.INSTRUCTION_DICT[iid](iid)
    kw = {k: v for k, v in kwargs.items() if v is not None}
    desc = inst.build_description(**kw)
    args = inst.get_instruction_args()
    if args and "prompt" in args:
        desc = inst.build_description(prompt=prompt)
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-dir", default=os.path.join(ROOT, "results", "ifeval", "score_base"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=os.path.join(IFE, "train_failures.json"))
    args = ap.parse_args()

    keys = set(json.load(open(os.path.join(IFE, f"{args.split}_keys.json"))))
    inrows = {r["key"]: r for r in (json.loads(l) for l in open(os.path.join(IFE, "input_data.jsonl")))}
    p2k = {r["prompt"]: k for k, r in inrows.items()}

    strict = os.path.join(args.score_dir, "eval_results_strict.jsonl")
    failures, hist = [], collections.Counter()
    n_total = n_fail = 0
    for l in open(strict):
        r = json.loads(l)
        k = p2k.get(r["prompt"])
        if k is None or k not in keys:
            continue
        n_total += 1
        if r["follow_all_instructions"]:
            continue
        n_fail += 1
        row = inrows[k]
        failed = []
        for iid, kw, ok in zip(row["instruction_id_list"], row["kwargs"], r["follow_instruction_list"]):
            if not ok:
                hist[iid] += 1
                try:
                    failed.append({"id": iid, "desc": describe(iid, kw, row["prompt"])})
                except Exception as e:
                    failed.append({"id": iid, "desc": f"(could not render: {e})"})
        failures.append({
            "key": k,
            "prompt": row["prompt"],
            "response": r["response"],
            "failed": failed,
        })

    out = {"split": args.split, "n_train": n_total, "n_failures": n_fail,
           "failed_instruction_histogram": dict(hist.most_common()), "failures": failures}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"{args.split}: {n_fail}/{n_total} prompts failed. Top failed constraint types:")
    for iid, c in hist.most_common(12):
        print(f"  {c:4d}  {iid}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
