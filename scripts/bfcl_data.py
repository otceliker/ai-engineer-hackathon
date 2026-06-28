#!/usr/bin/env python3
"""Build seeded, category-stratified BFCL V4 single-turn splits.

Splits the single-turn AST categories into three disjoint sets:
  stream_train (60%)  — the live task stream the agent sees; failures drive patches
  patch_dev    (20%)  — held-out gate; a patch is accepted only if it wins here
  stream_test  (20%)  — strictly held out; touched ONLY for the honest accuracy curve

The split is deterministic (seeded) and written self-contained to
`data/bfcl/splits.json` so the Mac and neptune operate on identical data without
re-deriving anything. Each record carries everything the runner/checker need:
the question, the tool specs, the ground truth (None for irrelevance/relevance),
and the `kind`/`test_category` that select the scoring path.

Categories (single-turn, Python AST only — no execution sandbox):
  ast         simple / multiple / parallel / parallel_multiple (+ their live_* variants)
  irrelevance no function should be called  -> correct iff model emits zero calls
  relevance   a relevant function exists    -> correct iff model emits >=1 parseable call

Usage:
    python scripts/bfcl_data.py                       # cap 200/category, seed 0
    python scripts/bfcl_data.py --max-per-category 0  # use all examples
    python scripts/bfcl_data.py --seed 7 --out data/bfcl/splits_seed7.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFCL_DIR = os.path.join(ROOT, "data", "bfcl")

# (filename stem, category label, checker test_category, kind)
# `category` is used for stratification + reporting; `test_category` selects the
# AST checker branch ("parallel"/"multiple" substring matching, per BFCL).
CATEGORIES = [
    ("BFCL_v4_simple_python.json",            "simple_python",          "simple",            "ast"),
    ("BFCL_v4_multiple.json",                 "multiple",               "multiple",          "ast"),
    ("BFCL_v4_parallel.json",                 "parallel",               "parallel",          "ast"),
    ("BFCL_v4_parallel_multiple.json",        "parallel_multiple",      "parallel_multiple", "ast"),
    ("BFCL_v4_live_simple.json",              "live_simple",            "simple",            "ast"),
    ("BFCL_v4_live_multiple.json",            "live_multiple",          "multiple",          "ast"),
    ("BFCL_v4_live_parallel.json",            "live_parallel",          "parallel",          "ast"),
    ("BFCL_v4_live_parallel_multiple.json",   "live_parallel_multiple", "parallel_multiple", "ast"),
    ("BFCL_v4_irrelevance.json",              "irrelevance",            "irrelevance",       "irrelevance"),
    ("BFCL_v4_live_irrelevance.json",         "live_irrelevance",       "irrelevance",       "irrelevance"),
    ("BFCL_v4_live_relevance.json",           "live_relevance",         "relevance",         "relevance"),
]

SPLIT_RATIOS = (("stream_train", 0.6), ("patch_dev", 0.2), ("stream_test", 0.2))


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_category(filename, category, test_category, kind, cap, rng):
    """Load one category file, merge ground truth, optionally subsample to `cap`."""
    path = os.path.join(BFCL_DIR, filename)
    if not os.path.exists(path):
        sys.exit(f"Missing data file: {path}\nRun the vendoring step in scripts/bfcl_data setup.")
    rows = _load_jsonl(path)

    gt_by_id = {}
    if kind == "ast":
        ans_path = os.path.join(BFCL_DIR, "possible_answer", filename)
        if not os.path.exists(ans_path):
            sys.exit(f"Missing possible_answer file for AST category: {ans_path}")
        for a in _load_jsonl(ans_path):
            gt_by_id[a["id"]] = a["ground_truth"]

    records = []
    for r in rows:
        rid = r["id"]
        records.append({
            "id": rid,
            "category": category,
            "test_category": test_category,
            "kind": kind,
            "question": r["question"],
            "function": r.get("function", []),
            "ground_truth": gt_by_id.get(rid) if kind == "ast" else None,
        })

    if kind == "ast":
        missing = [r["id"] for r in records if r["ground_truth"] is None]
        if missing:
            sys.exit(f"{category}: {len(missing)} records missing ground truth, e.g. {missing[:3]}")

    # Deterministic subsample (cap<=0 means keep all). Sort first so order is
    # independent of file iteration, then shuffle with the seeded rng.
    records.sort(key=lambda r: r["id"])
    rng.shuffle(records)
    if cap and cap > 0:
        records = records[:cap]
    return records


def stratified_split(records, rng):
    """60/20/20 within each category, so every split has the same category mix."""
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    out = {name: [] for name, _ in SPLIT_RATIOS}
    for cat in sorted(by_cat):
        items = by_cat[cat]
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * SPLIT_RATIOS[0][1]))
        n_dev = int(round(n * SPLIT_RATIOS[1][1]))
        # remainder to test so the three always sum to n
        out["stream_train"].extend(items[:n_train])
        out["patch_dev"].extend(items[n_train:n_train + n_dev])
        out["stream_test"].extend(items[n_train + n_dev:])
    # Shuffle each split so the stream interleaves categories (stream realism).
    for name in out:
        rng.shuffle(out[name])
    return out


def summarize(splits):
    cats = sorted({r["category"] for s in splits.values() for r in s})
    names = [n for n, _ in SPLIT_RATIOS]
    width = max(len(c) for c in cats) + 2
    header = "category".ljust(width) + "".join(n.rjust(14) for n in names) + "total".rjust(10)
    print(header)
    print("-" * len(header))
    for c in cats:
        row = c.ljust(width)
        tot = 0
        for n in names:
            k = sum(1 for r in splits[n] if r["category"] == c)
            tot += k
            row += str(k).rjust(14)
        row += str(tot).rjust(10)
        print(row)
    print("-" * len(header))
    total_row = "TOTAL".ljust(width)
    grand = 0
    for n in names:
        k = len(splits[n])
        grand += k
        total_row += str(k).rjust(14)
    total_row += str(grand).rjust(10)
    print(total_row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-category", type=int, default=200,
                    help="Cap examples per source file before splitting (0 = all). "
                         "Keeps the stream category-balanced and eval fast.")
    ap.add_argument("--out", default=os.path.join(BFCL_DIR, "splits.json"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_records = []
    for filename, category, test_category, kind in CATEGORIES:
        recs = load_category(filename, category, test_category, kind, args.max_per_category, rng)
        all_records.extend(recs)

    splits = stratified_split(all_records, rng)

    payload = {
        "meta": {
            "seed": args.seed,
            "max_per_category": args.max_per_category,
            "ratios": dict(SPLIT_RATIOS),
            "counts": {name: len(recs) for name, recs in splits.items()},
            "source": "bfcl-eval==2026.3.23 (vendored), V4 single-turn AST categories",
        },
        "splits": splits,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"seed={args.seed}  cap={args.max_per_category}  -> {args.out}\n")
    summarize(splits)


if __name__ == "__main__":
    main()
