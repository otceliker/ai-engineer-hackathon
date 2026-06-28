#!/usr/bin/env python3
"""Prepare BBH: sample per-task, seeded train/test split -> data/bbh/bbh.json.

Run with a venv that has `datasets` (.venv-ifeval). bbh_rsi.py then reads the json (no datasets dep).
"""
import argparse
import json
import os
import random

from datasets import get_dataset_config_names, load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "bbh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lukaemon/bbh")
    ap.add_argument("--per-task-train", type=int, default=25)
    ap.add_argument("--per-task-test", type=int, default=25)
    ap.add_argument("--limit-tasks", type=int, default=0, help="0=all (smoke: e.g. 3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cfgs = sorted(get_dataset_config_names(args.dataset))
    if args.limit_tasks:
        cfgs = cfgs[:args.limit_tasks]
    rng = random.Random(args.seed)
    rows, train_keys, test_keys = [], [], []
    n = args.per_task_train + args.per_task_test
    for task in cfgs:
        d = list(load_dataset(args.dataset, task, split="test"))
        rng.shuffle(d)
        chosen = d[:n]
        for i, ex in enumerate(chosen):
            key = f"{task}__{i}"
            rows.append({"key": key, "task": task, "input": ex["input"], "target": ex["target"]})
            (train_keys if i < args.per_task_train else test_keys).append(key)
    json.dump({"rows": rows, "train_keys": train_keys, "test_keys": test_keys},
              open(os.path.join(OUT, "bbh.json"), "w"))
    print(f"{len(cfgs)} tasks | {len(rows)} rows | {len(train_keys)} train / {len(test_keys)} test "
          f"-> {os.path.join(OUT, 'bbh.json')}")


if __name__ == "__main__":
    main()
