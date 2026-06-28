#!/usr/bin/env python3
"""Generate site/sample_viz.json matching the schema bbh_rsi_v2.py emits (<out>_viz.json),
so the website can be built before the real run lands. Pure-stdlib (random only)."""
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(0)

TASKS = {
    0: ("dyck_languages", "Nested bracket matching",
        ["Start from the innermost brackets and match each opener to its closer.",
         "Track the open-bracket stack and emit closers in reverse order."]),
    1: ("date_understanding", "Date arithmetic",
        ["Identify the reference date, then add or subtract the stated offset carefully."]),
    2: ("snarks", "Sarcasm detection",
        ["Look for contradiction or exaggeration that signals sarcasm."]),
    3: ("logical_deduction", "Constraint ordering",
        ["Translate each clue into an ordering constraint, then resolve them jointly."]),
}
CENTERS = {0: (-4, 0, 0), 1: (4, 1, -1), 2: (0, 4, 2), 3: (1, -4, 1)}


def jitter(c, s=1.1):
    return [c[i] + rng.gauss(0, s) for i in range(3)]


clusters, train_points = [], []
for k, (task, name, lessons) in TASKS.items():
    c = CENTERS[k]
    n = rng.randint(8, 18)
    clusters.append({"id": k, "name": name, "size": n, "lessons": lessons, "centroid": list(c)})
    for _ in range(n):
        train_points.append({"xyz": jitter(c), "cluster": k, "task": task,
                             "prompt": f"[{task}] example train failure prompt …"})

test_items = []
for i in range(16):
    k = rng.randint(0, 3)
    task, name, lessons = TASKS[k]
    base_ok = rng.random() < 0.55
    # routed tends to fix some base failures, rarely breaks
    routed_ok = base_ok or (rng.random() < 0.45)
    if base_ok and rng.random() < 0.08:
        routed_ok = False
    test_items.append({
        "key": f"{task}__{i}", "task": task, "prompt": f"[{task}] incoming test prompt #{i} …",
        "gold": "(A)", "xyz": jitter(CENTERS[k], 1.4), "routed_cluster": k, "routed_cluster_name": name,
        "routed_lessons": lessons,
        "base_output": "… model reasoning without lessons … So the answer is (B).",
        "base_correct": base_ok,
        "routed_output": "… reasoning guided by the routed lesson … So the answer is (A).",
        "routed_correct": routed_ok,
    })


def arm(items, field):
    a = sum(it[field] for it in items) / len(items)
    return {"acc": round(a, 4), "b": sum(1 for it in items if not it["base_correct"] and it[field]),
            "c": sum(1 for it in items if it["base_correct"] and not it[field]), "p": 0.2}


viz = {
    "model": "Qwen2.5-7B-Instruct (SAMPLE DATA)", "K": len(TASKS),
    "arms": {"base": {"acc": round(sum(it["base_correct"] for it in test_items) / len(test_items), 4),
                      "b": 0, "c": 0, "p": 1.0},
             "global": arm(test_items, "routed_correct"),
             "routed": arm(test_items, "routed_correct")},
    "clusters": clusters, "train_points": train_points, "test_items": test_items,
}
out = os.path.join(ROOT, "site", "sample_viz.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(viz, open(out, "w"), indent=1)
print(f"wrote {out}: {len(train_points)} train pts, {len(test_items)} test items, {len(clusters)} clusters")
