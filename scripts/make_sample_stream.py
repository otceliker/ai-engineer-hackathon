#!/usr/bin/env python3
"""Generate site/sample_stream.json matching bbh_stream.py output, so the streaming webview renders
before the real run lands. Pure-stdlib."""
import json
import math
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = random.Random(1)
N = 300
TASKS = ["dyck_languages", "date_understanding", "snarks", "logical_deduction", "navigate", "boolean_expressions"]

steps = []
cc = cl = 0  # cumulative correct: control, lessoned
nclu = 0
for t in range(N):
    base_p = 0.55
    ctrl_ok = 1 if rng.random() < base_p else 0
    # lessons kick in after a warm-up, margin grows then plateaus
    margin = 0.0 if t < 40 else min(0.16, 0.16 * (1 - math.exp(-(t - 40) / 90.0)))
    les_ok = 1 if rng.random() < min(0.98, base_p + margin) else 0
    if t % 20 == 0 and t > 0 and nclu < 10:
        nclu += 1
    cc += ctrl_ok; cl += les_ok
    steps.append({"t": t, "key": f"{TASKS[t%len(TASKS)]}__{t}", "task": TASKS[t % len(TASKS)],
                  "routed_cluster": (t % max(nclu, 1)) if nclu else -1, "had_lessons": t >= 40,
                  "lessoned_correct": bool(les_ok), "control_correct": bool(ctrl_ok),
                  "cum_lessoned": cl / (t + 1), "cum_control": cc / (t + 1), "n_clusters": nclu})

out = {"model": "Qwen2.5-7B-Instruct (SAMPLE)", "n": N, "batch": 20, "threshold": 0.82,
       "final_lessoned": cl / N, "final_control": cc / N, "steps": steps,
       "clusters": [{"id": i, "size": rng.randint(6, 30),
                     "lessons": [f"Strategy {i}.1 for this failure family.", f"Strategy {i}.2 (gated add)."]}
                    for i in range(nclu)]}
p = os.path.join(ROOT, "site", "sample_stream.json")
json.dump(out, open(p, "w"), indent=1)
print(f"wrote {p}: {N} steps, final lessoned {out['final_lessoned']:.1%} vs control {out['final_control']:.1%}, {nclu} clusters")
