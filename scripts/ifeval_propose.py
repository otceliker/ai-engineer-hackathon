#!/usr/bin/env python3
"""Strong proposer writes IFEval lessons from real train failures (DigitalOcean OpenAI-compatible).

Reads train_failures.json (prompt + response + violated-constraint descriptions), shows the
proposer the full failure-type histogram plus a sample of concrete failures, and asks for EXACTLY
N distinct, general, reusable lessons. Writes {"lessons":[...]} for ifeval_gen.py to inject.

Env: DO_TOKEN | PROPOSER_API_KEY ; PROPOSER_BASE_URL (default DO). Model via --proposer-model.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def first_json(text):
    """Extract the first JSON object/array from a possibly-chatty completion."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    s = min(starts)
    depth, instr, esc = 0, False, False
    for i in range(s, len(text)):
        ch = text[i]
        if instr:
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[s:i + 1])
    raise ValueError("unbalanced JSON")


SYSTEM = (
    "You improve an instruction-following agent by writing a SHORT playbook of general lessons that "
    "will be appended to its system prompt. You are given tasks the agent FAILED: the request, the "
    "agent's response, and the SPECIFIC constraints it violated.\n\n"
    "Write EXACTLY {n} lessons that, taken together, would most improve the agent.\n"
    "Requirements:\n"
    "- Each lesson is general and reusable (a behavior pattern that applies across many tasks), "
    "NEVER a single task's answer.\n"
    "- Imperative and short (one sentence).\n"
    "- The {n} lessons must be DISTINCT and NON-OVERLAPPING; cover the widest range of observed "
    "failure modes (counting, formatting, case, punctuation, keywords, structure, etc.).\n"
    "- These are ONE-SHOT instructions: the agent produces a single response, so phrase lessons as "
    "guidance for writing that response correctly, NOT as 'review and revise afterward'.\n"
    "- Order by how many failures they would fix (most impactful first).\n\n"
    'Output ONLY JSON: {{"lessons": ["...", "..."]}} with exactly {n} strings.'
)


def fmt(f):
    viol = "\n".join(f"  - {x['desc']}" for x in f["failed"])
    return (f"REQUEST: {f['prompt'][:400]}\n"
            f"AGENT RESPONSE: {f['response'][:500]}\n"
            f"VIOLATED CONSTRAINTS:\n{viol}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failures", required=True)
    ap.add_argument("--proposer-model", default="deepseek-v4-pro")
    ap.add_argument("--n-lessons", type=int, default=6)
    ap.add_argument("--max-show", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import requests
    key = os.environ.get("PROPOSER_API_KEY") or os.environ.get("DO_TOKEN")
    base = os.environ.get("PROPOSER_BASE_URL", "https://inference.do-ai.run/v1").rstrip("/")
    if not key:
        sys.exit("Need DO_TOKEN or PROPOSER_API_KEY.")

    data = json.load(open(args.failures))
    fails = data["failures"]
    rng = random.Random(args.seed)
    shown = fails[:]
    rng.shuffle(shown)
    shown = shown[:args.max_show]
    hist = "\n".join(f"  {c:4d}  {iid}" for iid, c in data["failed_instruction_histogram"].items())
    blocks = "\n\n".join(fmt(f) for f in shown)
    user = (f"The agent failed {data['n_failures']} of {data['n_train']} training tasks.\n\n"
            f"Failure counts by constraint type (full train set):\n{hist}\n\n"
            f"Here are {len(shown)} sampled failures:\n\n{blocks}\n\n"
            f"Write exactly {args.n_lessons} lessons as JSON.")

    payload = {"model": args.proposer_model, "max_tokens": 4096, "temperature": 0.3,
               "messages": [{"role": "system", "content": SYSTEM.format(n=args.n_lessons)},
                            {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    delay, lessons = 2, None
    for attempt in range(1, 7):
        try:
            r = requests.post(f"{base}/chat/completions", json=payload, headers=headers, timeout=240)
        except requests.RequestException as e:
            last = f"net error: {e}"
        else:
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                obj = first_json(text)
                lst = obj.get("lessons", obj) if isinstance(obj, dict) else obj
                lessons = [x for x in lst if isinstance(x, str)][:args.n_lessons]
                break
            if r.status_code not in (429, 500, 502, 503, 504):
                sys.exit(f"Proposer API {r.status_code}: {r.text[:300]}")
            last = f"HTTP {r.status_code}"
        if attempt < 6:
            print(f"  [proposer] {last}; backoff {delay}s", flush=True)
            time.sleep(delay); delay = min(delay * 2, 60)
    if not lessons:
        sys.exit(f"Proposer failed after retries: {last}")

    json.dump({"_proposer": args.proposer_model, "n_failures": data["n_failures"], "lessons": lessons},
              open(args.out, "w"), indent=1)
    print(f"[{args.proposer_model}] proposed {len(lessons)} lessons -> {args.out}")
    for i, l in enumerate(lessons, 1):
        print(f"  {i}. {l}")


if __name__ == "__main__":
    main()
