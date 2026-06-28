#!/usr/bin/env python3
"""IFEval generation step (in-process vLLM). The ONLY thing we write — scoring is the
official google-research evaluator (scripts/ifeval_vendor/evaluation_main.py).

Reads IFEval prompts (input_data.jsonl: {key, prompt, instruction_id_list, kwargs}),
generates one response per prompt with a local model, and writes the response file the
official evaluator expects: lines of {"prompt": <text>, "response": <model output>}.

A/B is done by swapping the system prompt:
  - baseline : neutral system message only.
  - +lessons : neutral system message + a "Lessons" block from --lessons-file (json {"lessons":[...]}).
Both arms share the SAME neutral base, so the only variable is the lessons (clean isolation).

Usage:
    python scripts/ifeval_gen.py --model models/Qwen__Qwen2.5-1.5B-Instruct \
        --input data/ifeval/input_data.jsonl --out results/ifeval/resp_base.jsonl
    python scripts/ifeval_gen.py --model ... --lessons-file scripts/ifeval_lessons.json \
        --out results/ifeval/resp_lessons.jsonl
"""
import argparse
import json
import os
import sys

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run inside the vllm venv (.venv).")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_SYSTEM = "You are a helpful assistant. Follow the user's instructions carefully and exactly."


def load_prompts(path):
    return [json.loads(l) for l in open(path)]


def build_system(lessons_file):
    if not lessons_file:
        return BASE_SYSTEM
    d = json.load(open(lessons_file))
    lessons = d.get("lessons", d) if isinstance(d, dict) else d
    lessons = [l for l in lessons if isinstance(l, str)]
    if not lessons:
        return BASE_SYSTEM
    block = "\n\nLessons learned from past mistakes:\n" + "\n".join(f"- {l}" for l in lessons)
    return BASE_SYSTEM + block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", default=os.path.join(ROOT, "data", "ifeval", "input_data.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lessons-file", default=None)
    ap.add_argument("--keys-file", default=None,
                    help="optional json list of keys to restrict to (e.g. a train/test split)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1280)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_prompts(args.input)
    if args.keys_file:
        keep = set(json.load(open(args.keys_file)))
        rows = [r for r in rows if r["key"] in keep]
    if args.limit:
        rows = rows[:args.limit]

    system = build_system(args.lessons_file)
    model_name = os.path.basename(args.model.rstrip("/"))
    print(f"Model: {model_name} | prompts: {len(rows)} | lessons: {bool(args.lessons_file)} "
          f"| max_tokens={args.max_tokens} temp={args.temperature}", flush=True)
    print(f"System prompt ({len(system)} chars):\n{system}\n{'-'*60}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

    llm = LLM(model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"prompt": r["prompt"], "response": o.outputs[0].text}) + "\n")
    print(f"Wrote {len(rows)} responses -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
