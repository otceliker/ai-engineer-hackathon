#!/usr/bin/env python3
"""BFCL V4 single-turn agent runner + AST scorer (vLLM, GPU host).

Exposes a reusable `BfclAgent` that holds the vLLM engine + tokenizer so the
streaming loop can re-evaluate many candidate system prompts without reloading
the model. Also runnable standalone to produce the static-prompt baseline.

Scoring paths (selected by record["kind"]):
  ast         -> vendored BFCL ast_checker on the parsed call(s)
  irrelevance -> correct iff the model emits ZERO function calls
  relevance   -> correct iff the model emits >=1 parseable function call

The function-calling format is Qwen-native (<tool_call>{...}</tool_call>), produced
by the model's own chat template with per-record `tools`. We parse those blocks
tolerantly (also accepting a bare JSON object/array if the tags are missing).

Examples:
    # static baseline on the strictly held-out split (the number to beat)
    python scripts/bfcl_eval.py --model models/Qwen__Qwen2.5-1.5B-Instruct --split stream_test

    # quick smoke test
    python scripts/bfcl_eval.py --model models/Qwen__Qwen2.5-1.5B-Instruct --split patch_dev --limit 20
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.bfcl_vendor import ast_checker, Language  # noqa: E402

DEFAULT_SPLITS = os.path.join(ROOT, "data", "bfcl", "splits.json")

# The static baseline system prompt. Patches append a lessons block to this.
BASE_SYSTEM_PROMPT = (
    "You are a precise function-calling assistant. You are given a set of tools "
    "in the system context. For the user's request, decide which tool(s) to call "
    "and with what arguments.\n"
    "Rules:\n"
    "- Only call a function if the request can be fulfilled by the available tools. "
    "If none of the tools apply, do not call any function.\n"
    "- Use the exact parameter names and value types from each tool's schema.\n"
    "- Provide every required parameter.\n"
    "- When the request asks for several independent actions, emit one tool call per action.\n"
    "- Output only tool calls, with no extra commentary."
)

# BFCL schema uses a few non-JSON-Schema type names; normalize for the prompt.
# (The vendored checker still scores against the ORIGINAL BFCL spec.)
_TYPE_FIX = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def _fix_types(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_FIX.get(v, v)
            else:
                out[k] = _fix_types(v)
        return out
    if isinstance(node, list):
        return [_fix_types(x) for x in node]
    return node


def to_openai_tools(functions):
    """BFCL function specs -> OpenAI tool format for the chat template."""
    tools = []
    for fn in functions:
        params = _fix_types(fn.get("parameters", {"type": "object", "properties": {}}))
        if params.get("type") == "dict":
            params["type"] = "object"
        tools.append({
            "type": "function",
            "function": {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": params,
            },
        })
    return tools


_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text):
    """Extract function calls from model output as [{"name", "arguments"}].

    Primary path: Qwen <tool_call>...</tool_call> blocks. Fallback: a bare JSON
    object/array if the model omitted the tags. Unparseable -> dropped (counts
    as a malformed call, which the scorer treats as a miss)."""
    calls = []
    blocks = _TOOLCALL_RE.findall(text)
    if blocks:
        for b in blocks:
            obj = _safe_json(b)
            calls.extend(_normalize_calls(obj))
        return calls
    # Fallback: try the first JSON object/array in the text.
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        calls.extend(_normalize_calls(_safe_json(m.group(1))))
    return calls


def _safe_json(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def _normalize_calls(obj):
    """Coerce parsed JSON into a list of {"name", "arguments"} dicts."""
    if obj is None:
        return []
    if isinstance(obj, list):
        out = []
        for o in obj:
            out.extend(_normalize_calls(o))
        return out
    if isinstance(obj, dict):
        if "name" in obj:
            args = obj.get("arguments", obj.get("parameters", {}))
            if not isinstance(args, dict):
                args = {}
            return [{"name": obj["name"], "arguments": args}]
        # {func_name: {args}} shape
        if len(obj) == 1:
            (k, v), = obj.items()
            if isinstance(v, dict):
                return [{"name": k, "arguments": v}]
    return []


def score_record(record, parsed_calls):
    """Return True iff the parsed calls are correct for this record's kind."""
    kind = record["kind"]
    if kind == "irrelevance":
        return len(parsed_calls) == 0
    if kind == "relevance":
        return len(parsed_calls) >= 1
    # ast
    model_output = [{c["name"]: c["arguments"]} for c in parsed_calls]
    try:
        res = ast_checker(
            record["function"], model_output, record["ground_truth"],
            Language.PYTHON, record["test_category"], "local",
        )
        return bool(res.get("valid", False))
    except Exception:
        return False


def render_system_prompt(lessons=None, base=BASE_SYSTEM_PROMPT):
    """Static base prompt, optionally with a capped lessons playbook appended."""
    if not lessons:
        return base
    bullets = "\n".join(f"- {l}" for l in lessons)
    return f"{base}\n\nLearned guidance (apply when relevant):\n{bullets}"


class BfclAgent:
    """Holds the vLLM engine + tokenizer; evaluate() scores a split under a prompt."""

    def __init__(self, model_path, max_model_len=4096, gpu_mem=0.9,
                 temperature=0.0, top_p=1.0, max_tokens=512, seed=0, enforce_eager=True):
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as e:
            sys.exit(f"Missing dependency: {e}. Install: uv pip install -r requirements-eval.txt")
        self.model_path = model_path
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(model=model_path, max_model_len=max_model_len or None,
                       gpu_memory_utilization=gpu_mem, dtype="bfloat16",
                       enforce_eager=enforce_eager, seed=seed)
        self.sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def complete(self, messages, max_tokens=512, temperature=0.7, top_p=0.95, seed=None):
        """Generic chat completion on the shared engine (used by the local proposer)."""
        from vllm import SamplingParams
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                            seed=seed)
        out = self.llm.generate([prompt], sp)
        return out[0].outputs[0].text

    def build_prompt(self, record, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]
        # question is [[turn messages]]; single-turn -> first turn's messages
        messages.extend(record["question"][0])
        tools = to_openai_tools(record["function"])
        return self.tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )

    def evaluate(self, records, system_prompt, sampling_params=None):
        """Generate + score. Returns a dict with per-id correctness and aggregates."""
        prompts = [self.build_prompt(r, system_prompt) for r in records]
        outputs = self.llm.generate(prompts, sampling_params or self.sp)
        per_id, by_cat, raw = {}, defaultdict(lambda: [0, 0]), {}
        for r, o in zip(records, outputs):
            text = o.outputs[0].text
            calls = parse_tool_calls(text)
            ok = score_record(r, calls)
            per_id[r["id"]] = ok
            raw[r["id"]] = text
            by_cat[r["category"]][0] += int(ok)
            by_cat[r["category"]][1] += 1
        n_correct = sum(per_id.values())
        return {
            "n": len(records),
            "n_correct": n_correct,
            "accuracy": n_correct / len(records) if records else 0.0,
            "per_id": per_id,
            "by_category": {k: tuple(v) for k, v in by_cat.items()},
            "raw": raw,
        }


def _load_split(splits_file, split):
    with open(splits_file) as f:
        payload = json.load(f)
    if split not in payload["splits"]:
        sys.exit(f"Unknown split '{split}'. Have: {list(payload['splits'])}")
    return payload["splits"][split], payload["meta"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="stream_test", choices=["stream_train", "patch_dev", "stream_test"])
    ap.add_argument("--splits-file", default=DEFAULT_SPLITS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lessons-file", default=None, help="JSON list of lesson strings to append to the base prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    records, meta = _load_split(args.splits_file, args.split)
    if args.limit:
        records = records[:args.limit]
    lessons = json.load(open(args.lessons_file)) if args.lessons_file else None
    system_prompt = render_system_prompt(lessons)

    print(f"Split: {args.split}  n={len(records)}  (splits meta: seed={meta['seed']}, cap={meta['max_per_category']})", flush=True)
    print(f"Lessons: {len(lessons) if lessons else 0}", flush=True)

    agent = BfclAgent(args.model, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                      temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    t0 = time.time()
    res = agent.evaluate(records, system_prompt)
    elapsed = time.time() - t0

    model_name = os.path.basename(args.model.rstrip("/"))
    out_path = args.out or os.path.join(ROOT, "results", f"bfcl_{model_name}_{args.split}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps({
                "id": r["id"], "category": r["category"], "kind": r["kind"],
                "correct": res["per_id"][r["id"]], "output": res["raw"][r["id"]],
            }) + "\n")

    print(f"\n=== {model_name} | {args.split} ===")
    print(f"Accuracy: {res['n_correct']}/{res['n']} = {res['accuracy']:.1%}")
    print(f"Wall time: {elapsed:.1f}s ({elapsed/max(res['n'],1):.2f}s/task)")
    print("\nBy category:")
    for cat in sorted(res["by_category"]):
        c, n = res["by_category"][cat]
        print(f"  {cat:24s}: {c:3d}/{n:3d} = {c/n:.1%}")
    print(f"\nPer-task results -> {out_path}")


if __name__ == "__main__":
    main()
