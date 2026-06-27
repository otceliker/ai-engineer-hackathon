#!/usr/bin/env python3
"""Generation worker for the STaR loop (vLLM). Runs as a subprocess so the GPU is
fully released before the training step.

Reads a prompts JSONL ({"id","prompt"} per line), generates, writes an attempts
JSONL ({"id","sample_idx","text"}). Optional LoRA adapter applied via LoRARequest.
Grading happens in the orchestrator (CPU), not here.
"""
import argparse, json, os, sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit for base model)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1, help="samples per prompt (K harvest / 1 eval)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts)]
    prompts = [r["prompt"] for r in rows]
    ids = [r["id"] for r in rows]

    llm = LLM(model=args.base, enable_lora=True, max_lora_rank=16,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True,
              max_model_len=args.max_model_len, dtype="bfloat16")
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=args.top_p,
                        max_tokens=args.max_tokens)
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

    outs = llm.generate(prompts, sp, lora_request=lora)
    with open(args.out, "w") as f:
        for pid, o in zip(ids, outs):
            for s, comp in enumerate(o.outputs):
                f.write(json.dumps({"id": pid, "sample_idx": s, "text": comp.text}) + "\n")
    print(f"[ok] wrote attempts for {len(ids)} prompts x {args.n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
