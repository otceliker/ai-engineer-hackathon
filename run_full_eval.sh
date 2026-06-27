#!/usr/bin/env bash
# Run the MATH eval across all downloaded 1B-class models (GPU host, e.g. neptune).
# Per-model generation settings: reasoning models get a longer budget.
set -u
cd "$(dirname "$0")"
export VLLM_USE_FLASHINFER_SAMPLER=0          # flashinfer JIT sampler fails to build on this box
export PATH="$PWD/.venv/bin:$PATH"            # vLLM shells out to `ninja`
P=.venv/bin/python

echo "### Qwen2.5-Math-1.5B-Instruct ($(date))"
$P scripts/eval_math.py --model models/Qwen__Qwen2.5-Math-1.5B-Instruct --max-tokens 3000 --max-model-len 4096
echo "### Llama-3.2-1B-Instruct ($(date))"
$P scripts/eval_math.py --model models/meta-llama__Llama-3.2-1B-Instruct --max-tokens 2048 --max-model-len 4096
echo "### DeepSeek-R1-Distill-Qwen-1.5B ($(date))"
$P scripts/eval_math.py --model models/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B --max-tokens 8192 --max-model-len 16384 --temperature 0.6 --top-p 0.95
echo "### ALL DONE ($(date))"
