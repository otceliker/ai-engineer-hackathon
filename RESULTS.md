# MATH benchmark results

Hendrycks MATH, full 500-problem test split (`nlile/hendrycks-MATH-benchmark`).
Graded with `math-verify` on the `\boxed{}` answer. Run on neptune (RTX 3090) via
`run_full_eval.sh`. Per-example outputs are written to `results/*.jsonl` (gitignored).

| Model | Accuracy | Wall time | Gen settings |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | **64.8%** (324/500) | 498s | T=0.6, top_p=0.95, max_tok=8192 |
| Qwen2.5-Math-1.5B-Instruct | **60.0%** (300/500) | 62s | greedy, max_tok=3000 |
| Llama-3.2-1B-Instruct | **24.8%** (124/500) | 38s | greedy, max_tok=2048 |

DeepSeek-R1-Distill accuracy by difficulty level:

| L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| 79.1% | 71.1% | 69.5% | 66.4% | 50.7% |

Notes:
- DeepSeek's 64.8% is below its published ~83% pass@1 — the 8192-token cap truncates
  some long reasoning traces, and this is a single low-temp sample. Raise `--max-tokens`
  and/or average multiple samples to chase the headline number.
- Not yet evaluated: `Qwen2.5-1.5B-Instruct` (general baseline) and `Qwen3-1.7B`.
