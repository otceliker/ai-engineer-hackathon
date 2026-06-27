# MATH benchmark results

Hendrycks MATH, full 500-problem test split (`nlile/hendrycks-MATH-benchmark`).
Graded with `math-verify` on the `\boxed{}` answer. Run on neptune (RTX 3090) via
`run_full_eval.sh`. Per-example outputs are written to `results/*.jsonl` (gitignored).
Deeper context, design decisions, and the steering experiment live in `docs/WIKI.md`.

| Model | Accuracy | Wall time | Gen settings |
|---|---|---|---|
| Qwen3-1.7B (thinking) | **69.0%** (345/500) | 1578s | T=0.6, top_p=0.95, max_tok=8192 |
| DeepSeek-R1-Distill-Qwen-1.5B | **64.8%** (324/500) | 498s | T=0.6, top_p=0.95, max_tok=8192 |
| Qwen2.5-Math-1.5B-Instruct | **60.0%** (300/500) | 62s | greedy, max_tok=3000 |
| Qwen2.5-1.5B-Instruct (general) | **45.4%** (227/500) | 64s | greedy, max_tok=3000 |
| Llama-3.2-1B-Instruct | **24.8%** (124/500) | 38s | greedy, max_tok=2048 |

DeepSeek-R1-Distill accuracy by difficulty level:

| L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|
| 79.1% | 71.1% | 69.5% | 66.4% | 50.7% |

Takeaways:
- **Math specialization buys ~+14.6 pts** at fixed size/recipe: Qwen2.5-Math (60.0%)
  vs Qwen2.5 general (45.4%).
- **Reasoning/thinking beats specialization**: Qwen3-1.7B thinking (69.0%) and DeepSeek
  distill (64.8%) top the math-tuned non-reasoning model — at large latency cost
  (Qwen3 ~25× slower than Qwen-Math).
- DeepSeek's 64.8% is below its published ~83% pass@1 — the 8192-token cap truncates
  some long reasoning traces and this is a single low-temp sample.

## Steering-vector capability gate
Can a difference-of-means activation vector raise MATH accuracy at inference?
**Verdict: FLAT / DEGRADE** on Qwen2.5-1.5B / Prealgebra — no positive α at layers
12/16/19 beat the 56.7% baseline; the direction is entangled with general capability.
Full design, controls, and grid in `docs/WIKI.md` §7. → Keep LoRAs as the workhorse;
activation vectors are diagnostic-only.
