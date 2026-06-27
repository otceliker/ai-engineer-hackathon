# ai-engineer-hackathon — Project Wiki

Living notebook for this repo: what we built, what we found, why we chose what we
chose. Newest sections at the bottom of each part. Last updated 2026-06-27.

---

## 1. What this is

Hackathon submission repo. Working thesis: explore **small (1B-class) open-weight
models on MATH**, and test whether cheap **activation steering vectors** can act as
a "skill primitive" — a way to add capability at inference without training in the
loop. If that works, the project leans on activation vectors; if not, LoRAs stay the
workhorse and activations are diagnostic-only.

Repo: https://github.com/otceliker/ai-engineer-hackathon (public).

---

## 2. Infrastructure

Two hosts, kept in sync via the same scripts + a committed `requirements*.txt`.

| | Mac (laptop) | neptune (server) |
|---|---|---|
| Role | interp/dev, mirror | GPU eval + experiments |
| Reach | local | Tailscale `100.64.113.61` (LAN `192.168.70.99` when home) |
| Compute | Apple Silicon, MPS | RTX 3090, 24 GB VRAM |
| Python env | `uv` venv `.venv` | `uv` venv `.venv` |

Notes / gotchas:
- During the hackathon the Mac was on venue wifi (`10.84.x`), **off** the home LAN,
  so neptune is only reachable over **Tailscale** (`100.64.113.61`). `sshn`/LAN IP
  time out from there.
- `uv` was not on the Mac initially — installed via the official script to
  `~/.local/bin/uv`. Already present on neptune at `/home/orhan/.local/bin/uv`.
- HF token lives in the Mac's `~/.zshrc` as `HF_TOKEN` (needed for the gated Llama).
  It was piped to neptune's `~/.cache/huggingface/token` (0600), never echoed.
- **neptune's `llm-server.service` (qwen3.6-27b) was STOPPED** to free the GPU for
  the hackathon (`sudo systemctl stop llm-server.service` → VRAM 23.6 GB → 1 MiB).
  It was `stop`ped, not `disable`d, so a reboot would bring it back.
  **To restore at the end:** `sudo systemctl start llm-server.service`.

---

## 3. Models & data on disk

All safetensors, under `models/` (gitignored). Downloaded with `scripts/download.py`
(idempotent, resumable, exponential backoff). Both hosts have the first three;
**neptune additionally** has the two general Qwens.

| Key | Repo | Size | Type | Hosts |
|---|---|---|---|---|
| deepseek | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 3.4G | reasoning | Mac + neptune |
| qwen | Qwen/Qwen2.5-Math-1.5B-Instruct | 2.9G | math-specialized | Mac + neptune |
| llama | meta-llama/Llama-3.2-1B-Instruct (gated) | 2.4G | general | Mac + neptune |
| qwen25-general | Qwen/Qwen2.5-1.5B-Instruct | 2.9G | general | neptune |
| qwen3 | Qwen/Qwen3-1.7B | 3.8G | general + thinking | neptune |

Dataset: **Hendrycks MATH** via `nlile/hendrycks-MATH-benchmark` (clean 12,000 train /
500 test split) under `data/` (gitignored). Columns: `problem, solution, answer,
subject, level, unique_id`. Graded with **`math-verify`** (HF) on the `\boxed{}`
answer — never hand-rolled.

---

## 4. MATH benchmark results (full 500-problem test set)

Run via `run_full_eval.sh` (vLLM + math-verify). See `RESULTS.md` for the canonical
copy; per-example JSONL in `results/` (gitignored).

| Model | Accuracy | Time | Notes |
|---|---|---|---|
| **Qwen3-1.7B (thinking)** | **69.0%** (345/500) | 1578s | newest, top scorer, long CoT |
| DeepSeek-R1-Distill-Qwen-1.5B | 64.8% (324/500) | 498s | reasoning distill |
| Qwen2.5-Math-1.5B-Instruct | 60.0% (300/500) | 62s | math-specialized |
| Qwen2.5-1.5B-Instruct (general) | 45.4% (227/500) | 64s | general baseline |
| Llama-3.2-1B-Instruct | 24.8% (124/500) | 38s | general, weak at math |

Takeaways:
- **Math specialization buys ~+14.6 pts** at fixed size/recipe: Qwen2.5-Math (60.0%)
  vs Qwen2.5 general (45.4%). The cleanest controlled comparison we have.
- **Reasoning/thinking helps more than specialization**: Qwen3-1.7B thinking (69.0%)
  and DeepSeek distill (64.8%) top the math-tuned non-reasoning model — at a large
  latency cost (Qwen3 ~25× slower than Qwen-Math).
- DeepSeek's 64.8% is below its published ~83% pass@1 — the 8192-token cap truncates
  some long traces and it's a single low-temp sample. Tunable, not chased.
- Per-subject (Qwen2.5-1.5B general, used to pick the steering category):
  Precalc 12.5% · Inter-Algebra 27.8% · Geometry 39.0% · **Counting&Prob 42.1%** ·
  **Prealgebra 52.4%** · Algebra 61.3% · Number Theory 67.7%.

---

## 5. Tooling

- `scripts/download.py` — fetch any of the 5 models + the MATH dataset. `--all`
  includes the gated Llama; `--only <key>` for one. Keeps safetensors, skips
  duplicate `.bin/.pth/.gguf`.
- `scripts/eval_math.py` — quick MATH eval via **vLLM + math-verify**. Reports
  accuracy overall / by level / by subject, writes per-example JSONL.
- `run_full_eval.sh` — runs all models with per-model gen settings.
- `scripts/steer_math.py` — the steering-vector capability gate (Part 7).
- `requirements.txt` (download/grading) + `requirements-eval.txt` (vLLM/ninja).

### vLLM gotchas on a fresh box (neptune)
Learned the hard way; baked into the scripts/`requirements-eval.txt`:
1. vLLM shells out to **`ninja`** — must be pip-installed **and** `<venv>/bin` on
   `PATH` at runtime (non-interactive SSH doesn't have it).
2. **flashinfer**'s JIT sampler kernels fail to compile here → uninstall
   `flashinfer-python` and run with `VLLM_USE_FLASHINFER_SAMPLER=0` (native sampling).
3. Default to **`enforce_eager`** (skip `torch.compile`) for reliable startup.
4. Stack: torch 2.11.0+cu130, transformers 5.12.1, vllm 0.23.0.

---

## 6. Key decisions (log)

- **Format = safetensors, not GGUF.** GGUF/llama.cpp is a closed inference engine
  with no clean per-layer hidden-state hooks; activation/steering work needs
  PyTorch + transformers. (Drove the whole download format choice.)
- **Eval engine = vLLM** for the 500-problem runs (batched, fast); **transformers +
  forward hooks** for the steering work (vLLM can't expose the residual stream).
- **Steering model = Qwen2.5-1.5B-Instruct** (non-reasoning): mean-pooling a vector
  over hundreds of CoT tokens would dilute it to mush, so reasoning models are wrong
  for this test. Category **Prealgebra** (52.4%, closest to the 40–60% Goldilocks
  band, plenty of train-split problems).
- **Build/eval steering on the MATH *train* split**, never the 500-item test set —
  otherwise the benchmark we report gets contaminated.
- Committed tooling only; weights, datasets, and `results/` are gitignored.

---

## 7. Steering-vector capability gate

**One question (a GATE, not a feature):** can a single difference-of-means activation
vector, added to the residual stream at inference, raise MATH accuracy on a skill
category? YES → activation vectors become the cheap skill primitive. NO → they're
diagnostic-only and LoRAs stay the workhorse. **A flat result is a valid, useful
answer** — it kills the path fast, which is the point. Time-boxed ~45 min.

### Design (after a review cycle)
Built `scripts/steer_math.py`: label train attempts correct/incorrect (greedy +
math-verify), build `v = mean(act|correct) − mean(act|incorrect)` mean-pooled over
generated tokens at mid-stack layers, normalize, add `alpha · mean_resid_norm · v̂`
to the residual during generation, sweep, measure held-out accuracy.

A critique → designer-ruling cycle hardened it. Items folded in:
- **#1 Paired McNemar, not a +5pp/N=60 threshold.** At N=60 the SE of a proportion
  difference is ~9pp, so +5pp sits inside the noise. Baseline & steered run on the
  *same* held-out items → analyze the **flip matrix** (fail→pass vs pass→fail) with
  exact McNemar. "Cold-fail recovery" *is* the fail→pass cell. WIN bar raised to
  **+10pp AND p<0.05**.
- **#2 Shuffled-label control** is the real null (the random unit vector is the weak
  one). `mean(correct)−mean(incorrect)` can encode "easy problem / be fluent", not
  skill, because correct problems are systematically easier. Permuting the
  correct/incorrect labels and rebuilding keeps the data distribution and destroys
  only the label correlation. Kept the random-vector control as secondary.
- **#4 Hook gated to decode steps only** (`hs.shape[1]==1`) so it steers generation,
  not the prompt during prefill.
- **#5 Alpha grid extended downward** `{-1,-0.5,0,0.1,0.25,0.5,0.75,1,2}` — a real
  effect, if any, shows at small α before coherence collapse.
- **#7 `\boxed{}` emission rate** logged separately, so a format break can't
  masquerade as a capability drop.
- **Winner's-curse hedge:** picking the best of 27 configs then testing the null only
  there is lenient. Each null (5 shuffled + 5 random) runs its **own** positive-alpha
  sweep at the best layer; WIN must beat the **null max**.
- Held the line on **not over-engineering the gate**: single coarse pass, no
  N=150–200 confirmation unless a candidate appears. FLAT framed as *power-limited*
  ("no effect detectable at N=60"), not "no effect exists". One permitted follow-up
  if dead flat: try one earlier layer (~30% depth), then stop.

### Bug caught by the smoke test (worth remembering)
In **transformers 5.x the Qwen decoder layer returns a bare tensor, not a tuple**, so
`out[0]` indexed the batch dim and the inject hook was a **silent no-op** (every α gave
identical output). Fixed to handle tensor-or-tuple + a guard that warns if injection
never fires. Smoke test then showed the expected dose-response (α=0 == baseline; α<0
and large α degrade). *Always smoke-test a steering hook by checking that a large α
actually breaks generation.*

### Result (Prealgebra, Qwen2.5-1.5B, baseline held-out 56.7%, N=60)
The grid is unambiguous: **no positive α at any of layers 12/16/19 beats baseline.**
The vector only ever does nothing or hurts, monotonically.

| layer | α=0.1 | 0.25 | 0.5 | 0.75 | 1 | 2 |
|---|---|---|---|---|---|---|
| 12 | −6.7 | −28.3 | −55.0 | −55.0 | −53.3 | −56.7 |
| 16 | **+0.0** | −28.3 | −55.0 | −53.3 | −53.3 | −56.7 |
| 19 | −5.0 | −16.7 | −48.3 | −55.0 | −56.7 | −56.7 |

(Δpp vs baseline. α=0 reproduces 56.7% exactly at all layers — hook sanity ✓.
α<0 destroys accuracy; large α collapses coherence.)

Best positive config: **layer 16, α=0.1 → 56.7% = +0.0pp** (5 fail→pass vs 5
pass→fail, McNemar p=1.00). Net zero.

### Verdict: **FLAT / DEGRADE**
Best real Δ is 0.0pp — it cannot clear the +10pp WIN bar regardless of controls.
Pushing along the difference-of-means direction never recovers a single net problem;
the only safe point is α≈0. The direction is **entangled with general capability**,
not a clean math-skill primitive.

**Recommendation:** steering can't carry capability here on this model/category.
**Keep LoRAs as the workhorse; use activation vectors as diagnostic/visual only.**
Caveat that matters even for a hypothetical WIN: a positive result could still reflect
a "be-more-careful/easy-problem" axis rather than a skill primitive — the
shuffled-label control is what would have licensed any causal read.

---

## 8. Open items / next steps

- [ ] Restore neptune `llm-server.service` at hackathon end (`sudo systemctl start`).
- [ ] Mac: only base + grading deps installed; add torch/transformers/nnsight there
      when starting interp work (MPS).
- [ ] Steering single permitted follow-up *only if* we revisit: one earlier layer
      (~depth 8) — otherwise the gate has answered.
- [ ] Not evaluated for steering: other categories/models (out of scope per stop rule).

## 9. Reproduce

```bash
# download (per host)
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download.py --all          # needs HF_TOKEN for llama

# MATH eval (GPU host)
uv pip install --python .venv/bin/python -r requirements-eval.txt
VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$PWD/.venv/bin:$PATH" bash run_full_eval.sh

# steering gate (GPU host, transformers + hooks)
VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python scripts/steer_math.py --model models/Qwen__Qwen2.5-1.5B-Instruct
```
