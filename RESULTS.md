# BFCL V4 — online lessons-memory prompt-patching (Direction 3, current)

## ★★★ BBH CLEAN RESULT — fully-consistent 3-shot, failure-aware lessons, n=675 (2026-06-28)

The authoritative BBH number. One consistent protocol end-to-end (vLLM-accelerated, `scripts/bbh_vllm.py`):
**3-shot CoT train base → correctly-scored real failures (226/675) → activation-clustered (K=20) →
failure-aware lessons (model shown its OWN 3-shot wrong answer + reasoning + correct answer) → 3-shot
test (base vs lessoned)**. Train→test disjoint (verified, no leakage). Matched n=675/arm, Qwen2.5-7B.

| metric | base | lessoned | Δ |
|---|---|---|---|
| strict (official answer-format) | 0.677 | 0.677 | **+0.000** |
| format-agnostic (reasoning) | 0.696 | 0.711 | **+0.015** |

Per-task (robust) net-positive: boolean +.24, date +.12, colored_objects +.12, word_sorting +.12,
logical_deduction_3/7 +.08, ruin_names +.08, movie +.08, geometric +.08; vs formal_fallacies −.16,
salient_translation −.16, object_counting −.12, penguins −.08.

**VERDICT:** under a trustworthy, leakage-free, consistent protocol, training-free failure-derived
self-lessons are **≈ neutral-to-marginally-positive on BBH reasoning** (strict exact tie; +1.5%
format-agnostic), with real positive swings on ~half the tasks. NOT the strong negative the old broken
scorer reported. The lessons' only systematic cost is slightly reduced answer-format compliance (the
strict↔robust gap). Combined with IFEval (clean +5.5% held-out, p=0.036): **prompt-lessons RSI clearly
helps instruction-following and is roughly break-even on reasoning.** Supersedes all BBH sections below.

---

## ⚠️★ BBH CORRECTION — homemade scorer was broken; official-protocol re-measure shows lessons ≈ NEUTRAL (2026-06-28)

**The BBH sections below used a rolled-our-own zero-shot scorer that is broken on multiple-choice output**
(it only strips `(X)` parens when the string starts `(` AND ends `)`, so "(O)." or restated options score
wrong). Re-ran on the **official BBH protocol** (3-shot CoT prompts from `suzgunmirac/BIG-Bench-Hard`,
2048 tok, proper MC-letter / yes-no / free-form matching; `scripts/bbh_official.py`, gen-cached + live).

6-task subset, 12 examples each (n=72), Qwen2.5-7B:

| task | base | lessoned | Δ |
|---|---|---|---|
| word_sorting | 0.083 | 0.000 | −0.083 |
| logical_deduction_7 | 0.750 | 0.500 | −0.250 |
| tracking_shuffled_7 | 0.667 | 0.750 | +0.083 |
| reasoning_colored_objects | 0.917 | 0.833 | −0.083 |
| date_understanding | 0.667 | 0.917 | +0.250 |
| navigate | 0.917 | 0.917 | 0.000 |
| **OVERALL (n=72)** | **0.667** | **0.653** | **−0.014** |

**Fixing the scorer moved base 1.8% → 66.7% on the subset (~37× undercount).** Corrected verdict: under a
trustworthy protocol, BBH lessons are **≈ neutral** (Δ −1.4% = one example, within noise; mixed by task:
date +25%, logical_deduction −25%) — **NOT** the strong negative the sections below report. The old
"11/11 sub-base, −4.6..−7.4%" result is a **measurement artifact** (broken scorer + 512-tok truncation ×
lesson-induced format drift) and should not be cited. IFEval (official google-research evaluator) is clean —
its positive stands. Open: scale the official run to all 27 tasks / more examples for a significance-grade
number. Lesson: never roll your own benchmark scorer.

---

## BBH lesson-config sweep (Qwen2.5-7B) — ⚠️ SUPERSEDED (broken scorer; see correction above) (2026-06-28)

Single-pass greedy, n=216 subset, greedy base 65.7%. Ranked Δ (none beat base):

| config | acc | Δ vs greedy base |
|---|---|---|
| cot_512 (CoT-written lessons) | 64.4% | −1.4% (least harm) |
| max_lessons2 | 62.5% | −3.2% |
| v2_repro / max_lessons1 | 61.1% | −4.6% |
| proposer_deepseek | 60.6% | −5.1% |
| ncand8_temp07 / k15 | 60.2% | −5.6% |
| topk2 | 59.7% | −6.0% |
| **gated_ref (gated selection)** | 59.3% | **−6.5%** |
| k6 | 58.8% | −6.9% |
| **proposer_glm** | 58.3% | **−7.4% (worst)** |

**Verdict: on BBH reasoning, self-clustered prompt-lessons are net-HARMFUL at greedy — NONE of 11
configs (any K, top-k, lesson count, gated or ungated, self or external-proposer) beats greedy base.**
Sub-findings: (a) better lesson-WRITING helps — CoT-self least harmful (−1.4%); (b) gating did NOT
rescue (−6.5%); (c) more/noisier candidates, fewer clusters all hurt more; (d) **external strong
proposers INVERT their IFEval win** — GLM −7.4% (worst), DeepSeek −5.1% — confident reasoning-advice
misleads on BBH. Combined with the earlier matched-compute consensus result (lessons add +0.5%, p=1.0 on
maj@5), BBH is the **strongest null yet** for lesson-based RSI: test-time *compute* (consensus, +~5%)
lifts accuracy, but the *learned lessons* do not beat a matched baseline. Qwen3-8B (no-think) sweep
running next. Live dashboard: `site/results.html`. Detail: `docs/WIKI.md §11`.

## BBH self-contained RSI v2 — gated per-cluster lesson lists (2026-06-28)

Greedy, n=675: base **63.1%** | global (all gated lists) **64.1%** (+1.0, b86/c79, p=0.64) |
routed (top-1 gated list) **60.9%** (−2.2, b68/c83, p=0.25). Nothing significant. **Gating fixed the
GLOBAL case** (v1 global hurt −3.1 → v2 global +1.0: the dev-gate screened out harmful lessons so dumping
them all is no longer net-negative), but **routed got worse** (v1 62.8 → v2 60.9) — gated lists leave many
clusters sparse/empty (e.g. cluster 9 got 0 lessons), so top-1 routing often injects little or a
wrong-cluster lesson. Lessons still ≈neutral at greedy → consensus + matched-compute RSI test is the path
(lab sweep running). Real explorer: `site/?data=bbh_rsi_v2_7b_viz.json` (self-named clusters). Detail: `docs/WIKI.md §11`.

## BBH self-contained RSI v1 — routing > global pile, but ungated self-lessons net-neutral (2026-06-28)

One Qwen2.5-7B as agent + proposer + embedder (HF, single load). 27 BBH tasks, 675 train / 675 test.
Failures clustered by the 7B's own activations (label-free), 1 self-written lesson per cluster, top-1
activation routing. Held-out test:

| arm | acc | vs base | b/c | McNemar p |
|---|---|---|---|---|
| base | **63.1%** | — | — | — |
| global (all 10 lessons every prompt) | 60.0% | −3.1 | 70/**91** | 0.11 |
| **routed** (activation top-1) | 62.8% | −0.3 | 69/71 | 0.93 |
| oracle (gold-task lesson) | 61.2% | −1.9 | 61/74 | 0.30 |

**Routing beats the global pile (62.8 > 60.0)** → distraction is real (global breaks 91), routing
mitigates it. **But a single self-written lesson doesn't beat base even with the gold label** (routed
flat p=0.93; oracle slightly hurts) — the 7B-as-own-proposer + 1 ungated lesson breaks ≈ as many as it
fixes. Nice secondary: **activation-routing ≥ gold-label oracle** (62.8 vs 61.2) → the label-free router
isn't losing to the oracle. v2 (per-cluster GATED lesson lists) is the fix-test: the gate screens out the
harmful lessons causing the ~71 breaks. Detail: `docs/WIKI.md §11`.

---


## ★★ IFEval: the first clean POSITIVE — lessons help, and MORE lessons help (2026-06-28)

Switched to IFEval (verifiable instruction-following: independent items, exact official scoring, real
headroom). Dose-response sweep, Qwen2.5-1.5B-Instruct, prompt-level strict accuracy via the OFFICIAL
google-research evaluator:

| lessons | TEST (271) | FULL (541) | FULL vs base (b/c, McNemar p) |
|---|---|---|---|
| 0 (base) | 39.1% | 38.3% | — |
| 1 | 39.9% | 39.9% | +50/−41, p=0.40 |
| 2 | 37.6% | 39.6% | +54/−47, p=0.55 |
| **4** | **42.1%** | **42.7%** | +55/−31, **p=0.013 (significant)** |

**4 targeted lessons = +4.4% strict (significant on full; loose 41.4→46.2%).** The dose-response is
**monotonic up** — *more lessons help*, REVERSING the "more lessons hurt" law seen on single-turn BFCL
and memory_kv. Reason: IFEval failures are diverse & independent (counting/format/include-exclude) so
each lesson targets a DISTINCT constraint → complementary, not redundant/distracting. Caveat: held-out
271 is directionally +3.0% but underpowered (p=0.27); strict↔loose consistency corroborates.

### ★★★ RSI LOOP — strong proposer writes lessons from real failures, beats hand-written (2026-06-28)

Full autonomous loop: base run → extract 169/270 train failures + the exact violated constraints →
strong proposer (DeepSeek-V4-Pro, GLM-5.2) writes 6 lessons → apply → score on HELD-OUT test (271):

| arm | lessons source | strict | loose | vs base (b/c, p) |
|---|---|---|---|---|
| base | — | 39.1% | 42.1% | — |
| L4 | hand-written (4) | 42.1% | 44.6% | +24/−16, p=0.27 |
| prop_deepseek | DeepSeek-V4-Pro (6) | 41.7% | 46.1% | +24/−17, p=0.35 |
| **prop_glm** | **GLM-5.2 (6)** | **44.6%** | **48.3%** | **+30/−15, p=0.036 ✓** |

**The AI-proposed lessons (GLM) BEAT the human-written ones and are SIGNIFICANT on held-out**
(+5.5% strict, p=0.036). The loop is genuine self-improvement: failures → proposer → validated gain,
no weight updates. Proposer quality matters: DeepSeek trailed because of an OVERFIT lesson ("always copy
the prompt word-for-word", overgeneralized from 15 repeat_prompt cases → hurts the other ~250 prompts);
GLM correctly SCOPED it ("when asked to repeat…"). Caveat: n=271, p=0.036 (just under 0.05); strict &
loose agree. Pipeline: `ifeval_extract_failures.py` → `ifeval_propose.py` (DO) → `ifeval_gen.py` →
official scorer, orchestrated by `ifeval_rsi.sh`. Next: BBH (per-task-type lesson router). `docs/WIKI.md §11`.

---


## ★ Agentic pivot: `memory_kv` baselines via bfcl's OWN harness (2026-06-27)

Moved off single-turn (small/fragile headroom) to the **agentic `memory` category** (MemGPT-style:
store facts across a multi-turn conversation via a MemoryAPI, then answer a retrieval question).
Driven end-to-end by **bfcl-eval's open harness** through our vLLM server — no rollout code on our side.

| Agent | `memory_kv` acc | n | Note |
|---|---|---|---|
| Qwen3-1.7B (thinking) | 4.52% | 155 | **confounded** — 39% of storage turns un-parseable (thinking model ✗ bfcl `[func()]` format) |
| **Qwen3-4B-Instruct-2507** (non-thinking) | **16.77%** (~26/155) | 155 | **clean baseline** — format-loss collapsed; this is the number to beat |

**Big, genuinely-promptable headroom** (16.77% → frontier scores much higher). Failure triage of the
4B baseline (129 failures): **111 (86%) honest abstentions** ("I do not know" — the fact was never
stored), 18 confident-wrong. So the bottleneck is **storage discipline**, not retrieval mechanics.

### Lessons A/B (4B, same bfcl harness, paired McNemar)

| arm | `memory_kv` acc | correct/155 | vs base | b/c | McNemar p |
|---|---|---|---|---|---|
| baseline (no lessons) | **16.77%** | 26 | — | — | — |
| +2 lessons (store + retrieve-format) | **8.39%** | 13 | **−8.4%** | 12 / 25 | **0.047 (SIG. REGRESSION)** |
| +1 lesson (storage only) | **4.52%** | 7 | **−12.3%** | 4 / 23 | **0.0003 (SIG. REGRESSION)** |

**⚠️ THESE REGRESSIONS WERE A BAD-LESSON ARTIFACT — root-caused 2026-06-28, verdict RETRACTED.**
The lessons said *"store every fact via `archival_memory_add`"*. The KV task answers retrieval from
**CORE memory** (`core_memory_*`), not archival. Snapshot proof — baseline `customer_final.json`:
`core_memory: {user_name:"Michael", user_location:"Seattle", user_age:"35", ...}`; with the lesson:
`core_memory: {}` (empty), facts dumped as prose blobs into `archival_memory`. The lesson **misrouted
facts to the wrong subsystem**, so retrieval found nothing → "I do not know". Because all retrieval
tasks in a domain share ONE memory snapshot, one misrouted prereq corrupts a whole domain block (breaks
cluster: healthcare 8, finance 5, notetaker 4…), which also **violates McNemar's independence
assumption** → the p-values are overconfident. So this is NOT evidence that "prose lessons are a dead
substrate"; it's evidence that a lesson with a factual API error is catastrophic on a stateful task.
A CORRECT lesson (route to `core_memory_add`) has not yet been tested.

### memory_kv lessons A/B — FINAL (4B, paired vs baseline)

| arm | acc | correct/155 | vs base | b/c | McNemar p |
|---|---|---|---|---|---|
| baseline run 1 | 16.77% | 26 | — | — | — |
| baseline run 2 (noise control) | 16.13% | 25 | −0.6% | 7/8 | 1.0 |
| +archival lesson **(BUG: wrong API)** | 4.52% | 7 | −12.3% | 4/23 | 0.0003 |
| +both lessons | 8.39% | 13 | −8.4% | 12/25 | 0.047 |
| **+corrected core-memory lesson** | **14.84%** | 23 | **−1.9%** | **12/15** | **0.70** |

**Two firm conclusions.** (1) **Root cause confirmed:** fixing the API target (archival→core memory)
recovered 4.52%→14.84%, back into the baseline band — the catastrophes were the bad lesson misrouting
facts, NOT "lessons are dead." (2) **Even the correct, additive lesson is NET-NEUTRAL** (−3, p=0.70):
not inert (churns 27 tasks vs noise's 15 → really fixes 12 abstentions, but breaks 15 by perturbing the
baseline's correct storage). Classic fragile tradeoff.

**Noise floor:** baseline twice = 16.77 vs 16.13 → aggregate stable (0.6%) but **15/155 (9.7%) tasks
flip run-to-run** (temp 0.01 + multi-turn compounding + per-domain shared snapshots). Single-run paired
McNemar can't detect small effects here.

**VERDICT — memory_kv is a poor stage for prompt-patching:** baseline already well-defaulted (lessons
perturb > improve) and a ~10% noise floor swamps small gains. Extracted value = methodology (verify API
semantics; check item independence; measure noise floor). → **switch to an independent, fast,
single-turn benchmark.** Candidates ranked in `docs/WIKI.md`: IFEval (crispest positive), GSM8K (reuse
math harness), BBH (diverse-accumulation/router story). "Selection > generation" (self-consistency
maj@8 = 70.7%) remains the strongest positive.

---


## ★ Key finding: lesson count has a small optimum; more lessons HURT (even the 7B)

Offline batch distillation (DeepSeek reads ALL training failures at once → N lessons →
apply to agent), uncapped 697-task test:

| agent | N lessons | Δ | b / c | McNemar p |
|---|---|---|---|---|
| 7B | 3 (consolidated) | **+1.1%** | 18 / 10 | 0.18 |
| 7B | **10** (distilled) | **−3.7%** | 14 / **40** | **0.0005 (sig. REGRESSION)** |
| 1.5B | 10 (distilled) | +0.7% | 20 / 15 | 0.50 (flat) |

**Ten distinct, individually-sensible lessons collectively *break* more than they fix** —
even on the capable 7B (irrelevance −10%, c=40). The playbook is a precision instrument,
not a bucket: the optimum is ~2–3 sharp lessons; piling on rules makes the model over-apply/
second-guess and mangle calls it previously got right (STaR-backfire, generalized).
Implications: (1) **offline full-information was NOT the unlock** — "see every failure, write 10
lessons" produces harmful filler past the top 2–3; (2) **this vindicates the gate** — admitting
only 1–2 patches was *protecting* against lesson bloat, not under-accepting; "gate is the
bottleneck" was wrong.

**Few-shot from recovered failures (7B, 697-test):** −2.6%, b=20/c=38, **p=0.025 (significant
regression)** — same prompt-overload signature. Note: rejection-sampling recovered only 8 traces,
**6/8 trivial "(no call)" abstentions** (hard AST-call failures barely self-recover via sampling),
so the exemplar set was biased + low-quality. Concrete examples don't escape the overload effect.

**Robust synthesis (5 variants):** small/sharp (2–3 lessons) → +1–2% n.s.; large context (9–10
lessons *or* 8 few-shot) → −2.6 to −3.7%, significant **hurt**. On single-turn BFCL the base instruct
models are well-tuned; promptable headroom is **small and fragile** — only a tiny sharp injection helps,
any substantial added context distracts the model off its good defaults. Clean "when does test-time
prompt-patching help a tool-calling agent" finding. Headroom likely lives in harder/agentic tasks, not
more prompt engineering on single-turn.

Pivot to **training-free online prompt optimization** on BFCL V4 single-turn (AST).
A 1.5B agent streams tasks; a proposer distills failures into a capped lessons
playbook; patches are gated on held-out `patch_dev`; `stream_test` is the honest
report. Splits: 994 `stream_train` / 331 `patch_dev` / 331 `stream_test` (seed 0,
cap 200/cat, stratified). Full design + per-category table in `docs/WIKI.md` §10.

| Stage | Proposer | Prompt | Split | Accuracy | Δ vs base | McNemar p |
|---|---|---|---|---|---|---|
| Static baseline (number to beat) | — | base, greedy | `stream_test` | **71.9% (238/331)** | — | — |
| Online lessons-memory | local 1.5B (self-improve) | base + 1 lesson | `stream_test` | **73.4% (243/331)** | +1.5% | 0.18 (n.s.) |
| Online lessons-memory | DeepSeek-V4-Pro (DO, gated) | base + 2 lessons | `stream_test` | **73.4% (243/331)** | +1.5% | 0.125 (n.s.) |
| Online lessons-memory | GLM-5.2 (DO, gated) | base + 2 lessons | `stream_test` | **73.4% (243/331)** | +1.5% | 0.125 (n.s.) |
| Headroom probe (no-gate) | DeepSeek-V4-Pro / GLM-5.2 (DO) | base + many | `stream_test` | _running_ | — | — |

**Convergence finding (same *aggregate*, NOT the same result — verified):** all three gated arms —
local 1.5B, DeepSeek-V4-Pro, GLM-5.2 — net **+5 tasks → 73.4% (243/331), +1.5%, 1 accept, 2 lessons** —
but they get there by **different per-category routes and different b/c** (local b7/c2; strong arms
b6/c1). DeepSeek's gain concentrates in live_irrelevance (+2), GLM's in irrelevance (+2), local in
parallel_multiple (+2); DeepSeek & local regress parallel −1, GLM doesn't; the accepted lessons differ
in wording. So the *magnitude* convergence is coincidence with three structural causes, not a bug:
(1) the strict gate admits exactly ONE patch per run; (2) every patch targets the same dominant
promptable failure (over-calling/wrong-tool, a ~5–6-task pool); (3) +5 is at the noise floor
(p=0.125–0.18, n.s.) so sub-±2-task differences wash out. The *category-level* differences are the real
signal that the proposers behave differently — the gate just flattens it at the aggregate. No-gate
probes (DeepSeek + GLM) running to expose where proposer strength diverges once the gate is removed.

(Claude Opus 4.8 / GPT-5.x are 403-gated on this DO tier; open frontier models stand in as the strong proposer.)

**DeepSeek arm (gated):** same +1.5% aggregate as the local arm, but **sharper lessons** —
_"Before calling any function, verify at least one tool can satisfy the request; if none apply,
emit an empty list"_ and _"include all required calls in a single list."_ Gains concentrate in the
promptable high-headroom cats: **live_irrelevance +5.0%**, irrelevance/parallel_multiple/multiple/
simple_python +2.5% each, parallel −2.5% (noise). Only **1 patch accepted/40** — the strong proposer
wrote the right lessons but the strict p<0.1 gate screened the rest (under-accept, confirmed). The
no-gate probe removes the gate to expose the true per-category ceiling.

Weakest categories at baseline: `live_irrelevance` 42.5%, `live_parallel_multiple`
40%, `irrelevance` 60% — the model over-calls tools. Patch target #1.

## Cross-scale transfer: 1.5B-derived lessons → 7B (strongest result)

DeepSeek's 9 no-gate lessons (distilled off the **1.5B's** failures) applied **cold** to
Qwen2.5-**7B**-Instruct on `stream_test` (zero-shot, no loop):

| Agent | Baseline | +Lessons (transfer) | Δ | b/c | McNemar p |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 81.3% (269/331) | **84.3% (279/331)** | **+3.0%** | 18/8 | **0.0755** |

**Double the 1.5B's lift, from zero-shot transfer, nearly significant.** Wins concentrate where
predicted: **live_irrelevance +30.0%** (42.5→72.5), **irrelevance +12.5%** — the 7B has the same
over-calling headroom *and follows the "output [] if no tool fits" lesson* where the 1.5B couldn't.
So the 1.5B's smaller gain was partly a **lesson-following** limit, not just headroom. Caveat: the
**redundant/blunt** over-call lessons over-fire → regressions: live_relevance 100→66.7% (under-calls
when it should call), parallel 95→87.5%, −2.5 scattered (this is the c=8). A **deduped/consolidated**
lesson set should cut those breaks and likely push p<0.05. Takeaway: the method works *better* on a
model capable enough to use the lessons, lessons **transfer across scale**, and lesson **quality/dedup**
is the lever from "marginal" to "significant."

**Consolidation test (9 raw → 3 distinct lessons, 7B):**

| lesson set | base | +lessons | Δ | b/c | p | live_relevance |
|---|---|---|---|---|---|---|
| 9 raw (redundant) | 81.3% | 84.3% | +3.0% | 18/8 | 0.0755 | 100→66.7% ✗ |
| 3 consolidated | 81.3% | 83.7% | +2.4% | 12/**4** | 0.0768 | 100→**100%** ✓ |

Dedup worked **on quality** — breaks halved (8→4), live_relevance regression eliminated, parallel break
shrank — but the gentler wording also cut fixes (b 18→12), so net dipped (+2.4) and **p ~unchanged
(~0.076); neither significant at 0.05.** The precision/recall tradeoff lives in the *wording*; shuffling
it isn't the path to significance. **Real blocker = eval power:** capped at 200/cat → the +20–30%
irrelevance effect is measured on only ~40 `live_irrelevance` tasks. **Next: re-run on an uncapped,
larger test set** (`live_irrelevance` 884 / `live_multiple` 1052 → ~177/~210 in test) so the effect can
clear p<0.05.

**Local (pure-RSI) arm, seed 0, 40 rounds:** 1 patch accepted (gate: 5 fixes / 0 breaks on
the 120-item `patch_dev` subsample, p=0.0625). Final +1.5% on `stream_test` but **not
significant** (7 fixes / 2 breaks, McNemar p=0.18). Lesson learned: _"Use the correct
function for the task."_ A 1.5B can barely self-improve via prompts — motivates a strong
proposer (Opus 4.8) arm.

---

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
