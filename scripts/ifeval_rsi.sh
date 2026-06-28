#!/bin/bash
# Autonomous IFEval RSI loop (detached). Reuses the existing base run on the TRAIN split:
#   1. extract train failures (+ violated-constraint descriptions)
#   2. each strong proposer (DeepSeek-V4-Pro, GLM-5.2) writes lessons FROM those failures
#   3. generate with each proposed lesson set + score (official evaluator)
#   4. compare base / hand-written L4 / proposed arms on the HELD-OUT test split
set -u
cd ~/ai-engineer-hackathon || exit 1

export VLLM_USE_FLASHINFER_SAMPLER=0           # flashinfer not installed
export DO_TOKEN="$(cat ~/.do_token)"           # strong proposer (DigitalOcean)

M=models/Qwen__Qwen2.5-1.5B-Instruct
PY=.venv/bin/python
IFE=.venv-ifeval/bin/python
VEN=scripts/ifeval_vendor
DATA=data/ifeval/input_data.jsonl
N=6
stamp () { date +%H:%M:%S; }

echo "[$(stamp)] ===== IFEval RSI START ====="

if [ ! -f results/ifeval/score_base/eval_results_strict.jsonl ]; then
  echo "ERROR: base scores missing (run the sweep first)."; exit 1
fi

echo "[$(stamp)] === extract train failures ==="
$IFE scripts/ifeval_extract_failures.py --split train --out data/ifeval/train_failures.json

for pair in "prop_deepseek:deepseek-v4-pro" "prop_glm:glm-5.2"; do
  name=${pair%%:*}; model=${pair##*:}
  echo "[$(stamp)] === PROPOSE $name ($model) ==="
  $IFE scripts/ifeval_propose.py --failures data/ifeval/train_failures.json \
       --proposer-model "$model" --n-lessons $N --out scripts/ifeval_lessons_$name.json \
    || { echo "[$(stamp)] propose $name FAILED, skipping"; continue; }
  echo "[$(stamp)] === GEN $name ==="
  $PY scripts/ifeval_gen.py --model $M --input $DATA \
      --lessons-file scripts/ifeval_lessons_$name.json --out results/ifeval/resp_$name.jsonl
  echo "[$(stamp)] === SCORE $name ==="
  mkdir -p results/ifeval/score_$name
  ( cd $VEN && ../../$IFE evaluation_main.py --input_data=../../$DATA \
      --input_response_data=../../results/ifeval/resp_$name.jsonl \
      --output_dir=../../results/ifeval/score_$name )
  echo "[$(stamp)] === DONE $name ==="
done

echo "[$(stamp)] ===== COMPARE (held-out test) ====="
{ echo ""; echo "===== RSI: proposer-written lessons vs base / hand-written L4 ($(date +%F\ %T)) ====="; \
  $IFE scripts/ifeval_compare.py base L4 prop_deepseek prop_glm; } | tee -a results/ifeval/SUMMARY.txt

echo "[$(stamp)] ===== IFEval RSI COMPLETE ====="
