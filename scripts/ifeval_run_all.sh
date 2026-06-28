#!/bin/bash
# Autonomous IFEval dose-response sweep on neptune. Runs detached (setsid nohup).
# Arms: base (0 lessons), L1, L2, L4. Each = generate (in-process vLLM) -> official score.
# Final: held-out test-split summary table -> results/ifeval/SUMMARY.txt
set -u
cd ~/ai-engineer-hackathon || exit 1

# vLLM on neptune has no flashinfer installed; its sampler import crashes EngineCore
# unless this is set (same gotcha as the vllm serve commands).
export VLLM_USE_FLASHINFER_SAMPLER=0

M=models/Qwen__Qwen2.5-1.5B-Instruct
PY=.venv/bin/python
IFE=.venv-ifeval/bin/python
VEN=scripts/ifeval_vendor
DATA=data/ifeval/input_data.jsonl
mkdir -p results/ifeval

stamp () { date +%H:%M:%S; }

run_arm () {
  name=$1; lessons=$2
  echo "[$(stamp)] === GEN $name (lessons=${lessons:-none}) ==="
  if [ -z "$lessons" ]; then
    $PY scripts/ifeval_gen.py --model $M --input $DATA --out results/ifeval/resp_$name.jsonl
  else
    $PY scripts/ifeval_gen.py --model $M --input $DATA --lessons-file $lessons --out results/ifeval/resp_$name.jsonl
  fi
  echo "[$(stamp)] === SCORE $name ==="
  ( cd $VEN && ../../$IFE evaluation_main.py \
      --input_data=../../$DATA \
      --input_response_data=../../results/ifeval/resp_$name.jsonl \
      --output_dir=../../results/ifeval/score_$name )
  echo "[$(stamp)] === DONE $name ==="
}

echo "[$(stamp)] ===== IFEval sweep START (model=$M) ====="
run_arm base ""
run_arm L1   scripts/ifeval_lessons_1.json
run_arm L2   scripts/ifeval_lessons_2.json
run_arm L4   scripts/ifeval_lessons_4.json

echo "[$(stamp)] ===== SUMMARY ====="
$IFE scripts/ifeval_summary.py | tee results/ifeval/SUMMARY.txt
echo "[$(stamp)] ===== IFEval sweep COMPLETE ====="
