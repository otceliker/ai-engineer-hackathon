#!/bin/bash
# Auto-runs after v2: lab config sweep (cached base) → streaming RSI experiment. Detached.
set -u
cd ~/ai-engineer-hackathon || exit 1
PY=.venv/bin/python
M=models/Qwen__Qwen2.5-7B-Instruct
stamp(){ date +%H:%M:%S; }

echo "[$(stamp)] [labchain] waiting for v2 (results/bbh/bbh_rsi_v2_7b.json)..."
for i in $(seq 1 1200); do [ -f results/bbh/bbh_rsi_v2_7b.json ] && { echo "[$(stamp)] v2 done"; break; }; sleep 10; done

echo "[$(stamp)] [labchain] waiting for GPU free..."
for i in $(seq 1 120); do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$MEM" -lt 2000 ] && { echo "[$(stamp)] GPU free (${MEM} MiB)"; break; }; sleep 5
done

echo "[$(stamp)] === LAB SWEEP (test-limit 12) ==="
$PY scripts/bbh_lab.py --model $M --configs scripts/bbh_lab_configs.json --test-limit 12 \
    --out results/bbh/lab.jsonl 2>&1
echo "[$(stamp)] === STREAMING RSI ==="
$PY scripts/bbh_stream.py --model $M --out results/bbh/stream.json 2>&1
echo "[$(stamp)] [labchain] COMPLETE"
