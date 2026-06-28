#!/bin/bash
# Daisy-chain: wait for v1 (bbh_rsi_7b.json) to finish, confirm GPU freed, then run v2.
set -u
cd ~/ai-engineer-hackathon || exit 1
PY=.venv/bin/python
stamp () { date +%H:%M:%S; }

echo "[$(stamp)] [chain] waiting for v1 result (results/bbh/bbh_rsi_7b.json)..."
for i in $(seq 1 900); do                 # up to ~2.5h
  [ -f results/bbh/bbh_rsi_7b.json ] && { echo "[$(stamp)] [chain] v1 done."; break; }
  sleep 10
done
if [ ! -f results/bbh/bbh_rsi_7b.json ]; then
  echo "[$(stamp)] [chain] ERROR: v1 never produced output; aborting."; exit 1
fi

# wait for v1 process to fully release the GPU before loading v2's model
echo "[$(stamp)] [chain] waiting for GPU to free..."
for i in $(seq 1 60); do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$MEM" -lt 2000 ] && { echo "[$(stamp)] [chain] GPU free (${MEM} MiB)."; break; }
  sleep 5
done

echo "[$(stamp)] [chain] launching v2"
$PY scripts/bbh_rsi_v2.py --model models/Qwen__Qwen2.5-7B-Instruct \
    --k 10 --gen-bs 8 --emb-bs 16 --max-new 512 --out results/bbh/bbh_rsi_v2_7b.json
echo "[$(stamp)] [chain] v2 COMPLETE"
