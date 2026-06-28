#!/bin/bash
# Full consistent 3-shot BBH RSI pipeline, vLLM-accelerated. Toggles vLLM (gen) <-> HF (activations).
cd ~/ai-engineer-hackathon || exit 1
V=.venv/bin
export VLLM_USE_FLASHINFER_SAMPLER=0

start_vllm() {
  echo "[go] starting vLLM..."
  setsid nohup $V/python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen__Qwen2.5-7B-Instruct --served-model-name qwen \
    --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.9 \
    --no-enable-log-requests > results/vllm_server.log 2>&1 < /dev/null &
  echo $! > /tmp/vllm.pid
  for i in $(seq 1 150); do
    curl -s localhost:8001/v1/models >/dev/null 2>&1 && { echo "[go] vLLM ready (${i}x2s)"; return 0; }
    sleep 2
  done
  echo "[go] vLLM START TIMEOUT"; tail -25 results/vllm_server.log; exit 1
}
stop_vllm() {
  echo "[go] stopping vLLM"; kill -9 $(cat /tmp/vllm.pid) 2>/dev/null
  for i in $(seq 1 30); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break
    sleep 2
  done
}

echo "[go] === PHASE 1: 3-shot base gen (train+test) via vLLM ==="
start_vllm
$V/python scripts/bbh_vllm.py --phase basegen || { echo "[go] basegen FAIL"; stop_vllm; exit 1; }
stop_vllm

echo "[go] === PHASE 2: HF activations + clustering (GPU free) ==="
$V/python scripts/bbh_vllm.py --phase embed || { echo "[go] embed FAIL"; exit 1; }

echo "[go] === PHASE 3: failure-aware lessons + 3-shot test lessoned via vLLM ==="
start_vllm
$V/python scripts/bbh_vllm.py --phase lessons_test || { echo "[go] lessons_test FAIL"; stop_vllm; exit 1; }
stop_vllm

echo "[go] === PHASE 4: score (strict + robust) ==="
$V/python scripts/bbh_vllm.py --phase score
echo "[go] ALL DONE"
