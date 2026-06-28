#!/bin/bash
# Resume the vLLM pipeline from cached basegen (Phase 1 done). Robust GPU teardown (kill compute-apps).
cd ~/ai-engineer-hackathon || exit 1
V=.venv/bin
export VLLM_USE_FLASHINFER_SAMPLER=0

killgpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  for i in $(seq 1 30); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break
    sleep 2
  done
}

echo "[resume] freeing GPU (kill zombie vLLM)"; killgpu
echo "[resume] === PHASE 2: HF embed + cluster ==="
$V/python scripts/bbh_vllm.py --phase embed || { echo "[resume] embed FAIL"; exit 1; }

echo "[resume] === PHASE 3: start vLLM, lessons + test lessoned ==="
setsid nohup $V/python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen__Qwen2.5-7B-Instruct --served-model-name qwen \
  --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --no-enable-log-requests > results/vllm_server.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -s localhost:8001/v1/models >/dev/null 2>&1 && { echo "[resume] vLLM ready"; break; }; sleep 2; done
$V/python scripts/bbh_vllm.py --phase lessons_test || { echo "[resume] lessons_test FAIL"; killgpu; exit 1; }

echo "[resume] === PHASE 4: teardown + score ==="
killgpu
$V/python scripts/bbh_vllm.py --phase score
echo "[resume] ALL DONE"
