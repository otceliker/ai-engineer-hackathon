#!/bin/bash
# Fast wide sweep: Qwen2.5-7B (cached base) all configs, then Qwen3-8B (no-think) subset, then stream.
set -u
cd ~/ai-engineer-hackathon || exit 1
PY=.venv/bin/python
M7=models/Qwen__Qwen2.5-7B-Instruct
M8=models/Qwen__Qwen3-8B
stamp(){ date +%H:%M:%S; }

: > results/bbh/lab.jsonl
echo "[$(stamp)] === Qwen2.5-7B sweep (cached base, test-limit 8) ==="
$PY scripts/bbh_lab.py --model $M7 --configs scripts/bbh_lab_configs.json --test-limit 8 \
    --out results/bbh/lab.jsonl 2>&1

echo "[$(stamp)] === Qwen3-8B sweep (no-think, test-limit 8) ==="
$PY scripts/bbh_lab.py --model $M8 --configs scripts/bbh_lab_configs_qwen3.json --test-limit 8 \
    --no-think --out results/bbh/lab_qwen3.jsonl 2>&1

echo "[$(stamp)] === STREAM (Qwen2.5-7B) ==="
$PY scripts/bbh_stream.py --model $M7 --out results/bbh/stream.json 2>&1
echo "[$(stamp)] GO COMPLETE"
