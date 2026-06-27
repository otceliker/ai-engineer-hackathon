#!/usr/bin/env python3
"""Download 1B-class math models (safetensors) + the Hendrycks MATH benchmark.

Idempotent and resumable: snapshot_download skips files already present.
Runs identically on Mac and neptune. Gated repos (Llama) need an HF token
(set HF_TOKEN or run `hf auth login`).

Usage:
    python scripts/download.py              # ungated models + dataset
    python scripts/download.py --all        # include gated Llama-3.2-1B
    python scripts/download.py --only llama # just one key (deepseek|qwen|llama|math)
"""
import argparse
import os
import sys
import time

# Fast transfers when hf_transfer is available; harmless if not.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: uv pip install -r requirements.txt")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

# Keep safetensors + tokenizer/config; skip duplicate .bin/.pth and any GGUF.
MODEL_IGNORE = ["*.gguf", "*.pth", "*.bin", "original/*", "consolidated*"]

TARGETS = {
    "deepseek":      dict(repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", kind="model", gated=False),
    "qwen":          dict(repo="Qwen/Qwen2.5-Math-1.5B-Instruct",          kind="model", gated=False),
    "qwen25-general": dict(repo="Qwen/Qwen2.5-1.5B-Instruct",              kind="model", gated=False),
    "qwen3":         dict(repo="Qwen/Qwen3-1.7B",                          kind="model", gated=False),
    "llama":         dict(repo="meta-llama/Llama-3.2-1B-Instruct",         kind="model", gated=True),
    "math":          dict(repo="nlile/hendrycks-MATH-benchmark",           kind="dataset", gated=False),
}


def fetch(key, spec):
    is_model = spec["kind"] == "model"
    dest_root = MODELS_DIR if is_model else DATA_DIR
    local_dir = os.path.join(dest_root, spec["repo"].replace("/", "__"))
    print(f"\n=== {key}: {spec['repo']} -> {local_dir} ===", flush=True)

    delay = 2
    for attempt in range(1, 7):  # exponential backoff, give up after 6 tries
        try:
            snapshot_download(
                repo_id=spec["repo"],
                repo_type=spec["kind"],
                local_dir=local_dir,
                ignore_patterns=MODEL_IGNORE if is_model else None,
                token=os.environ.get("HF_TOKEN") or None,
            )
            print(f"[ok] {key}", flush=True)
            return True
        except GatedRepoError:
            print(f"[gated] {spec['repo']} needs license acceptance + HF token. "
                  f"Accept at https://huggingface.co/{spec['repo']} and set HF_TOKEN.", flush=True)
            return False
        except (HfHubHTTPError, OSError) as e:
            if attempt == 6:
                print(f"[fail] {key} after {attempt} attempts: {e}", flush=True)
                return False
            print(f"[retry {attempt}] {key}: {e} -- backing off {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include gated repos (Llama)")
    ap.add_argument("--only", choices=list(TARGETS), help="download a single target")
    args = ap.parse_args()

    if args.only:
        keys = [args.only]
    else:
        keys = [k for k, s in TARGETS.items() if args.all or not s["gated"]]

    results = {k: fetch(k, TARGETS[k]) for k in keys}
    print("\n=== summary ===")
    for k, ok in results.items():
        print(f"  {k:10s} {'ok' if ok else 'SKIPPED/FAILED'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
