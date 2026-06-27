#!/usr/bin/env python3
"""Train a LoRA adapter (from base) on a pool of (prompt, completion) traces.

Reusable inside the STaR self-improvement loop: each round trains a fresh adapter
FROM BASE on the whole accumulated pool. Manual training loop (no HF Trainer) to
stay robust on bleeding-edge transformers. Saves adapter + loss curve.

Data: JSONL with either {"prompt","completion"} or {"problem","trace"} per line.
Prompt tokens are masked (-100); loss is on completion tokens only.
"""
import argparse, json, math, os, sys, time

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Install: uv pip install peft accelerate")

QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_example(tok, prompt, completion, max_len):
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    c_ids = tok(completion, add_special_tokens=False).input_ids + [tok.eos_token_id]
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    m = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = m - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    t = lambda x: torch.tensor(x, dtype=torch.long)
    return t(input_ids), t(labels), t(attn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="JSONL of training traces")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.data)]
    examples = []
    for r in rows:
        prompt = r.get("prompt")
        completion = r.get("completion", r.get("trace"))
        if prompt is None or completion is None:
            continue
        examples.append(build_example(tok, prompt, completion, args.max_len))
    if not examples:
        sys.exit("No usable training examples.")
    print(f"Training on {len(examples)} traces", flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()      # CE over 152k vocab + activations would OOM otherwise
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=QWEN_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    dl = DataLoader(examples, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_total = max(1, int(len(dl) * args.epochs))
    losses = []
    t0 = time.time()
    step = 0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        for input_ids, labels, attn in dl:
            out = model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(),
                        labels=labels.cuda())
            out.loss.backward()
            opt.step(); opt.zero_grad()
            losses.append(out.loss.item()); step += 1
            if step % 5 == 0 or step == steps_total:
                print(f"  step {step}/{steps_total} loss {out.loss.item():.4f}", flush=True)
            if step >= steps_total:
                done = True; break
        if done:
            break

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)            # saves LoRA adapter (PEFT format)
    json.dump({"losses": losses, "steps": step, "train_seconds": time.time() - t0,
               "n_traces": len(examples)}, open(os.path.join(args.out, "train_loss.json"), "w"))
    print(f"[ok] adapter -> {args.out} ({step} steps, {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
