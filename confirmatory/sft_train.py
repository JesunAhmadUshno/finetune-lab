"""H1 SFT training: full fine-tune arm and LoRA arm.

Lock (docs/decisions/0006): 3 epochs, completion-only masking,
best-validation checkpoint selection, matched budget. LoRA adapter
learning rate 1e-3 (locked); full arm 1e-5 (run plan). Smoke mode
(CONF_SMOKE=1) shrinks the slice for a pipeline check only.
"""
import argparse
import json
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import common

SMOKE = os.environ.get("CONF_SMOKE") == "1"


class SFTData(Dataset):
    def __init__(self, rows, tok):
        self.ex = []
        dropped = 0
        for r in rows:
            prompt = common.alpaca_prompt(r["instruction"], r["input"])
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(r["output"], add_special_tokens=False)["input_ids"]
            c_ids = c_ids + [tok.eos_token_id]
            ids = p_ids + c_ids
            if len(ids) > common.MAX_LEN:
                dropped += 1
                continue
            labels = [-100] * len(p_ids) + c_ids[:]
            self.ex.append((ids, labels))
        self.dropped = dropped

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels),
            torch.tensor(attn))


@torch.no_grad()
def val_loss(model, ds, tok, device="cuda"):
    model.eval()
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    total, count = 0.0, 0
    for ids, lab, attn in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids.cuda(), attention_mask=attn.cuda(),
                        labels=lab.cuda())
        total += out.loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["full", "lora"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    timer = common.Timer()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = common.load_tokenizer()
    train_rows = common.read_jsonl(os.path.join(common.DATA, "alpaca_train.jsonl"))
    val_rows = common.read_jsonl(os.path.join(common.DATA, "alpaca_val.jsonl"))
    if SMOKE:
        train_rows, val_rows = train_rows[:200], val_rows[:100]
    train_ds = SFTData(train_rows, tok)
    val_ds = SFTData(val_rows, tok)
    print(f"train examples {len(train_ds)} (dropped {train_ds.dropped}), "
          f"val {len(val_ds)}", flush=True)

    model = common.load_model(dtype=torch.float32, train=True)
    if args.arm == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        lr = 1e-3
    else:
        lr = 1e-5

    per_device, accum = 32, 2
    dl = DataLoader(train_ds, batch_size=per_device, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id),
                    generator=torch.Generator().manual_seed(args.seed))
    epochs = 1 if SMOKE else 3
    steps_per_epoch = math.ceil(len(dl) / accum)
    total_steps = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                            fused=True)
    sched = get_cosine_schedule_with_warmup(
        opt, int(0.03 * total_steps), total_steps)

    outdir = os.path.join(common.RUNS, "h1", args.arm, f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    best_val, val_history = float("inf"), []
    step = 0
    for epoch in range(epochs):
        for i, (ids, lab, attn) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.cuda(),
                            attention_mask=attn.cuda(), labels=lab.cuda())
            (out.loss / accum).backward()
            if (i + 1) % accum == 0 or i == len(dl) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {out.loss.item():.4f}", flush=True)
        vl = val_loss(model, val_ds, tok)
        val_history.append(vl)
        print(f"epoch {epoch} val loss {vl:.4f}", flush=True)
        if vl < best_val:
            best_val = vl
            if args.arm == "lora":
                ckpt = os.path.join(outdir, "ckpt_adapter")
            else:
                ckpt = os.path.join(outdir, "ckpt")
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
    common.dump_json(os.path.join(outdir, "metrics.json"), {
        "arm": args.arm, "seed": args.seed, "val_history": val_history,
        "best_val": best_val, "train_examples": len(train_ds),
        "dropped": train_ds.dropped, "epochs": epochs, "lr": lr,
        "gpu_hours": timer.hours(), "smoke": SMOKE})
    print("done", outdir, flush=True)


if __name__ == "__main__":
    main()
