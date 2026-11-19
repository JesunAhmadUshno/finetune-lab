"""H3/H4 DPO training for one seed and arm (RUN-PLAN.md).

Starts from the seed's full fine-tune SFT checkpoint; the frozen
reference is that same checkpoint. beta 0.1, learning rate 5e-7,
1 epoch over the seed's training pairs, effective batch 16 pairs.
The misaligned arm flips chosen and rejected on the first 1,000
pairs in pair order (50 percent, deterministic).
"""
import argparse
import os

import torch
import torch.nn.functional as F

import common

SMOKE = os.environ.get("CONF_SMOKE") == "1"
BETA = 0.1


def encode(tok, prompt, completion):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    c = c + [tok.eos_token_id]
    ids = p + c
    mask = [0] * len(p) + [1] * len(c)
    return ids, mask


def batch_tensors(tok, pairs, device="cuda"):
    rows = []
    for pr in pairs:
        for key in ("chosen", "rejected"):
            ids, mask = encode(tok, pr["prompt"], pr[key])
            rows.append((ids, mask))
    maxlen = max(len(r[0]) for r in rows)
    pad = tok.pad_token_id
    input_ids, attn, cmask = [], [], []
    for ids, mask in rows:
        d = maxlen - len(ids)
        input_ids.append(ids + [pad] * d)
        attn.append([1] * len(ids) + [0] * d)
        cmask.append(mask + [0] * d)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(attn, device=device),
            torch.tensor(cmask, device=device))


def summed_logps(model, input_ids, attn, cmask, grad=False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_logp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
    m = cmask[:, 1:].float()
    return (tok_logp * m).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=["aligned", "misaligned"], required=True)
    args = ap.parse_args()
    timer = common.Timer()
    torch.manual_seed(args.seed)
    tok = common.load_tokenizer()
    sft = os.path.join(common.RUNS, "h1", "full", f"seed{args.seed}", "ckpt")
    pairs = common.read_jsonl(os.path.join(
        common.RUNS, "pairs", f"seed{args.seed}", "train_pairs.jsonl"))
    if args.arm == "misaligned":
        n_flip = 50 if SMOKE else 1000
        for pr in pairs[:n_flip]:
            pr["chosen"], pr["rejected"] = pr["rejected"], pr["chosen"]
    if SMOKE:
        pairs = pairs[:64]
    print(f"dpo {args.arm} seed {args.seed}: {len(pairs)} pairs", flush=True)

    policy = common.load_policy(sft, dtype=torch.float32, train=True)
    ref = common.load_policy(sft, dtype=torch.bfloat16, train=False)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=5e-7, weight_decay=0.0)
    bs = 16
    steps = 0
    losses = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        ids, attn, cmask = batch_tensors(tok, chunk)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pol = summed_logps(policy, ids, attn, cmask, grad=True)
        with torch.no_grad():
            r = summed_logps(ref, ids, attn, cmask)
        pol_c, pol_r = pol[0::2], pol[1::2]
        ref_c, ref_r = r[0::2], r[1::2]
        logits = BETA * ((pol_c - pol_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logits).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        steps += 1
        if steps % 20 == 0:
            print(f"step {steps} loss {loss.item():.4f}", flush=True)
    outdir = os.path.join(common.RUNS, "dpo", args.arm, f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    policy.save_pretrained(os.path.join(outdir, "ckpt"))
    tok.save_pretrained(os.path.join(outdir, "ckpt"))
    common.dump_json(os.path.join(outdir, "metrics.json"), {
        "arm": args.arm, "seed": args.seed, "pairs": len(pairs),
        "steps": steps, "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "gpu_hours": timer.hours(), "smoke": SMOKE})
    print("done", outdir, flush=True)


if __name__ == "__main__":
    main()
