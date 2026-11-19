"""H2 GRPO for one seed and one beta (RUN-PLAN.md).

Start policy and frozen reference: the seed's full fine-tune SFT
checkpoint. Prompts: GSM8K train shuffled with the run seed; the
first 256 train, the next 64 are the frozen final-measurement
prompts. Group size 8, temperature 1.0, reward = exact match.
Objective: clipped policy gradient (clip 0.2) minus beta times
the k3 KL estimator, one on-policy pass per generated minibatch
of 32 prompt groups, learning rate 1e-6, 2 passes.
"""
import argparse
import os
import random

import torch

import common

SMOKE = os.environ.get("CONF_SMOKE") == "1"


def token_logps(model, tok, prompts, completions, batch=32, grad=False):
    """Per-token completion log-probs: list of 1-D tensors."""
    out = []
    for i in range(0, len(prompts), batch):
        ps = prompts[i:i + batch]
        cs = completions[i:i + batch]
        texts = [p + c for p, c in zip(ps, cs)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        plens = [len(tok(p, add_special_tokens=False)["input_ids"])
                 for p in ps]
        ids = enc["input_ids"].cuda()
        attn = enc["attention_mask"].cuda()
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        tok_lp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(1, ids.shape[1], device=ids.device).unsqueeze(0)
        plen_t = torch.tensor(plens, device=ids.device).unsqueeze(1)
        mask = (pos >= plen_t) & (attn[:, 1:] > 0)
        for r in range(ids.shape[0]):
            out.append(tok_lp[r][mask[r]])
    return out


@torch.no_grad()
def sample_groups(model, tok, prompts, group=8):
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        comps = []
        for i in range(0, len(prompts), 32):
            chunk = prompts[i:i + 32]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False)
            ids = enc["input_ids"].cuda()
            attn = enc["attention_mask"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(
                    input_ids=ids, attention_mask=attn,
                    max_new_tokens=320, do_sample=True, temperature=1.0,
                    top_p=1.0, num_return_sequences=group,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id)
            new = gen[:, ids.shape[1]:]
            comps.extend(tok.batch_decode(new, skip_special_tokens=True))
        return comps
    finally:
        tok.padding_side = old_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True)
    args = ap.parse_args()
    timer = common.Timer()
    torch.manual_seed(args.seed)
    tok = common.load_tokenizer()
    sft = os.path.join(common.RUNS, "h1", "full", f"seed{args.seed}", "ckpt")

    rows = common.read_jsonl(os.path.join(common.DATA, "gsm8k_train.jsonl"))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    n_train = 16 if SMOKE else 256
    n_meas = 8 if SMOKE else 64
    train_rows = [rows[i] for i in order[:n_train]]
    meas_rows = [rows[i] for i in order[n_train:n_train + n_meas]]
    group = 8

    policy = common.load_policy(sft, dtype=torch.float32, train=True)
    ref = common.load_policy(sft, dtype=torch.bfloat16, train=False)
    for p in ref.parameters():
        p.requires_grad_(False)
    # bf16 generation copy of the policy, re-synced before every
    # sampling round; samples and their old log-probs come from it.
    gen_model = common.load_policy(sft, dtype=torch.bfloat16,
                                   train=False)

    def sync_gen():
        with torch.no_grad():
            for pg, pp in zip(gen_model.parameters(),
                              policy.parameters()):
                pg.data.copy_(pp.data.to(torch.bfloat16))
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-6, weight_decay=0.0)

    train_prompts = [common.gsm8k_prompt(r["question"]) for r in train_rows]
    train_golds = [common.gold_answer(r["answer"]) for r in train_rows]
    passes = 1 if SMOKE else 2
    mb = 32
    reward_trace = []
    for p in range(passes):
        for s in range(0, len(train_prompts), mb):
            ps = train_prompts[s:s + mb]
            gs = train_golds[s:s + mb]
            sync_gen()
            comps = sample_groups(gen_model, tok, ps, group)
            flat_p = [q for q in ps for _ in range(group)]
            rewards = torch.tensor(
                [1.0 if common.answers_equal(common.extract_answer(c), g)
                 else 0.0 for c, g in zip(comps, [q for q in gs
                                                 for _ in range(group)])],
                device="cuda")
            reward_trace.append(rewards.mean().item())
            adv = rewards.view(-1, group)
            adv = (adv - adv.mean(dim=1, keepdim=True)) / (
                adv.std(dim=1, keepdim=True) + 1e-4)
            adv = adv.view(-1)
            with torch.no_grad():
                old_lps = token_logps(gen_model, tok, flat_p, comps)
                ref_lps = token_logps(ref, tok, flat_p, comps)
            policy.train()
            # one on-policy update, sequences in chunks of 32
            total_loss = 0.0
            n_seq = len(comps)
            for c0 in range(0, n_seq, 32):
                cur_lps = token_logps(policy, tok, flat_p[c0:c0 + 32],
                                      comps[c0:c0 + 32], grad=True)
                losses = []
                for j, cur in enumerate(cur_lps):
                    k = c0 + j
                    old = old_lps[k]
                    rfl = ref_lps[k]
                    n = min(len(cur), len(old), len(rfl))
                    cur, old, rfl = cur[:n], old[:n], rfl[:n]
                    ratio = torch.exp(cur - old)
                    a = adv[k]
                    surr = torch.min(ratio * a,
                                     torch.clamp(ratio, 0.8, 1.2) * a)
                    d = rfl - cur
                    kl = torch.exp(d.detach()) - d - 1.0
                    losses.append(((-surr + args.beta * kl).mean()))
                loss = torch.stack(losses).mean() * (len(cur_lps) / n_seq)
                loss.backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            print(f"pass {p} mb {s // mb} reward "
                  f"{reward_trace[-1]:.3f} loss {total_loss:.4f}",
                  flush=True)

    # frozen final measurement on the held-out-from-training prompts
    policy.eval()
    sync_gen()
    meas_prompts = [common.gsm8k_prompt(r["question"]) for r in meas_rows]
    meas_golds = [common.gold_answer(r["answer"]) for r in meas_rows]
    comps = sample_groups(gen_model, tok, meas_prompts, group)
    flat_p = [q for q in meas_prompts for _ in range(group)]
    flat_g = [q for q in meas_golds for _ in range(group)]
    rewards = [1.0 if common.answers_equal(common.extract_answer(c), g)
               else 0.0 for c, g in zip(comps, flat_g)]
    final_reward = sum(rewards) / len(rewards)
    with torch.no_grad():
        pol_lps = token_logps(gen_model, tok, flat_p, comps)
        ref_lps = token_logps(ref, tok, flat_p, comps)
    kls = []
    for pl, rl in zip(pol_lps, ref_lps):
        n = min(len(pl), len(rl))
        d = rl[:n] - pl[:n]
        kls.append((torch.exp(d) - d - 1.0).mean().item())
    final_kl = sum(kls) / len(kls)

    outdir = os.path.join(common.RUNS, "grpo", f"beta{args.beta}",
                          f"seed{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    policy.save_pretrained(os.path.join(outdir, "ckpt"))
    tok.save_pretrained(os.path.join(outdir, "ckpt"))
    common.dump_json(os.path.join(outdir, "metrics.json"), {
        "seed": args.seed, "beta": args.beta,
        "final_reward": final_reward, "final_kl": final_kl,
        "train_reward_trace": reward_trace,
        "n_train_prompts": n_train, "n_meas_prompts": n_meas,
        "gpu_hours": timer.hours(), "smoke": SMOKE})
    print("done", outdir, final_reward, final_kl, flush=True)


if __name__ == "__main__":
    main()
