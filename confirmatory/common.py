"""Shared code for the finetune-lab confirmatory run (lock: docs/decisions/0006).

One protocol everywhere: prompt formats, the GSM8K checker, model
loading, log-probability and generation helpers. See RUN-PLAN.md.
"""
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
MODEL_ID = "Qwen/Qwen2.5-1.5B"
MAX_LEN = 512

GSM8K_EVAL_INSTRUCTION = (
    "Solve the following math word problem. "
    "Give the final numeric answer after '####'."
)


def alpaca_prompt(instruction, inp):
    text = "### Instruction:\n" + instruction
    if inp and inp.strip():
        text += "\n\n### Input:\n" + inp
    text += "\n\n### Response:\n"
    return text


def gsm8k_prompt(question):
    return (
        "### Instruction:\n" + GSM8K_EVAL_INSTRUCTION
        + "\n\n### Input:\n" + question + "\n\n### Response:\n"
    )


def strip_calc_annotations(text):
    return re.sub(r"<<[^>]*>>", "", text)


def gold_answer(answer_field):
    return answer_field.split("####")[-1].strip().replace(",", "")


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text):
    """Final numeric answer per the run plan: after the last ####
    when present, else the last number in the text."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUM_RE.search(tail)
        if m:
            return m.group(0).replace(",", "")
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def answers_equal(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred.strip() == str(gold).strip()


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path=None, dtype=torch.bfloat16, train=False,
               grad_ckpt=False):
    src = path if path is not None else MODEL_ID
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype)
    model.cuda()
    if train:
        model.config.use_cache = False
        if grad_ckpt:
            model.gradient_checkpointing_enable()
        model.train()
    else:
        model.eval()
    return model


def load_policy(path, dtype=torch.bfloat16, train=False,
                grad_ckpt=False):
    """Load a policy from a checkpoint dir; merges a LoRA adapter
    dir (one holding adapter_config.json) into the base model."""
    if path and os.path.exists(os.path.join(path, "adapter_config.json")):
        from peft import PeftModel
        base = load_model(None, dtype=dtype, train=False)
        model = PeftModel.from_pretrained(base, path)
        model = model.merge_and_unload()
        model.cuda()
        if train:
            model.config.use_cache = False
            if grad_ckpt:
                model.gradient_checkpointing_enable()
            model.train()
        else:
            model.eval()
        return model
    return load_model(path, dtype=dtype, train=train,
                      grad_ckpt=grad_ckpt)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


@torch.no_grad()
def batch_logps(model, tok, prompts, completions, batch_size=16):
    """Summed log-prob of each completion given its prompt."""
    out = []
    for i in range(0, len(prompts), batch_size):
        p_chunk = prompts[i:i + batch_size]
        c_chunk = completions[i:i + batch_size]
        texts = [p + c for p, c in zip(p_chunk, c_chunk)]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False)
        plen = []
        for p in p_chunk:
            plen.append(len(tok(p, add_special_tokens=False)["input_ids"]))
        ids = enc["input_ids"].cuda()
        attn = enc["attention_mask"].cuda()
        logits = model(input_ids=ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        tok_logp = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
        # token position j (in tgt indexing) predicts ids[j+1]; it is a
        # completion token when j+1 >= prompt_len
        pos = torch.arange(1, ids.shape[1], device=ids.device).unsqueeze(0)
        plen_t = torch.tensor(plen, device=ids.device).unsqueeze(1)
        mask = (pos >= plen_t) & (attn[:, 1:] > 0)
        out.extend((tok_logp * mask).sum(dim=1).tolist())
    return out


@torch.no_grad()
def generate_batch(model, tok, prompts, max_new_tokens=320, batch_size=64,
                   do_sample=False, temperature=1.0, num_return_sequences=1):
    """Generate completions; returns list of completion strings,
    flattened in prompt-major order when num_return_sequences > 1."""
    old_side = tok.padding_side
    tok.padding_side = "left"
    results = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=False)
            ids = enc["input_ids"].cuda()
            attn = enc["attention_mask"].cuda()
            gen = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            new = gen[:, ids.shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            results.extend(texts)
    finally:
        tok.padding_side = old_side
    return results


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def hours(self):
        return (time.time() - self.t0) / 3600.0
