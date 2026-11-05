# Website project page copy: finetune-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

finetune-lab: LoRA, GRPO, and DPO built from scratch and scored before and after, with a reward hacking probe and QLoRA accounting

## Summary

finetune-lab is an open fine-tuning and post-training project
built by Jesun Ahmad Ushno as project P10 in Roadmap v3. It
builds the post-training stack in three levels: LoRA from
scratch in NumPy with hand-checked GRPO advantages and a written
RLHF pipeline; a full SFT-style LoRA pipeline with checkpointing
and an eval harness, plus a GRPO track that trains a tabular
toy policy; and a compressed Deep with a DPO run, forgetting
checks, a reward hacking probe, and a QLoRA memory accounting
study. Labelled honestly: this is toy NumPy teaching work, not
a real LLM. No real text data and no open model was run, the
preferences are synthetic from a deterministic checker rule,
and all work ran on CPU, not on a GPU node.

Status: v1.0, 2026-11-05. Learn, Build, and compressed Deep are
done. Total tests: 39, all passing at v1.0. Level grades: Learn
8,200 of 10,000, Build 8,300 of 10,000, Deep 8,000 of 10,000.

## Results

All results below are measured runs on synthetic data, recorded
in the level READMEs in the GitHub repo. They prove the LoRA,
GRPO, DPO, checkpoint, and probe machinery at teaching scale,
not production fine-tuning quality.

| Level | Set | Headline result | Notes |
| --- | --- | --- | --- |
| Learn | Synthetic task, 512 train, 128 test | Test accuracy 0.5469 (70 of 128) to 0.9219 (118 of 128), loss 1.0707 to 0.2534 | Rank 2 adapter, 40 trainable parameters of 64; base unchanged; 14 tests |
| Build | SFT 512 train, 128 validation, 128 held-out test; GRPO 8 checker prompts | SFT held-out accuracy 0.6797 (87 of 128) to 0.9141 (117 of 128); GRPO mean reward 0.2070 to 0.9766, greedy 0.125 to 1.0000, KL 1.1795 | Checkpoint round-trip exact; 14 tests, 28 total at that level |
| Deep (compressed) | DPO on the Build tabular policy, 8 train pairs, 16 held-out pairs | Held-out margin -0.0654 to 0.8297 (+0.8951), held-out accuracy 0.4375 (7 of 16) to 1.0000 (16 of 16), loss 0.6931 to 0.6076 | Forgetting check greedy 0.125 to 1.0000, reward 0.2070 to 0.5430; suppression probe games train to 1.0000, held-out stays 0.4375; QLoRA 7B accounting fraction 0.2512; 11 tests, 39 total |

## Honest limits

- Toy NumPy work only, at every level. No real LLM, no real
  text, no tokenizer, and no open model run.
- Deep preferences are synthetic, from a deterministic checker
  rule, not human or model preference data.
- QLoRA is weight-storage accounting, not measured GPU memory.
  No quantized model was loaded or trained.
- The forgetting check is weak by construction, because the
  preferences are aligned with the retained task.
- Held-out in Deep means unseen rejected contrasts on known
  prompts, not unseen prompts or tasks.
- Compressed Deep is not the full v2 Deep scope. The remainder
  is the v1.1 backlog in the repo ROADMAP.md.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/finetune-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, compressed Deep): https://github.com/JesunAhmadUshno/finetune-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
