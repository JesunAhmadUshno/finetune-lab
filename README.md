# finetune-lab

Fine-tuning and post-training: LoRA, QLoRA, DPO, and a GRPO track, scored before and after.

Status: Learn DONE 2026-11-05. On a synthetic NumPy task only (512 train, 128 test, 4 classes, input dim 16), a rank 2 LoRA adapter (40 trainable parameters against 64 for full fine-tuning of the same layer) raised test accuracy from 0.5469 (70 of 128) before training to 0.9219 (118 of 128) after 200 full batch epochs, with test loss from 1.0707 to 0.2534, the frozen base weights unchanged, and GRPO advantages computed for 3 stated reward groups of 4, all on Muse VM CPU with no real LLM and no SFT or RL policy run. Build and Deep are not started.
Project: P10. Fleet node: BlackBox (GPU). Wave: Wave 5.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path. DONE 2026-11-05.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

| Test | Result |
| --- | --- |
| Learn: LoRA test accuracy, synthetic task (512 train, 128 test, 4 classes), before training | 0.5469 (70 of 128), test loss 1.0707 |
| Learn: LoRA test accuracy, same task, after 200 full batch epochs, rank 2 adapter, 40 trainable parameters of 64 full | 0.9219 (118 of 128), test loss 0.2534 |
| Learn: GRPO advantages, 3 stated reward groups of 4 | Computed and hand checked; see level-1-learn/README.md |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
