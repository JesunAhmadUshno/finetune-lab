# finetune-lab

Fine-tuning and post-training: LoRA, QLoRA, DPO, and a GRPO track, scored before and after.

Status: Learn DONE 2026-11-05. Build DONE 2026-11-05. Deep DONE 2026-11-05, v1.0 shipped. On synthetic NumPy tasks only, the Build SFT pipeline (512 train, 128 validation, 128 held-out test, 4 classes, input dim 16) trained a rank 2 LoRA adapter (40 trainable parameters against 64 for full fine-tuning) with minibatch training and raised held-out test accuracy from 0.6797 (87 of 128) before training to 0.9141 (117 of 128) after 60 epochs, with test loss from 1.0007 to 0.2032, the frozen base weights unchanged, and a checkpoint round-trip reproducing the after numbers exactly; and the Build GRPO track trained a tabular toy policy on 8 checker prompts (group size 8, clip eps 0.2, KL beta 0.04, 60 iterations) and raised fresh-sample mean reward from 0.2070 to 0.9766, with greedy accuracy from 0.125 (1 of 8) to 1.0000 (8 of 8), all on Muse VM CPU with no real LLM, no real text data, and no open model run.

Deep main claim: on a synthetic NumPy tabular policy only (8 prompts, 4 actions, reused from Build), a DPO run (beta 0.1, frozen reference, 300 full-batch steps) raised the held-out preference margin from -0.0654 to 0.8297 and held-out preference accuracy from 0.4375 (7 of 16) to 1.0000 (16 of 16), with train margin from -0.1123 to 1.6779, greedy checker accuracy from 0.125 (1 of 8) to 1.0000 (8 of 8) and no forgetting on that aligned task (change +0.875 greedy, +0.336 mean reward), and a QLoRA accounting study (not measured GPU memory) putting a 7B-scale model's 4-bit base plus rank 16 adapter at 3,516,777,216 bytes against 14,000,000,000 bytes for fp16 full weights; all on Muse VM CPU with no real LLM.
Project: P10. Fleet node: BlackBox (GPU). Wave: Wave 5.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path. DONE 2026-11-05.
- Level 2, Build: a real working system other people can run. DONE 2026-11-05.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases. DONE 2026-11-05 (compressed Deep: DPO run, forgetting checks, QLoRA memory accounting study, reward hacking probe; the rest of the v2 Deep scope is the v1.1 backlog in ROADMAP.md).

## Benchmarks

| Test | Result |
| --- | --- |
| Learn: LoRA test accuracy, synthetic task (512 train, 128 test, 4 classes), before training | 0.5469 (70 of 128), test loss 1.0707 |
| Learn: LoRA test accuracy, same task, after 200 full batch epochs, rank 2 adapter, 40 trainable parameters of 64 full | 0.9219 (118 of 128), test loss 0.2534 |
| Learn: GRPO advantages, 3 stated reward groups of 4 | Computed and hand checked; see level-1-learn/README.md |
| Build: SFT held-out test accuracy, synthetic task (512 train, 128 validation, 128 test, 4 classes), before training | 0.6797 (87 of 128), test loss 1.0007 |
| Build: SFT held-out test accuracy, same held-out slice, after 60 minibatch epochs (batch 64), rank 2 adapter, 40 trainable parameters of 64 full | 0.9141 (117 of 128), test loss 0.2032 |
| Build: GRPO fresh-sample mean reward, 8 checker prompts x 32 samples, before training | 0.2070, greedy accuracy 0.125 (1 of 8) |
| Build: GRPO fresh-sample mean reward, same eval, after 60 iterations (group 8, clip 0.2, KL beta 0.04) | 0.9766, greedy accuracy 1.0000 (8 of 8) |
| Deep: DPO held-out preference margin and accuracy, 16 held-out pairs, before training | -0.0654, accuracy 0.4375 (7 of 16) |
| Deep: DPO held-out preference margin and accuracy, same pairs, after 300 full-batch steps (beta 0.1) | 0.8297, accuracy 1.0000 (16 of 16) |
| Deep: forgetting check, greedy checker accuracy on the Build GRPO task, before and after DPO | 0.125 (1 of 8) before, 1.0000 (8 of 8) after, change +0.875 |
| Deep: QLoRA accounting, large-7B stated size, 4-bit base plus rank 16 adapter against fp16 full weights | 3,516,777,216 bytes against 14,000,000,000 bytes (fraction 0.2512, accounting, not measured GPU memory) |

Note on slices: the Build validation slice is the slice Learn used as its test set, and the Build test slice is fresh held-out data. Build and Learn test numbers are not the same set; see level-2-build/README.md.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
