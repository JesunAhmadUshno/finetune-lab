# Level 1, Learn: finetune-lab

Status: DONE (2026-11-05). LoRA from scratch in NumPy and a GRPO
advantage teaching module, plus the written RLHF pipeline and the
GRPO against PPO note from the SPEC Learn scope.

## Honest scope line

This level uses a synthetic task and a NumPy teaching model. It is
not a real LLM fine-tune, no GPU run happened at this level, and the
RLHF pipeline below is a written and small code treatment: no real
SFT run, no reward model training, and no RL policy run happened.
The run was on the Muse VM, CPU, by design for this teaching level.

Summary claim: Learn DONE: on a synthetic NumPy task only (512 train, 128 test, 4 classes, input dim 16), a rank 2 LoRA adapter (40 trainable parameters against 64 for full fine-tuning of the same layer) raised test accuracy from 0.5469 (70 of 128) before training to 0.9219 (118 of 128) after 200 full batch epochs, with test loss from 1.0707 to 0.2534, the frozen base weights unchanged, and GRPO advantages computed for 3 stated reward groups of 4, all on Muse VM CPU with no real LLM and no SFT or RL policy run.

## The task, defined exactly

- Inputs: synthetic vectors in R^16 from a standard normal, data
  seed 7. 512 train examples, 128 test examples.
- Labels: argmax over 4 classes of x @ W_target.
- W_target = W0 + scaling * (A_true @ B_true). W0 is the frozen base
  weight matrix, shape (16, 4), seed 3, scale 0.5. A_true is (16, 2)
  and B_true is (2, 4), seed 9, scale 0.6. Scaling is 2.0.
- The construction makes the change from the base exactly rank 2, so
  a rank 2 adapter can express it. Real tasks do not promise that.

Think of each input as a stand-in instruction embedding and each
class as a stand-in response type. That analogy is the teaching
framing only; no text, tokens, or language model are involved.

## Method: LoRA from scratch (lora_learn.py)

1. Frozen base. The base linear layer W0 is created once from its
   seed and is never updated. Its sha256 hash is recorded before and
   after training in results_learn.json and is identical.
2. Adapter. The effective weight is W0 + scaling * (A @ B), with A
   of shape (16, 2), B of shape (2, 4), rank 2, alpha 4.0, and
   scaling alpha / rank = 2.0. A starts small and random (seed 5,
   scale 0.1). B starts at zero, so at step 0 the adapted model is
   exactly the base model.
3. Training. A hand-written full batch gradient descent loop trains
   only A and B on mean cross entropy, learning rate 0.5, 200 epochs.
   Gradients are derived by hand through the softmax loss and the
   A @ B product. No framework, no optimizer library.
4. Scoring. Accuracy and loss are measured on train and test before
   training (base alone, because B starts at zero) and after.

## The RLHF pipeline, end to end

Stage 1, SFT (supervised fine-tuning). Start from a pretrained base
model. Train it on instruction and response pairs written or chosen
by humans, with the ordinary next token loss on the response. The
output is a model that follows instructions in roughly the right
style. This SFT model is both the starting policy for the next
stages and the reference model used later for comparison.

Stage 2, reward model. Collect comparisons: for one prompt, humans
rank two or more sampled responses from the SFT model. Train a
separate reward model to predict a scalar score that agrees with
those rankings, usually with a pairwise ranking loss. The output is
a scorer that stands in for human preference at training scale. Its
known weakness is that it is a proxy: it can be wrong off its data
and it can be exploited in the next stage.

Stage 3, policy run. Sample responses from the SFT policy on new
prompts, score them with the reward model, and update the policy to
raise the expected reward, classically with PPO and a penalty that
keeps the policy from drifting too far from the reference SFT model.
Evaluate on held out prompts, watch for reward hacking (high reward
scores with worse real quality), then stop, ship, or collect more
preference data and repeat. Nothing in this level runs any of these
three stages on a real model; Build is where a small real run lives.

## GRPO against PPO note

- What PPO uses. PPO needs a baseline to turn a reward into an
  advantage, and in RLHF that baseline usually comes from a learned
  value function, a critic network trained alongside the policy.
- What the GRPO group baseline replaces. GRPO samples a group of
  responses for the same prompt, scores each one, and uses the group
  mean reward as the baseline for that prompt. The advantage of one
  response is its reward minus that group mean. The normalized form
  also divides by the group standard deviation. The learned critic
  is removed; the baseline is computed from the group itself.
- What it costs. There is no free baseline. GRPO must sample and
  score a whole group per prompt before any advantage exists, so
  sampling cost per prompt rises with group size. A small group
  gives a noisy baseline, an all equal group gives zero advantages
  and no learning signal for that prompt (shown in the examples),
  and the method still needs a trustworthy reward or checker.
  grpo_learn.py computes the baseline, the raw advantages, and the
  normalized advantages only; it trains no policy.

## Run it

    python level-1-learn/run_learn.py
    python -m unittest tests.test_learn_finetune -v
    python tests/test_smoke.py

The runner writes level-1-learn/results_learn.json deterministically:
run it twice and the file is identical (verified by hash in the
ship record for this level).

## Results (measured 2026-11-05, Muse VM, CPU, Python 3.12, NumPy 1.26.4)

LoRA, synthetic task as defined above:

| Split | Before loss | Before accuracy | After loss | After accuracy |
| --- | --- | --- | --- | --- |
| Train, 512 | 1.0472 | 0.6094 | 0.1690 | 0.9473 |
| Test, 128 | 1.0707 | 0.5469 (70 of 128) | 0.2534 | 0.9219 (118 of 128) |

Parameters: LoRA trainable 40, full fine-tune of the same layer 64,
trainable fraction 0.625. Base weights unchanged: hash before equals
hash after in results_learn.json.

GRPO worked examples (rewards are stated teaching values):

| Group | Rewards | Baseline mean | Std | Advantages | Normalized |
| --- | --- | --- | --- | --- | --- |
| prompt_a_binary | 1, 0, 1, 0 | 0.5 | 0.5 | 0.5, -0.5, 0.5, -0.5 | 1, -1, 1, -1 |
| prompt_b_graded | 2, 1, 0, 1 | 1.0 | 0.7071 | 1, 0, -1, 0 | 1.4142, 0, -1.4142, 0 |
| prompt_c_all_equal | 1, 1, 1, 1 | 1.0 | 0.0 | 0, 0, 0, 0 | 0, 0, 0, 0 |

Unit tests: 14 tests, all passing, plus the scaffold smoke test.

Hand checks:
- LoRA parameters: A is 16 x 2 = 32 and B is 2 x 4 = 8, so
  32 + 8 = 40 trainable. Full layer: 16 x 4 = 64. 40 / 64 = 0.625.
- GRPO prompt_a: mean of 1, 0, 1, 0 is 0.5. Each advantage is the
  reward minus 0.5, giving 0.5, -0.5, 0.5, -0.5. The population
  standard deviation is 0.5, so the normalized values are
  0.5 / 0.5 = 1 and -0.5 / 0.5 = -1, alternating.
- Test counts: 0.5469 x 128 rounds to 70 and 0.9219 x 128 rounds to
  118; the exact fractions are 70 / 128 = 0.546875 and
  118 / 128 = 0.921875.

## Limits named at review

- Synthetic task and NumPy teaching model, not a real LLM.
- RLHF pipeline is a written and small code treatment, no real SFT, reward model, or RL policy run.
- Run on Muse VM CPU, not the assigned BlackBox GPU node, even though BlackBox was verified on this cycle, because this Learn design is CPU teaching work.
- The true change from the base is rank 2 by construction, so the adapter result is easier than a real task of unknown rank.
- One linear layer only, no bias, no separate optimizer, full batch training only.
- GRPO code computes advantages only, with no policy loss, clipping, or KL term.

## Failure modes and limits

- Rank is a ceiling. If the needed change has higher rank than the
  adapter, the adapter cannot express it, however long it trains.
  This task hides that failure by construction; a real task would
  need a rank sweep.
- The parameter saving is modest at this size: 40 against 64. LoRA
  saves more on large layers, where in_dim x classes is much bigger
  than rank x (in_dim + classes). The small numbers here are the
  honest teaching case, not the large model case.
- B starting at zero means the gradient for A is zero at step 0.
  Training starts through B. That is standard LoRA behaviour and is
  visible in the code.
- Full batch gradient descent on 512 synthetic points says nothing
  about stochastic training, real data quality, or forgetting of
  base abilities, none of which this level measures.
- GRPO advantages are only as good as the reward. A bad or hackable
  reward gives confident, wrong advantages. An all equal reward
  group gives no signal at all.
- No real model, dataset, or GPU was used. Any number here is about
  this synthetic task only.

## What Build adds next

A LoRA fine-tune of a small open model on a custom instruction set
with before and after eval numbers, and one small GRPO run on an
open model with a verifiable task, scored before and after. See
PROPOSAL.md (original scope) and the root README level map.

## References

- PPO: Proximal Policy Optimization, arXiv 1707.06347. Reference
  only, no text quoted.
- GRPO, as introduced in DeepSeekMath, arXiv 2402.03300. Reference
  only, no text quoted.
