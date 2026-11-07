# Rigor study: seeds, sweeps, baselines, ablations

Every level result in this repo is one seed configuration. This
folder reruns the same experiments with seeds and settings as
parameters, so each recorded number can be read with its spread
and against a baseline. All synthetic, NumPy, CPU, no real model.

Run it (about 3 seconds on one CPU core):

    python3 rigor/run_rigor.py
    python3 -m unittest tests/test_rigor.py

Source of every number below: rigor/results_rigor.json. Running
run_rigor.py twice gives a byte identical file.

## Method

- Seed rule: offset s adds 1000 * s to every seed of a run (SFT:
  data, base, target, adapter, shuffle; GRPO: policy, train, eval;
  DPO: policy, eval). s runs 0 to 9. Offset 0 is the recorded
  configuration, and tests/test_rigor.py checks that it reproduces
  results_build.json and results_deep.json exactly.
- A new seed draws a new synthetic task (SFT) or a new initial
  policy and sampling stream (GRPO, DPO), so the spread covers
  task, initialisation and sampling together.
- Summaries: mean, sample standard deviation (ddof 1), and a
  percentile bootstrap 95% interval of the mean (10,000 resamples,
  bootstrap seed 12345). With 10 seeds the bootstrap interval is
  approximate and tends to run narrow; read it as a spread, not a
  significance test.

## Multi-seed results (10 seeds)

| Track | Measure | Recorded (seed 0) | Mean over 10 seeds | Std | Bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| SFT, LoRA rank 2 | Test accuracy, final epoch | 0.9141 | 0.9039 | 0.0486 | 0.8742 to 0.9305 |
| SFT, LoRA rank 2 | Test accuracy, best-validation epoch | n/a | 0.9219 | 0.0386 | 0.8969 to 0.9414 |
| SFT, LoRA rank 2 | Test accuracy before training | 0.6797 | 0.5813 | 0.1092 | 0.5188 to 0.6430 |
| GRPO | Fresh-sample mean reward after 60 iterations | 0.9766 | 0.9465 | 0.0215 | 0.9340 to 0.9590 |
| GRPO | Mean KL to reference | 1.1795 | 1.1220 | 0.0292 | 1.1056 to 1.1394 |
| DPO | Held-out preference margin | 0.8297 | 0.8832 | 0.0395 | 0.8603 to 0.9065 |
| DPO | Fresh-sample mean reward on the checker task | 0.5430 | 0.5141 | 0.0363 | 0.4930 to 0.5352 |

Reading: the recorded seed sits inside the spread for every
track. The recorded GRPO reward (0.9766) is the best of the 10
seeds, so the typical GRPO result is about 0.95, not 0.98. The
recorded DPO margin is the lowest of the 10 seeds.

## SFT baselines: full fine-tune against LoRA ranks (10 seeds)

| Model | Trainable parameters | Test accuracy, final epoch (mean, CI) | Best-validation epoch |
| --- | --- | --- | --- |
| Full fine-tune | 64 | 0.9391 (0.9219 to 0.9555) | 0.9391 |
| LoRA rank 1 | 20 | 0.7672 (0.7180 to 0.8149) | 0.7953 |
| LoRA rank 2 | 40 | 0.9039 (0.8742 to 0.9305) | 0.9219 |
| LoRA rank 3 | 60 | 0.9547 (0.9398 to 0.9695) | 0.9539 |
| LoRA rank 4 | 80 | 0.9539 (0.9375 to 0.9703) | 0.9547 |

Reading: with the same schedule, full fine-tuning of the 64
weights beats the 40-parameter rank 2 adapter on this task (0.939
against 0.904 at the final epoch). Ranks 3 and 4 have more
parameters than the full layer here (the layer is only 16 x 4), so
they are not parameter savings in this toy; they are reported to
show the rank curve. Rank 1 is too small for a rank 2 target.

## GRPO KL beta sweep and budget (10 seeds, mean)

| Beta | Reward at 10 iterations | Reward at 30 | Reward at 60 | KL at 60 |
| --- | --- | --- | --- | --- |
| 0 | 0.6160 | 0.8840 | 0.9637 | 1.1980 |
| 0.01 | 0.6129 | 0.8781 | 0.9617 | 1.1816 |
| 0.04 (recorded setting) | 0.6055 | 0.8516 | 0.9465 | 1.1220 |
| 0.1 | 0.5828 | 0.7992 | 0.8859 | 0.9544 |

One iteration samples 64 actions (8 prompts x group 8). Reading:
the KL term trades reward for staying near the reference. On this
task there is nothing to protect by staying close, so beta 0 wins
on reward; on a real model the KL term guards other capabilities,
which this toy cannot show.

## GRPO ablations (10 seeds)

| Variant | Mean reward (CI) | KL to reference |
| --- | --- | --- |
| GRPO (clip 0.2, beta 0.04, 2 inner epochs) | 0.9465 (0.9340 to 0.9590) | 1.1220 |
| GRPO without clipping | 0.9469 (0.9344 to 0.9598) | 1.1264 |
| GRPO without KL term | 0.9637 (0.9543 to 0.9735) | 1.1980 |
| REINFORCE with group baseline (1 epoch, no clip, no KL) | 0.8922 (0.8746 to 0.9098) | 0.9546 |

Reading: removing clipping changes almost nothing at this
learning rate and 2 inner epochs, and removing the KL term raises
reward. Since neither removal lowers reward, the gap to the
single-pass REINFORCE update comes from GRPO's second inner epoch
on each sampled batch. This is a tabular stand-in for the
GRPO against PPO question; no critic is trained, so it is not a
PPO comparison.

## DPO sweep (recorded seed, 36 settings)

beta in {0.05, 0.1, 0.2, 0.5}, learning rate in {0.1, 0.5, 1.0},
steps in {100, 300, 1000}. Each row reports train and held-out
margin, held-out accuracy, greedy accuracy, sampled mean reward
and KL to the initial policy; all 36 rows are in the results file.
Across the grid, larger beta, learning rate and step counts
mostly move the margin and the KL up together (3 ordered pairs
of settings break that pattern), and held-out margin tracks
sampled reward closely (correlation 0.93 over the 36 rows). The best held-out
margin in the grid is 5.38 (beta 0.2, lr 1.0, 1000 steps) with
sampled reward 0.9961 and KL 1.3915. The recorded setting (beta
0.1, lr 0.5, 300 steps) is a light-touch point on that curve,
which is why its sampled reward (0.5430) is far below GRPO's.

## Misaligned forgetting check (10 seeds)

On the first k prompts the chosen action is correct plus 1 and
the rejected action is the checker-correct one, so DPO pulls
against the task there.

| k misaligned prompts | Greedy accuracy after | Greedy on misaligned prompts | Greedy on aligned prompts | Mean reward |
| --- | --- | --- | --- | --- |
| 0 | 1.0000 | n/a | 1.0000 | 0.5141 |
| 2 | 0.7500 | 0.0000 | 1.0000 | 0.4035 |
| 4 | 0.5000 | 0.0000 | 1.0000 | 0.3066 |

Before DPO the greedy accuracy is 0.2375 on average over the same
10 seeds. Reading: misaligned preferences remove the task on every
prompt where they disagree with the checker, and leave the other
prompts untouched. That isolation is structural (tabular rows share
no weights), so this check shows DPO follows the preference data
over the task, not how far damage spreads in a shared-weight model.
