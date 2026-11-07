// Known-answer check, run by `npm run build` before vite: the in-browser
// ports in lab.js must reproduce the repo's recorded exact quantities.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  memoryRow, groupAdvantages, makePairs, preferenceMetrics, dpoLossAndGrad, trainDpo, suppress,
  greedyAccuracy, clippedRatio, surrogate, klRow, trainGrpo, expectedReward,
  loraParams, makeLoraTask, trainLora, loraGrad, loraEvaluate, loraEffective,
} from './lab.js';

const load = (p) => JSON.parse(readFileSync(new URL(`../../${p}`, import.meta.url)));
const learn = load('level-1-learn/results_learn.json');
const build = load('level-2-build/results_build.json');
const deep = load('level-3-deep/results_deep.json').deep;
const rigor = load('rigor/results_rigor.json');
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

// 1. QLoRA accounting: all 4 recorded rows, every field, to the byte.
const study = deep.qlora_memory_study;
for (const row of study.rows) {
  const got = memoryRow({ parameters: row.parameters, layers: row.layers, hidden: row.hidden, rank: row.rank, baseBits: study.assumptions.bytes_4bit * 8 });
  for (const [k, v] of Object.entries(got)) assert.equal(v, row[k], `${row.name} ${k}`);
}

// 2. GRPO group advantages: the 3 recorded Learn examples.
for (const [name, ex] of Object.entries(learn.grpo_examples)) {
  const got = groupAdvantages(ex.rewards);
  assert.deepEqual(got, { baseline_mean: ex.baseline_mean, std_population: ex.std_population, advantages: ex.advantages, advantages_normalized: ex.advantages_normalized }, name);
}

// 3. DPO gradient hand case (tests/test_deep_dpo.py): zero margin, beta 0.1.
{
  const z = [[0, 0, 0, 0]];
  const { loss, grad } = dpoLossAndGrad(z, z, [[0, 0, 1]], 0.1);
  assert.equal(Math.round(loss * 1e4) / 1e4, 0.6931, 'loss at zero margin is log 2');
  assert.deepEqual(grad[0].map((v) => Math.round(v * 1e9) / 1e9), [-0.05, 0.05, 0, 0]);
}

// 4. DPO run from the recorded init policy reproduces results_deep.json.
const init = rigor.site_fixtures.init_policy_logits;
const correct = build.grpo_track.config.correct_actions;
const pairs = makePairs(correct);
const d = deep.dpo;
assert.deepEqual(preferenceMetrics(init, pairs.train), { mean_margin: d.train_pairs_before.mean_margin, preference_accuracy: d.train_pairs_before.preference_accuracy, preference_correct: d.train_pairs_before.preference_correct });
assert.deepEqual(preferenceMetrics(init, pairs.held), { mean_margin: d.held_out_pairs_before.mean_margin, preference_accuracy: d.held_out_pairs_before.preference_accuracy, preference_correct: d.held_out_pairs_before.preference_correct });
const run = trainDpo({ init, pairs, beta: deep.config.beta, lr: deep.config.learning_rate, steps: deep.config.steps });
assert.deepEqual(run.history, d.history, 'DPO history');
assert.equal(run.final_loss, d.loss_after);
assert.equal(preferenceMetrics(run.logits, pairs.train).mean_margin, d.train_pairs_after.mean_margin);
assert.deepEqual(preferenceMetrics(run.logits, pairs.held), { mean_margin: d.held_out_pairs_after.mean_margin, preference_accuracy: d.held_out_pairs_after.preference_accuracy, preference_correct: d.held_out_pairs_after.preference_correct });
assert.equal(greedyAccuracy(init, correct), d.task_before.greedy_accuracy);
assert.equal(greedyAccuracy(run.logits, correct), d.task_after.greedy_accuracy);

// 5. Reward hacking probe, deterministic parts.
const probe = deep.reward_hacking_probe;
const sup = suppress(init, pairs.train);
assert.equal(preferenceMetrics(sup, pairs.train).preference_accuracy, probe.rejected_only_suppression_probe.train_preference_accuracy);
assert.equal(preferenceMetrics(sup, pairs.held).preference_accuracy, probe.rejected_only_suppression_probe.held_out_preference_accuracy);
assert.equal(greedyAccuracy(sup, correct), probe.rejected_only_suppression_probe.greedy_accuracy);
const constant = init.map(() => [10, 0, 0, 0]);
assert.equal(preferenceMetrics(constant, pairs.train).preference_accuracy, probe.constant_policy_probe.train_preference_accuracy);
assert.equal(preferenceMetrics(constant, pairs.held).preference_accuracy, probe.constant_policy_probe.held_out_preference_accuracy);

// 6. GRPO pieces: hand cases from tests/test_build_finetune.py, then a live run.
assert.equal(clippedRatio(1.5, 0.2), 1.2);
assert.equal(clippedRatio(0.5, 0.2), 0.8);
close(surrogate(1.5, 2, 0.2), 2.4, 1e-12, 'surrogate +A');
close(surrogate(1.5, -2, 0.2), -3.0, 1e-12, 'surrogate -A');
close(klRow([1, 0], [0.5, 0.5]), Math.log(2), 1e-9, 'KL by hand');
const g = build.grpo_track.config;
const grpo = trainGrpo({ init, correct, beta: g.beta_kl, eps: g.eps_clip, group: g.group_size, iterations: g.iterations, lr: g.learning_rate, inner: g.inner_epochs });
assert.equal(grpo.history[0].greedy, build.grpo_track.before_training.greedy_accuracy);
assert.ok(expectedReward(grpo.logits, correct) > expectedReward(init, correct) + 0.5, 'live GRPO learns');
assert.equal(grpo.history.at(-1).greedy, build.grpo_track.after_training.greedy_accuracy);

// 7. LoRA: parameter counts from results_learn.json, start equals base, gradient by finite differences.
assert.equal(loraParams(learn.lora.lora.rank), learn.lora.lora.trainable_parameters);
assert.equal(16 * 4, learn.lora.lora.full_finetune_parameters);
const task = makeLoraTask(7, { nTrain: 64, nTest: 32 });
const small = trainLora(task, { rank: 2, epochs: 0 });
const baseEval = loraEvaluate(task.xTest, task.yTest, task.base);
assert.equal(small.before.test.accuracy, baseEval.accuracy, 'B = 0 so the untrained adapter is the base');
{
  const A = [[0.3, -0.2], ...Array.from({ length: 15 }, (_, i) => [0.01 * i, -0.02 * i])];
  const B = [[0.1, -0.1, 0.2, 0], [0, 0.3, -0.2, 0.1]];
  const { dA, dB } = loraGrad(task.xTrain, task.yTrain, task.base, A, B, 2);
  const L = (A2, B2) => loraEvaluate(task.xTrain, task.yTrain, loraEffective(task.base, A2, B2, 2)).loss;
  const h = 1e-6;
  for (const [i, j] of [[0, 0], [3, 1]]) {
    const Ap = A.map((r) => r.slice()), Am = A.map((r) => r.slice()); Ap[i][j] += h; Am[i][j] -= h;
    close(dA[i][j], (L(Ap, B) - L(Am, B)) / (2 * h), 1e-6, `dA[${i}][${j}]`);
  }
  const Bp = B.map((r) => r.slice()), Bm = B.map((r) => r.slice()); Bp[1][2] += h; Bm[1][2] -= h;
  close(dB[1][2], (L(A, Bp) - L(A, Bm)) / (2 * h), 1e-6, 'dB[1][2]');
}
const full = trainLora(makeLoraTask(), { rank: 2, epochs: 200 });
assert.ok(full.after.test.accuracy > full.before.test.accuracy + 0.1, 'live LoRA learns');

console.log('lab ok', { dpoHeldOutMargin: preferenceMetrics(run.logits, pairs.held).mean_margin, grpoExpectedReward: expectedReward(grpo.logits, correct).toFixed(4), loraLive: [full.before.test.accuracy, full.after.test.accuracy] });
