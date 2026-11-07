// Computations that run live in the browser. Each one is a port of a
// repo module; lab.check.mjs asserts the exact quantities against the
// repo's results files during `npm run build`.

const r4 = (x) => Math.round(x * 1e4) / 1e4;

// ---------- seeded RNG (JS stream; it cannot reproduce NumPy's PCG64 draws)
export function mulberry32(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function gaussian(rand) {
  let u = 0; while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

// ---------- QLoRA accounting: port of level-3-deep/memory_study.py
export function memoryRow({ parameters, layers, hidden, rank = 16, baseBits = 4, matrices = 2 }) {
  const loraParameters = layers * matrices * (rank * hidden + hidden * rank);
  const full = parameters * 2;
  const adapter = loraParameters * 2;
  const qBase = parameters * baseBits / 8;
  return {
    lora_parameters: loraParameters,
    full_finetune_bytes: full,
    lora_adapter_bytes: adapter,
    lora_total_bytes: full + adapter,
    qlora_base_4bit_bytes: qBase,
    qlora_total_bytes: qBase + adapter,
    qlora_total_fraction_of_full: r4((qBase + adapter) / full),
    adapter_fraction_of_full: r4(adapter / full),
  };
}

// ---------- GRPO group advantages: port of level-1-learn/grpo_learn.py
export function groupAdvantages(rewards) {
  const n = rewards.length;
  const mean = rewards.reduce((a, b) => a + b, 0) / n;
  const adv = rewards.map((r) => r - mean);
  const std = Math.sqrt(adv.reduce((a, d) => a + d * d, 0) / n);
  return {
    baseline_mean: r4(mean),
    std_population: r4(std),
    advantages: adv.map(r4),
    advantages_normalized: adv.map((d) => (std === 0 ? 0 : r4(d / std))),
  };
}

// ---------- tabular policy helpers: ports of level-2-build/grpo_build.py
export const softmaxRow = (row) => { const m = Math.max(...row); const e = row.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
export const softmaxRows = (L) => L.map(softmaxRow);
export const klRow = (p, q) => p.reduce((a, pi, i) => a + pi * (Math.log(pi + 1e-12) - Math.log(q[i] + 1e-12)), 0);
export const meanKl = (L, refProbs) => softmaxRows(L).reduce((a, p, i) => a + klRow(p, refProbs[i]), 0) / L.length;
export const clippedRatio = (r, eps) => Math.min(1 + eps, Math.max(1 - eps, r));
export const surrogate = (r, A, eps) => Math.min(r * A, clippedRatio(r, eps) * A);
export const argmax = (row) => row.indexOf(Math.max(...row));
export const greedyAccuracy = (L, correct) => L.filter((row, p) => argmax(row) === correct[p]).length / L.length;
// Exact expected checker reward of a policy: mean probability of the correct action.
export const expectedReward = (L, correct) => softmaxRows(L).reduce((a, p, i) => a + p[correct[i]], 0) / L.length;

function sampleIndex(probs, rand) {
  let u = rand();
  for (let i = 0; i < probs.length; i++) { u -= probs[i]; if (u < 0) return i; }
  return probs.length - 1;
}

// One GRPO run: group sampling, normalized group advantages, clipped
// surrogate plus beta * KL to the frozen reference, inner epochs per batch.
export function trainGrpo({ init, correct, beta = 0.04, eps = 0.2, group = 8, iterations = 60, lr = 0.5, inner = 2, seed = 31, corrected = false }) {
  const rand = mulberry32(seed);
  let L = init.map((r) => r.slice());
  const ref = softmaxRows(L);
  const nA = L[0].length;
  const history = [{ iteration: 0, reward: expectedReward(L, correct), kl: 0, greedy: greedyAccuracy(L, correct) }];
  for (let it = 1; it <= iterations; it++) {
    const old = softmaxRows(L);
    const batch = old.map((probs, p) => {
      const acts = Array.from({ length: group }, () => sampleIndex(probs, rand));
      const rw = acts.map((a) => (a === correct[p] ? 1 : 0));
      // same rule as groupAdvantages, left unrounded for training
      const m = rw.reduce((x, y) => x + y, 0) / group;
      const sd = Math.sqrt(rw.reduce((x, y) => x + (y - m) ** 2, 0) / group);
      return { acts, adv: rw.map((y) => (sd === 0 ? 0 : (y - m) / sd)), old: probs };
    });
    // grpo_build.py divides prompt p's surrogate gradient by the running sample
    // count (p + 1) * group and leaves the KL gradient unaveraged; `corrected`
    // uses the gradient of the stated mean loss instead (see rigor/README.md).
    const div = (p) => (corrected ? L.length * group : (p + 1) * group);
    const klScale = corrected ? 1 / L.length : 1;
    for (let e = 0; e < inner; e++) {
      const probs = softmaxRows(L);
      const grad = L.map(() => new Array(nA).fill(0));
      batch.forEach(({ acts, adv, old: op }, p) => {
        acts.forEach((a, i) => {
          const ratio = probs[p][a] / (op[a] + 1e-12);
          const cr = clippedRatio(ratio, eps);
          const clippedActive = ratio * adv[i] >= cr * adv[i];
          const outside = ratio < 1 - eps || ratio > 1 + eps;
          if (clippedActive && outside) return;
          const coeff = (-adv[i] * ratio) / div(p);
          for (let k = 0; k < nA; k++) grad[p][k] += coeff * ((k === a ? 1 : 0) - probs[p][k]);
        });
        const kl = klRow(probs[p], ref[p]);
        for (let k = 0; k < nA; k++) grad[p][k] += klScale * beta * probs[p][k] * (Math.log(probs[p][k] + 1e-12) - Math.log(ref[p][k] + 1e-12) - kl);
      });
      L = L.map((row, p) => row.map((v, k) => v - lr * grad[p][k]));
    }
    history.push({ iteration: it, reward: expectedReward(L, correct), kl: meanKl(L, ref), greedy: greedyAccuracy(L, correct) });
  }
  return { logits: L, history };
}

// ---------- DPO: port of level-3-deep/dpo_deep.py
export function makePairs(correct, nA = 4) {
  const train = [], held = [];
  correct.forEach((c, p) => {
    train.push([p, c, (c + 1) % nA]);
    held.push([p, c, (c + 2) % nA], [p, c, (c + 3) % nA]);
  });
  return { train, held };
}
const logProbs = (L) => softmaxRows(L).map((r) => r.map((v) => Math.log(v + 1e-12)));
export function preferenceMetrics(L, pairs) {
  const lp = logProbs(L);
  const m = pairs.map(([p, c, r]) => lp[p][c] - lp[p][r]);
  const correct = m.filter((x) => x > 0).length;
  return { mean_margin: r4(m.reduce((a, b) => a + b, 0) / m.length), preference_accuracy: r4(correct / m.length), preference_correct: correct };
}
export function dpoLossAndGrad(L, ref, pairs, beta) {
  const lp = logProbs(L), rl = logProbs(ref);
  const grad = L.map((row) => row.map(() => 0));
  let loss = 0;
  for (const [p, c, r] of pairs) {
    const margin = (lp[p][c] - rl[p][c]) - (lp[p][r] - rl[p][r]);
    const sig = 1 / (1 + Math.exp(-beta * margin));
    loss += -Math.log(sig + 1e-12);
    const f = (-beta * (1 - sig)) / pairs.length;
    grad[p][c] += f; grad[p][r] -= f;
  }
  return { loss: loss / pairs.length, grad };
}
export function trainDpo({ init, pairs, beta = 0.1, lr = 0.5, steps = 300, every = 50 }) {
  let L = init.map((r) => r.slice());
  const history = [];
  for (let s = 0; s < steps; s++) {
    const { loss, grad } = dpoLossAndGrad(L, init, pairs.train, beta);
    if (s % every === 0 || s === steps - 1) history.push({ step: s, loss: r4(loss), train_mean_margin: preferenceMetrics(L, pairs.train).mean_margin, held_out_mean_margin: preferenceMetrics(L, pairs.held).mean_margin });
    L = L.map((row, p) => row.map((v, k) => v - lr * grad[p][k]));
  }
  return { logits: L, history, final_loss: r4(dpoLossAndGrad(L, init, pairs.train, beta).loss) };
}
// Rejected-only suppression probe: lower only the train rejected logit.
export function suppress(init, trainPairs, amount = 5) {
  const L = init.map((r) => r.slice());
  for (const [p, , r] of trainPairs) L[p][r] -= amount;
  return L;
}

// ---------- LoRA: port of level-1-learn/lora_learn.py on a JS seed stream
export const loraParams = (rank, inDim = 16, classes = 4) => inDim * rank + rank * classes;
const matmul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const randMat = (r, c, scale, rand) => Array.from({ length: r }, () => Array.from({ length: c }, () => gaussian(rand) * scale));

export function makeLoraTask(seed = 7, { inDim = 16, classes = 4, nTrain = 512, nTest = 128, trueRank = 2 } = {}) {
  const rand = mulberry32(seed);
  const x = randMat(nTrain + nTest, inDim, 1, rand);
  const base = randMat(inDim, classes, 0.5, rand);
  const target = matmul(randMat(inDim, trueRank, 0.6, rand), randMat(trueRank, classes, 0.6, rand));
  const W = base.map((row, i) => row.map((v, j) => v + 2 * target[i][j]));
  const y = matmul(x, W).map(argmax);
  return { xTrain: x.slice(0, nTrain), yTrain: y.slice(0, nTrain), xTest: x.slice(nTrain), yTest: y.slice(nTrain), base, seed };
}

export function loraEffective(base, A, B, scaling) {
  const AB = matmul(A, B);
  return base.map((row, i) => row.map((v, j) => v + scaling * AB[i][j]));
}

export function loraEvaluate(X, y, W) {
  const P = matmul(X, W).map(softmaxRow);
  let loss = 0, ok = 0;
  P.forEach((p, i) => { loss -= Math.log(p[y[i]] + 1e-12); if (argmax(p) === y[i]) ok++; });
  return { loss: loss / y.length, accuracy: ok / y.length, correct: ok, probs: P };
}

// Gradient of mean cross entropy with respect to A and B (base frozen).
export function loraGrad(X, y, base, A, B, scaling) {
  const { probs } = loraEvaluate(X, y, loraEffective(base, A, B, scaling));
  const n = y.length;
  const d = probs.map((p, i) => p.map((v, j) => (v - (j === y[i] ? 1 : 0)) / n));
  const dW = X[0].map((_, a) => d[0].map((__, c) => X.reduce((s, row, i) => s + row[a] * d[i][c], 0)));
  const dDelta = dW.map((row) => row.map((v) => v * scaling));
  const Bt = B[0].map((_, j) => B.map((row) => row[j]));
  const At = A[0].map((_, j) => A.map((row) => row[j]));
  return { dA: matmul(dDelta, Bt), dB: matmul(At, dDelta) };
}

export function trainLora(task, { rank = 2, lr = 0.5, epochs = 200, alpha = 4, adapterSeed = 5, every = 5 } = {}) {
  const scaling = alpha / rank;
  const rand = mulberry32(adapterSeed);
  let A = randMat(task.base.length, rank, 0.1, rand);
  let B = Array.from({ length: rank }, () => new Array(task.base[0].length).fill(0));
  const score = () => {
    const W = loraEffective(task.base, A, B, scaling);
    return { train: loraEvaluate(task.xTrain, task.yTrain, W), test: loraEvaluate(task.xTest, task.yTest, W) };
  };
  const s0 = score();
  const history = [{ epoch: 0, train: s0.train.accuracy, test: s0.test.accuracy, loss: s0.train.loss }];
  for (let e = 1; e <= epochs; e++) {
    const { dA, dB } = loraGrad(task.xTrain, task.yTrain, task.base, A, B, scaling);
    A = A.map((row, i) => row.map((v, j) => v - lr * dA[i][j]));
    B = B.map((row, i) => row.map((v, j) => v - lr * dB[i][j]));
    if (e % every === 0 || e === epochs) { const s = score(); history.push({ epoch: e, train: s.train.accuracy, test: s.test.accuracy, loss: s.train.loss }); }
  }
  const s1 = score();
  return { before: s0, after: s1, history, params: loraParams(rank, task.base.length, task.base[0].length), full: task.base.length * task.base[0].length, A, B, scaling };
}
