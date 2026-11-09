import { useMemo, useState, useEffect, useId, useRef } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import build from '../../level-2-build/results_build.json';
import deepFile from '../../level-3-deep/results_deep.json';
import rigor from '../../rigor/results_rigor.json';
import deepReadme from '../../level-3-deep/README.md?raw';
import report from '../../REPORT.md?raw';
import readme from '../../README.md?raw';
import {
  memoryRow, groupAdvantages, makePairs, preferenceMetrics, trainDpo, suppress, greedyAccuracy,
  expectedReward, trainGrpo, makeLoraTask, trainLora, loraParams, meanKl, softmaxRows,
} from './lab.js';

const deep = deepFile.deep;
const sft = build.sft_pipeline;
const grpoRec = build.grpo_track;
const dpoRec = deep.dpo;
const mem = deep.qlora_memory_study;
const probe = deep.reward_hacking_probe;
const INIT = rigor.site_fixtures.init_policy_logits;
const CORRECT = grpoRec.config.correct_actions;
const PAIRS = makePairs(CORRECT);

const REPO = 'https://github.com/JesunAhmadUshno/finetune-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const f4 = (x) => Number(x).toFixed(4);
const f2 = (x) => Number(x).toFixed(2);
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const sgn = (x, d = 4) => `${x >= 0 ? '+' : ''}${Number(x).toFixed(d)}`;
const ci = (s) => `${f4(s.ci95[0])} to ${f4(s.ci95[1])}`;

const SECTIONS = [['top', 'Overview', '◎'], ['lora', 'LoRA', '⊕'], ['sft', 'SFT', '▦'], ['grpo', 'GRPO', '↗'], ['dpo', 'DPO', '⇅'], ['probe', 'Probe', '⚠'], ['qlora', 'Memory', '▤'], ['limits', 'Limits', '✎'], ['reproduce', 'Reproduce', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">Finetune Lab</a>
      <a className="m" href={blob('level-1-learn/README.md')}>Learn</a>
      <a className="m" href={blob('level-2-build/README.md')}>Build</a>
      <a className="m" href={blob('level-3-deep/README.md')}>Deep</a>
      <a className="m" href={blob('rigor/README.md')}>Rigor</a>
      <span className="spacer" />
      <span className="status"><i /> CPU · NumPy · deterministic</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined} aria-label={label}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

const Live = ({ children = 'live · JS seed stream' }) => <span className="pill good">● {children}</span>;
const Rec = ({ src }) => <span className="pill info">recorded · {src}</span>;

// Hand-written line chart: series of [x, y] points, optional hover readout.
function Lines({ series, xMax, yMin = 0, yMax, label, xLabel, yFmt = f2, yTicks, W = 480, H = 230 }) {
  const pad = { l: 40, r: 10, t: 12, b: 26 };
  const [hx, setHx] = useState(null);
  const ref = useRef(null);
  const sx = (x) => pad.l + (x / xMax) * (W - pad.l - pad.r);
  const sy = (y) => pad.t + (1 - (Math.min(yMax, Math.max(yMin, y)) - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const ticks = yTicks ?? [yMin, (yMin + yMax) / 2, yMax];
  const move = (e) => {
    const r = ref.current.getBoundingClientRect();
    const x = (((e.clientX - r.left) / r.width) * W - pad.l) / (W - pad.l - pad.r) * xMax;
    setHx(Math.max(0, Math.min(xMax, x)));
  };
  const nearest = (pts) => pts.reduce((b, p) => (Math.abs(p[0] - hx) < Math.abs(b[0] - hx) ? p : b), pts[0]);
  return (
    <>
      <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onMouseMove={move} onMouseLeave={() => setHx(null)}>
        {ticks.map((t) => <g key={t}><line className="axis" x1={pad.l} x2={W - pad.r} y1={sy(t)} y2={sy(t)} /><text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{yFmt(t)}</text></g>)}
        {series.map((s) => (
          <g key={s.name}>
            {s.line !== false && <polyline fill="none" stroke={s.color} strokeWidth={s.width ?? 2.2} strokeDasharray={s.dashed ? '5 4' : ''} points={s.pts.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} />}
            {s.dots && s.pts.map(([x, y]) => <circle key={x} cx={sx(x)} cy={sy(y)} r="4" fill="var(--surface-strong)" stroke={s.color} strokeWidth="2" />)}
          </g>
        ))}
        {hx != null && <line x1={sx(hx)} x2={sx(hx)} y1={pad.t} y2={H - pad.b} stroke="var(--muted)" strokeDasharray="3 3" />}
        <text x={pad.l} y={H - 6}>0</text><text x={W - pad.r} y={H - 6} textAnchor="end">{xLabel} {xMax}</text>
      </svg>
      <div className="legend">{series.map((s) => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
      <div className="tip" aria-live="polite">
        {hx == null ? 'Hover the chart to read values.' : series.map((s) => { const p = nearest(s.pts); return <span key={s.name} style={{ marginRight: 14 }}>{s.name} @ {p[0]}: <b>{yFmt(p[1])}</b></span>; })}
      </div>
    </>
  );
}

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Lora />
      <Sft />
      <Grpo />
      <Dpo />
      <Probe />
      <Memory />
      <Limits />
      <Reproduce />
      <Dock />
      <footer>
        Code: Apache-2.0. Every recorded number on this page is imported from the repo's results JSON files at build time; the live panels run in your browser and are checked against those files by <code>npm run build</code>.
      </footer>
    </div>
  );
}

function Hero() {
  const t = sft.before_training.test, a = sft.after_training.test;
  const big = mem.rows.find((r) => r.name === 'large-7B');
  return (
    <header className="card hero window" id="top" data-title="finetune-lab.md">
      <div className="head">
        <p className="eyebrow">Post-training lab · LoRA · SFT · GRPO · DPO · QLoRA</p>
        <span className="pill warn">● synthetic NumPy · CPU · no real LLM</span>
      </div>
      <h1>Teaching a model <em>manners</em>, one tiny matrix at a time.</h1>
      <p className="sub">
        On synthetic CPU tasks, a {sft.config.trainable_parameters}-parameter LoRA adapter lifts held-out accuracy from {f2(t.accuracy)} to {f2(a.accuracy)},
        GRPO lifts checker reward from {f2(grpoRec.before_training.mean_reward)} to {f2(grpoRec.after_training.mean_reward)}, and DPO turns a negative held-out preference margin ({f4(dpoRec.held_out_pairs_before.mean_margin)}) positive ({f4(dpoRec.held_out_pairs_after.mean_margin)}) while keeping the checker task.
        Every run is a teaching-scale model: a {sft.data_card.in_dim} × {sft.data_card.num_classes} linear layer and a tabular {grpoRec.config.n_prompts} × {grpoRec.config.n_actions} policy. Rerun across {rigor.seeds.n_seeds} seeds, the typical GRPO reward is {f4(rigor.seeds.grpo.summary.final_mean_reward.mean)} (95% CI {ci(rigor.seeds.grpo.summary.final_mean_reward)}).
      </p>
      <p className="hint" style={{ marginTop: 10 }}>Scope, from the results files: “{build.note}”</p>
      <div className="actions">
        <a className="btn primary" href="#lora">Train a LoRA adapter live</a>
        <a className="btn ghost" href="#limits">Read the honest limits</a>
        <span className="hint">Reproduce with <code>python3 level-3-deep/run_deep.py</code></span>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{f4(a.accuracy)}</div><div className="cap">SFT held-out test accuracy, from {f4(t.accuracy)} ({t.correct} → {a.correct} of {a.n})</div></div>
        <div className="tile"><div className="num">{f4(grpoRec.after_training.mean_reward)}</div><div className="cap">GRPO fresh-sample checker reward, from {f4(grpoRec.before_training.mean_reward)}</div></div>
        <div className="tile"><div className="num">{f4(dpoRec.held_out_pairs_after.mean_margin)}</div><div className="cap">DPO held-out preference margin, from {f4(dpoRec.held_out_pairs_before.mean_margin)}</div></div>
        <div className="tile"><div className="num">{f2(big.qlora_total_bytes / 1e9)} GB</div><div className="cap">7B-scale 4-bit base + rank {big.rank} adapter, against {f2(big.full_finetune_bytes / 1e9)} GB fp16. Accounting, not measured</div></div>
      </div>
    </header>
  );
}

function Lora() {
  const L = learn.lora;
  const [rank, setRank] = useState(L.lora.rank);
  const [lr, setLr] = useState(L.lora.learning_rate);
  const [epochs, setEpochs] = useState(L.lora.epochs);
  const [seed, setSeed] = useState(L.task.data_seed);
  const task = useMemo(() => makeLoraTask(seed), [seed]);
  const res = useMemo(() => trainLora(task, { rank, lr, epochs, alpha: L.lora.alpha }), [task, rank, lr, epochs, L.lora.alpha]);
  return (
    <section id="lora" className="card window reveal" data-title="lora_learn.py — live trainer">
      <div className="head">
        <div>
          <p className="eyebrow">LoRA playground</p>
          <h2>Freeze the base. Train only <em>A</em> and <em>B</em>.</h2>
          <p className="sub">
            A frozen {L.task.in_dim} × {L.task.num_classes} base layer plus a rank-r adapter, W = W₀ + (α / r)·A·B, trained by full-batch gradient descent on A and B only.
            B starts at zero, so the untrained model is exactly the base. The label rule hides a rank-{L.lora.rank} change from the base, so a smaller rank cannot fully express it.
            This trainer is a JavaScript port of <a href={blob('level-1-learn/lora_learn.py')}>lora_learn.py</a> running on a JS random stream, so its data and numbers differ from the recorded NumPy run shown beside it.
          </p>
        </div>
        <Live />
      </div>
      <div className="controls">
        <Range label="Adapter rank r" value={rank} min={1} max={4} step={1} onChange={setRank} show={`${rank} (${loraParams(rank)} params)`} />
        <Range label="Learning rate" value={lr} min={0.05} max={1.5} step={0.05} onChange={setLr} show={f2(lr)} />
        <Range label="Epochs" value={epochs} min={10} max={400} step={10} onChange={setEpochs} show={epochs} />
        <div className="field"><label htmlFor="lora-seed">Task seed (JS stream)</label><input id="lora-seed" type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 1)} /></div>
      </div>
      <div className="grid2">
        <div>
          <h3>Accuracy during training (live)</h3>
          <Lines label="Live LoRA train and test accuracy by epoch" xMax={epochs} xLabel="epoch" yMin={0} yMax={1} yTicks={[0, 0.25, 0.5, 0.75, 1]}
            series={[{ name: 'test accuracy', color: 'var(--primary)', pts: res.history.map((h) => [h.epoch, h.test]) }, { name: 'train accuracy', color: 'var(--warn)', dashed: true, pts: res.history.map((h) => [h.epoch, h.train]) }]} />
        </div>
        <div>
          <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0 }}>
            <div className="tile"><div className="num">{f4(res.after.test.accuracy)}</div><div className="cap">Live test accuracy after training, from {f4(res.before.test.accuracy)}</div></div>
            <div className="tile"><div className="num">{res.params} / {res.full}</div><div className="cap">Live trainable parameters / full layer ({pct(res.params / res.full, 0)})</div></div>
          </div>
          <div className="tile" style={{ marginTop: 12 }}>
            <div className="head" style={{ marginBottom: 6 }}><h3>Recorded Learn run</h3><Rec src="results_learn.json" /></div>
            <p className="hint" style={{ margin: 0 }}>
              NumPy, rank {L.lora.rank}, {L.lora.trainable_parameters} of {L.lora.full_finetune_parameters} parameters, {L.lora.epochs} epochs, lr {L.lora.learning_rate}:
              test accuracy <b>{f4(L.before_training.test_accuracy)} → {f4(L.after_training.test_accuracy)}</b> ({L.before_training.test_correct} → {L.after_training.test_correct} of {L.task.n_test}),
              test loss {f4(L.before_training.test_loss)} → {f4(L.after_training.test_loss)}. Base weights unchanged: {String(L.lora.base_weights_unchanged)}.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Heatmap({ m, title }) {
  const n = m.length, c = 46, off = 28;
  const max = Math.max(...m.flat());
  return (
    <figure style={{ margin: 0 }}>
      <svg className="chart" viewBox={`0 0 ${off + n * c + 4} ${off + n * c + 4}`} role="img" aria-label={`${title}: rows are true class, columns are predicted class`} style={{ maxWidth: 260 }}>
        {m.map((row, i) => row.map((v, j) => (
          <g key={`${i}-${j}`}>
            <rect x={off + j * c} y={off + i * c} width={c - 3} height={c - 3} rx="6" fill={i === j ? 'var(--good)' : 'var(--bad)'} opacity={0.08 + 0.85 * (v / max)} />
            <text x={off + j * c + (c - 3) / 2} y={off + i * c + (c - 3) / 2 + 4} textAnchor="middle" style={{ fill: 'var(--text)', fontSize: 13 }}>{v}</text>
          </g>
        )))}
        {m.map((_, k) => <g key={k}><text x={off + k * c + (c - 3) / 2} y={18} textAnchor="middle">p{k}</text><text x={14} y={off + k * c + c / 2 + 2} textAnchor="middle">t{k}</text></g>)}
      </svg>
      <figcaption className="hint">{title}</figcaption>
    </figure>
  );
}

function Sft() {
  const [split, setSplit] = useState('test');
  const b = sft.before_training[split], a = sft.after_training[split];
  const base = rigor.sft_baselines.rows;
  const seeds = rigor.seeds.sft.summary;
  return (
    <section id="sft" className="card window reveal" data-title="pipeline.py — eval harness">
      <div className="head">
        <div>
          <p className="eyebrow">SFT pipeline</p>
          <h2>Where the errors were, and where they <em>went</em>.</h2>
          <p className="sub">
            Minibatch LoRA training (batch {sft.config.batch_size}, {sft.config.epochs} epochs, rank {sft.config.rank}) on {sft.data_card.train.n} train, {sft.data_card.val.n} validation and {sft.data_card.test.n} held-out test vectors.
            The validation slice is the slice Learn used as its test set, so before-training validation accuracy ({f4(sft.before_training.val.accuracy)}) equals Learn's before test accuracy. Compare before and after inside one split.
          </p>
        </div>
        <div style={{ display: 'grid', gap: 8, justifyItems: 'end' }}>
          <Rec src="results_build.json" />
          <div className="seg" role="group" aria-label="Choose data split">
            {[['train', 'Train'], ['val', 'Validation'], ['test', 'Held-out test']].map(([k, l]) => <button key={k} aria-pressed={split === k} onClick={() => setSplit(k)}>{l}</button>)}
          </div>
        </div>
      </div>
      <div className="grid2">
        <div className="heat-pair">
          <Heatmap m={b.confusion} title={`Before: ${f4(b.accuracy)} (${b.correct} of ${b.n})`} />
          <Heatmap m={a.confusion} title={`After: ${f4(a.accuracy)} (${a.correct} of ${a.n})`} />
        </div>
        <div>
          <h3>Per-class accuracy, {split === 'val' ? 'validation' : split}</h3>
          <div className="bars" style={{ marginTop: 12 }}>
            {b.per_class_accuracy.map((v, k) => (
              <div className="bar-row" key={k}>
                <div className="lab"><span>Class {k} ({sft.data_card[split].class_counts[k]} examples)</span><b>{f4(v)} → {f4(a.per_class_accuracy[k])}</b></div>
                <div className="track"><div className="fill" style={{ width: pct(a.per_class_accuracy[k]), background: 'var(--primary)' }} /><div className="ref" style={{ left: pct(v) }} title={`before ${f4(v)}`} /></div>
              </div>
            ))}
          </div>
          <p className="hint" style={{ marginTop: 10 }}>Bar: after training. Tick: before training. Loss {f4(b.loss)} → {f4(a.loss)}.</p>
        </div>
      </div>
      <div className="head" style={{ marginTop: 24, marginBottom: 8 }}>
        <h3>Baselines across {rigor.seeds.n_seeds} seeds: full fine-tune against LoRA ranks</h3>
        <Rec src="results_rigor.json" />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Model</th><th>Trainable params</th><th>Test acc, final epoch (mean)</th><th>95% bootstrap CI</th><th>Best-val epoch (mean)</th></tr></thead>
          <tbody>
            {base.map((r) => (
              <tr key={r.rank}><td>{r.rank === 'full' ? 'Full fine-tune' : `LoRA rank ${r.rank}`}{r.rank === sft.config.rank ? ' (recorded setting)' : ''}</td><td>{r.trainable_parameters}</td><td>{f4(r.final_test_accuracy.mean)}</td><td>{ci(r.final_test_accuracy)}</td><td>{f4(r.best_val_test_accuracy.mean)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint" style={{ marginTop: 10 }}>
        On this toy the full {base[0].trainable_parameters}-weight fine-tune beats the rank {sft.config.rank} adapter. Picking the best-validation epoch instead of the last lifts rank {sft.config.rank} test accuracy from {f4(seeds.final_test_accuracy.mean)} to {f4(seeds.best_val_test_accuracy.mean)} on average; the recorded run ships the final epoch.
      </p>
    </section>
  );
}

function AdvantageCalc() {
  const examples = learn.grpo_examples;
  const [text, setText] = useState(examples.prompt_b_graded.rewards.join(', '));
  const rewards = text.split(/[\s,]+/).filter(Boolean).map(Number);
  const valid = rewards.length > 0 && rewards.every(Number.isFinite);
  const out = valid ? groupAdvantages(rewards) : null;
  const match = valid && Object.entries(examples).find(([, ex]) => ex.rewards.length === rewards.length && ex.rewards.every((r, i) => r === rewards[i]));
  const agrees = match && JSON.stringify(out.advantages_normalized) === JSON.stringify(match[1].advantages_normalized) && out.baseline_mean === match[1].baseline_mean;
  return (
    <div>
      <div className="head" style={{ marginBottom: 8 }}><h3>Group advantage calculator</h3><Live>live · exact port</Live></div>
      <div className="field"><label htmlFor="rewards">Rewards for one prompt's group (comma separated)</label><input id="rewards" type="text" className="text" value={text} onChange={(e) => setText(e.target.value)} /></div>
      <div className="actions" style={{ marginTop: 8 }}>
        {Object.entries(examples).map(([k, ex]) => <button key={k} className="btn ghost" onClick={() => setText(ex.rewards.join(', '))}>{k.replace('prompt_', '').replaceAll('_', ' ')}</button>)}
      </div>
      {out ? (
        <div style={{ overflowX: 'auto', marginTop: 12 }}>
          <table>
            <tbody>
              <tr><th>Baseline (group mean)</th><td>{f4(out.baseline_mean)}</td></tr>
              <tr><th>Population std</th><td>{f4(out.std_population)}{out.std_population === 0 ? ' (all equal, so advantages are 0)' : ''}</td></tr>
              <tr><th>Advantages r − mean</th><td>{out.advantages.map(f4).join(', ')}</td></tr>
              <tr><th>Normalized</th><td>{out.advantages_normalized.map(f4).join(', ')}</td></tr>
            </tbody>
          </table>
        </div>
      ) : <p className="hint">Enter numbers separated by commas.</p>}
      <p className="hint" style={{ marginTop: 8 }}>{match ? (agrees ? `✓ Matches the recorded ${match[0]} example in results_learn.json.` : '✗ Differs from the recorded example.') : 'Load an example to compare against results_learn.json.'}</p>
    </div>
  );
}

function Grpo() {
  const c = grpoRec.config;
  const [eps, setEps] = useState(c.eps_clip);
  const [beta, setBeta] = useState(c.beta_kl);
  const [group, setGroup] = useState(c.group_size);
  const [iters, setIters] = useState(c.iterations);
  const [corrected, setCorrected] = useState(false);
  const run = useMemo(() => trainGrpo({ init: INIT, correct: CORRECT, eps, beta, group, iterations: iters, lr: c.learning_rate, inner: c.inner_epochs, corrected }), [eps, beta, group, iters, corrected, c]);
  const last = run.history.at(-1);
  const sweep = rigor.grpo_beta_budget.rows;
  const colors = ['var(--bad)', 'var(--warn)', 'var(--primary)', 'var(--good)'];
  return (
    <section id="grpo" className="card window reveal" data-title="grpo_build.py — group relative policy optimization">
      <p className="eyebrow">GRPO trainer</p>
      <h2>No critic. The group <em>is</em> the baseline.</h2>
      <p className="sub">
        For each of {c.n_prompts} prompts the policy samples a group of actions, a checker scores each (1 if correct, else 0), and each sample's advantage is its reward minus the group mean, divided by the group's standard deviation.
        The update maximizes a clipped surrogate (ε) minus β times the KL divergence to the frozen starting policy.
      </p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <AdvantageCalc />
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Live GRPO run from the recorded starting policy</h3><Live /></div>
          <div className="controls" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <Range label="Clip ε" value={eps} min={0.05} max={0.5} step={0.05} onChange={setEps} show={f2(eps)} />
            <Range label="KL β" value={beta} min={0} max={0.2} step={0.01} onChange={setBeta} show={f2(beta)} />
            <Range label="Group size" value={group} min={2} max={16} step={1} onChange={setGroup} show={group} />
            <Range label="Iterations" value={iters} min={10} max={120} step={10} onChange={setIters} show={iters} />
          </div>
          <label className="check"><input type="checkbox" checked={corrected} onChange={(e) => setCorrected(e.target.checked)} /> Use the corrected gradient normalization (see rigor/README.md)</label>
          <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 12 }}>
            <div className="tile"><div className="num">{f4(last.reward)}</div><div className="cap">Live expected reward after {iters} iterations (exact from policy probabilities), from {f4(run.history[0].reward)}</div></div>
            <div className="tile"><div className="num">{f4(last.kl)}</div><div className="cap">Live mean KL to the start policy; greedy accuracy {f4(last.greedy)}</div></div>
          </div>
        </div>
      </div>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <h3>Reward: live curve against the recorded 60-iteration run</h3>
          <Lines label="GRPO reward by iteration, live expected reward and recorded sampled reward" xMax={Math.max(iters, c.iterations)} xLabel="iteration" yMin={0} yMax={1} yTicks={[0, 0.25, 0.5, 0.75, 1]}
            series={[{ name: 'live expected reward', color: 'var(--primary)', pts: run.history.map((h) => [h.iteration, h.reward]) },
              { name: 'recorded eval reward (32 samples per prompt)', color: 'var(--text)', dots: true, line: false, pts: [[0, grpoRec.before_training.mean_reward], ...grpoRec.history.map((h) => [h.iteration, h.eval_mean_reward])] }]} />
        </div>
        <div>
          <h3>KL to reference: live against recorded</h3>
          <Lines label="GRPO mean KL to reference by iteration, live and recorded" xMax={Math.max(iters, c.iterations)} xLabel="iteration" yMin={0} yMax={1.6} yTicks={[0, 0.4, 0.8, 1.2, 1.6]}
            series={[{ name: 'live KL', color: 'var(--warn)', pts: run.history.map((h) => [h.iteration, h.kl]) },
              { name: `recorded KL (final ${f4(grpoRec.final_mean_kl_to_ref)})`, color: 'var(--text)', dots: true, line: false, pts: grpoRec.history.map((h) => [h.iteration, h.mean_kl_to_ref]) }]} />
        </div>
      </div>
      <p className="hint">The live run uses the recorded starting logits but a JS sampling stream, so its curve differs from the recorded NumPy run. Recorded points are fresh-sample means; the live curve is the exact expected reward.</p>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>β sweep and budget, {rigor.seeds.n_seeds}-seed means</h3><Rec src="results_rigor.json" /></div>
          <Lines label="GRPO eval reward against iterations for four KL beta values, mean of ten seeds" xMax={rigor.grpo_beta_budget.checkpoints.at(-1)} xLabel="iteration" yMin={0.4} yMax={1} yTicks={[0.4, 0.6, 0.8, 1]}
            series={sweep.map((r, i) => ({ name: `β ${r.beta}`, color: colors[i], dots: true, pts: r.checkpoints.map((k) => [k.iteration, k.mean_reward.mean]) }))} />
        </div>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Ablations, {rigor.seeds.n_seeds} seeds</h3><Rec src="results_rigor.json" /></div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Variant</th><th>Reward (mean)</th><th>95% CI</th><th>KL</th></tr></thead>
              <tbody>{rigor.grpo_ablations.rows.map((r) => <tr key={r.variant}><td>{r.variant}</td><td>{f4(r.final_mean_reward.mean)}</td><td>{ci(r.final_mean_reward)}</td><td>{f4(r.final_kl_to_ref.mean)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="hint">Tabular stand-ins for the GRPO against PPO question; no critic is trained, so none of these is PPO.</p>
        </div>
      </div>
    </section>
  );
}

function Dpo() {
  const cfg = deep.config;
  const [beta, setBeta] = useState(cfg.beta);
  const [lr, setLr] = useState(cfg.learning_rate);
  const [steps, setSteps] = useState(cfg.steps);
  const run = useMemo(() => trainDpo({ init: INIT, pairs: PAIRS, beta, lr, steps, every: Math.max(1, Math.round(steps / 30)) }), [beta, lr, steps]);
  const tr = preferenceMetrics(run.logits, PAIRS.train), ho = preferenceMetrics(run.logits, PAIRS.held);
  const isRec = beta === cfg.beta && lr === cfg.learning_rate && steps === cfg.steps;
  const fc = deep[Object.keys(deep).find((k) => k.endsWith('_checks'))]; // the Deep retention checks block
  const mis = rigor.misaligned_retention.rows;
  const yMax = Math.max(2, ...run.history.map((h) => h.train_mean_margin));
  return (
    <section id="dpo" className="card window reveal" data-title="dpo_deep.py — preference optimization">
      <div className="head">
        <div>
          <p className="eyebrow">DPO and task retention</p>
          <h2>Prefer the right answer, and check what it <em>costs</em>.</h2>
          <p className="sub">
            Pair rule: {cfg.pair_rule}. The held-out rejected actions never appear in training.
            The gradient of each pair's reference-adjusted margin is exactly one-hot(chosen) − one-hot(rejected). This port starts from the recorded policy, and at the recorded settings it reproduces results_deep.json exactly (checked at build).
          </p>
        </div>
        <Live>live · deterministic port</Live>
      </div>
      <div className="controls">
        <Range label="β" value={beta} min={0.05} max={0.5} step={0.05} onChange={setBeta} show={f2(beta)} />
        <Range label="Learning rate" value={lr} min={0.1} max={1.5} step={0.1} onChange={setLr} show={f2(lr)} />
        <Range label="Steps" value={steps} min={50} max={1000} step={50} onChange={setSteps} show={steps} />
      </div>
      <div className="grid2">
        <div>
          <h3>Margins during training {isRec ? '(recorded settings)' : '(live settings)'}</h3>
          <Lines label="DPO train and held-out margins by step, live and recorded" xMax={Math.max(steps, cfg.steps)} xLabel="step" yMin={-0.5} yMax={yMax} yTicks={[-0.5, 0, yMax / 2, yMax]}
            series={[{ name: 'live train margin', color: 'var(--warn)', pts: run.history.map((h) => [h.step, h.train_mean_margin]) },
              { name: 'live held-out margin', color: 'var(--primary)', pts: run.history.map((h) => [h.step, h.held_out_mean_margin]) },
              { name: 'recorded held-out margin', color: 'var(--text)', dots: true, line: false, pts: dpoRec.history.map((h) => [h.step, h.held_out_mean_margin]) }]} />
        </div>
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Measure</th><th>Live</th><th>Recorded</th></tr></thead>
              <tbody>
                <tr><td>Train margin</td><td>{f4(tr.mean_margin)}</td><td>{f4(dpoRec.train_pairs_after.mean_margin)}</td></tr>
                <tr><td>Held-out margin</td><td>{f4(ho.mean_margin)}</td><td>{f4(dpoRec.held_out_pairs_after.mean_margin)}</td></tr>
                <tr><td>Held-out accuracy</td><td>{f4(ho.preference_accuracy)} ({ho.preference_correct} of {PAIRS.held.length})</td><td>{f4(dpoRec.held_out_pairs_after.preference_accuracy)}</td></tr>
                <tr><td>DPO loss</td><td>{f4(run.final_loss)}</td><td>{f4(dpoRec.loss_after)}</td></tr>
                <tr><td>Greedy checker accuracy</td><td>{f4(greedyAccuracy(run.logits, CORRECT))}</td><td>{f4(dpoRec.task_after.greedy_accuracy)}</td></tr>
                <tr><td>Checker reward</td><td>{f4(expectedReward(run.logits, CORRECT))} expected</td><td>{f4(dpoRec.task_after.mean_reward)} sampled</td></tr>
                <tr><td>KL to start policy</td><td>{f4(meanKl(run.logits, softmaxRows(INIT)))}</td><td>{f4(fc.mean_kl_after_to_before)}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="hint" style={{ marginTop: 8 }}>
            DPO's recorded checker reward ({f4(dpoRec.task_after.mean_reward)}) sits far below GRPO's ({f4(grpoRec.after_training.mean_reward)}). That is not a regression: DPO optimizes preference margins at β {cfg.beta} for {cfg.steps} steps, not reward. Raise β, learning rate or steps and watch the reward climb.
          </p>
        </div>
      </div>
      <div className="grid2" style={{ marginTop: 20 }}>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Retention check on the checker task</h3><Rec src="results_deep.json" /></div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Measure</th><th>Before</th><th>After</th><th>Change</th></tr></thead>
              <tbody>
                <tr><td>Greedy accuracy</td><td>{f4(fc.before.greedy_accuracy)}</td><td>{f4(fc.after.greedy_accuracy)}</td><td className="ok">{sgn(fc.change.greedy_accuracy, 3)}</td></tr>
                <tr><td>Mean reward</td><td>{f4(fc.before.mean_reward)}</td><td>{f4(fc.after.mean_reward)}</td><td className="ok">{sgn(fc.change.mean_reward, 3)}</td></tr>
                <tr><td>Untouched-action order kept</td><td colSpan="3">{fc.untouched_action_order_kept} of {fc.untouched_action_order_prompts} prompts</td></tr>
              </tbody>
            </table>
          </div>
          <p className="hint"><span className="pill warn">weak by construction</span> The preferences agree with the checker and tabular rows share no weights, so this check could only show no task loss.</p>
        </div>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Misaligned preferences, {rigor.seeds.n_seeds} seeds</h3><Rec src="results_rigor.json" /></div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Prompts preferring a wrong action</th><th>Greedy after</th><th>On those prompts</th><th>On the rest</th></tr></thead>
              <tbody>{mis.map((r) => <tr key={r.k_misaligned}><td>{r.k_misaligned} of {CORRECT.length}</td><td>{f4(r.greedy_accuracy.mean)}</td><td>{r.greedy_on_misaligned_prompts ? f4(r.greedy_on_misaligned_prompts.mean) : '—'}</td><td>{f4(r.greedy_on_aligned_prompts.mean)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="hint">The stronger test: where preferences contradict the checker, DPO follows the preferences and the task is lost on exactly those prompts. Damage cannot spread here because rows are independent.</p>
          <h3 style={{ marginTop: 16 }}>With shared weights, damage spreads</h3>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>Misaligned train prompts</th><th>Unseen prompts, greedy before → after</th><th>Aligned train prompts after</th></tr></thead>
              <tbody>{rigor.shared_weight_dpo.rows.map((r) => <tr key={r.k_misaligned}><td>{r.k_misaligned}</td><td>{f4(r.unseen_greedy_before.mean)} → {f4(r.unseen_greedy_after.mean)}</td><td>{f4(r.aligned_train_greedy_after.mean)}</td></tr>)}</tbody>
            </table>
          </div>
          <p className="hint">A linear policy (logits = φ·Θ) trained on some prompts and scored on unseen ones, {rigor.seeds.n_seeds}-seed means. Aligned DPO generalizes to new prompts; contradicting preferences erase that gain and hurt the aligned prompts too.</p>
        </div>
      </div>
    </section>
  );
}

function Probe() {
  const [amount, setAmount] = useState(5);
  const sup = suppress(INIT, PAIRS.train, amount);
  const live = { train: preferenceMetrics(sup, PAIRS.train).preference_accuracy, held: preferenceMetrics(sup, PAIRS.held).preference_accuracy, greedy: greedyAccuracy(sup, CORRECT) };
  const rows = [
    [`Constant policy: ${probe.constant_policy_probe.description}`, probe.constant_policy_probe],
    [`Suppression: ${probe.rejected_only_suppression_probe.description}`, probe.rejected_only_suppression_probe],
    ['Trained DPO policy', probe.trained_policy_check],
  ];
  return (
    <section id="probe" className="card window reveal" data-title="probe.log — reward hacking">
      <div className="head">
        <div>
          <p className="eyebrow">Reward-hacking probe</p>
          <h2>A policy can win the training contrast <em>without</em> learning the task.</h2>
          <p className="sub">Push down only the rejected action of each train pair and never raise the chosen one. Train preference accuracy goes to 1, but held-out contrasts and the task score do not move. Only those extra surfaces catch it.</p>
        </div>
        <Rec src="results_deep.json" />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Probe</th><th>Train pref. acc</th><th>Held-out pref. acc</th><th>Greedy acc</th><th>Mean reward</th></tr></thead>
          <tbody>{rows.map(([k, r]) => <tr key={k}><td>{k}</td><td>{f4(r.train_preference_accuracy)}</td><td>{f4(r.held_out_preference_accuracy)}</td><td>{f4(r.greedy_accuracy)}</td><td>{r.mean_reward != null ? f4(r.mean_reward) : '—'}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="hint">Trained policy flagged as hacking or overfitting: <b>{String(probe.trained_policy_check.hacking_or_overfit_flagged)}</b> (rule: {probe.trained_policy_check.description}; pre-DPO greedy {f4(probe.trained_policy_check.greedy_accuracy_before)}).</p>
      <div className="head" style={{ marginTop: 20, marginBottom: 8 }}><h3>Try it: suppression size</h3><Live>live · deterministic port</Live></div>
      <div className="grid2">
        <Range label="Amount subtracted from each train rejected logit" value={amount} min={0} max={10} step={0.5} onChange={setAmount} show={f2(amount)} />
        <div className="bars">
          {[['Train preference accuracy', live.train, 'var(--bad)'], ['Held-out preference accuracy', live.held, 'var(--primary)'], ['Greedy task accuracy', live.greedy, 'var(--good)']].map(([k, v, col]) => (
            <div className="bar-row" key={k}><div className="lab"><span>{k}</span><b>{f4(v)}</b></div><div className="track"><div className="fill" style={{ width: pct(v), background: col }} /></div></div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Memory() {
  const [preset, setPreset] = useState('large-7B');
  const p0 = mem.rows.find((r) => r.name === preset);
  const [params, setParams] = useState(p0.parameters / 1e6);
  const [layers, setLayers] = useState(p0.layers);
  const [hidden, setHidden] = useState(p0.hidden);
  const [rank, setRank] = useState(mem.assumptions.rank);
  const [bits, setBits] = useState(mem.assumptions.bytes_4bit * 8);
  const pick = (name) => { const r = mem.rows.find((x) => x.name === name); setPreset(name); setParams(r.parameters / 1e6); setLayers(r.layers); setHidden(r.hidden); setRank(r.rank); setBits(mem.assumptions.bytes_4bit * 8); };
  const row = memoryRow({ parameters: Math.round(params * 1e6), layers, hidden, rank, baseBits: bits });
  const bars = [['Full fine-tune, fp16 weights', row.full_finetune_bytes, 'var(--bad)'], ['LoRA: fp16 base + adapter', row.lora_total_bytes, 'var(--warn)'], [`QLoRA: ${bits}-bit base + adapter`, row.qlora_total_bytes, 'var(--good)']];
  const top = Math.max(...bars.map((b) => b[1]));
  const keys = ['full_finetune_bytes', 'lora_adapter_bytes', 'lora_total_bytes', 'qlora_base_4bit_bytes', 'qlora_total_bytes', 'qlora_total_fraction_of_full'];
  const recomputed = mem.rows.map((r) => ({ r, got: memoryRow({ parameters: r.parameters, layers: r.layers, hidden: r.hidden, rank: r.rank }) }));
  return (
    <section id="qlora" className="card window reveal" data-title="memory_study.py — QLoRA accounting">
      <div className="head">
        <div>
          <p className="eyebrow">QLoRA memory calculator</p>
          <h2>Weight bytes on paper: a quarter of <em>full</em>.</h2>
          <p className="sub">Bytes from parameter counts: fp16 at {mem.assumptions.bytes_fp16} B per parameter, a {mem.assumptions.bytes_4bit * 8}-bit base at {mem.assumptions.bytes_4bit} B, and a rank-r adapter on the {mem.assumptions.adapted_matrices} ({mem.assumptions.matrices_per_layer} matrices per layer) at fp16.</p>
        </div>
        <span className="pill warn">accounting, not measured GPU memory</span>
      </div>
      <div className="controls">
        <div className="field"><label htmlFor="preset">Recorded model size</label>
          <select id="preset" value={preset} onChange={(e) => pick(e.target.value)}>{mem.rows.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}</select>
        </div>
        <div className="field"><label htmlFor="params">Parameters (millions)</label><input id="params" type="number" min="1" value={params} onChange={(e) => setParams(Math.max(1, +e.target.value || 1))} /></div>
        <div className="field"><label htmlFor="layers">Layers</label><input id="layers" type="number" min="1" value={layers} onChange={(e) => setLayers(Math.max(1, +e.target.value || 1))} /></div>
        <div className="field"><label htmlFor="hidden">Hidden width</label><input id="hidden" type="number" min="1" value={hidden} onChange={(e) => setHidden(Math.max(1, +e.target.value || 1))} /></div>
        <Range label="Adapter rank" value={rank} min={1} max={128} step={1} onChange={setRank} show={rank} />
        <div className="field"><label htmlFor="bits">Base bit width</label>
          <select id="bits" value={bits} onChange={(e) => setBits(+e.target.value)}>{[4, 8, 16].map((b) => <option key={b} value={b}>{b}-bit</option>)}</select>
        </div>
      </div>
      <div className="grid2">
        <div className="bars">
          {bars.map(([k, v, col]) => (
            <div className="bar-row" key={k}><div className="lab"><span>{k}</span><b>{fmt(v)} B</b></div><div className="track"><div className="fill" style={{ width: pct(v / top), background: col }} /></div></div>
          ))}
        </div>
        <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0 }}>
          <div className="tile"><div className="num">{f4(row.qlora_total_fraction_of_full)}</div><div className="cap">QLoRA total as a fraction of full fp16</div></div>
          <div className="tile"><div className="num">{fmt(row.lora_parameters)}</div><div className="cap">Adapter parameters ({fmt(row.lora_adapter_bytes)} B)</div></div>
        </div>
      </div>
      <h3 style={{ marginTop: 20 }}>The {mem.rows.length} recorded rows, recomputed in your browser</h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Model</th><th>Full fp16 B</th><th>Adapter B</th><th>LoRA total B</th><th>4-bit base B</th><th>QLoRA total B</th><th>Fraction</th><th>Matches file</th></tr></thead>
          <tbody>{recomputed.map(({ r, got }) => <tr key={r.name}><td>{r.name}</td>{keys.map((k) => <td key={k}>{k.includes('fraction') ? f4(got[k]) : fmt(got[k])}</td>)}<td className={keys.every((k) => got[k] === r[k]) ? 'ok' : 'no'}>{keys.every((k) => got[k] === r[k]) ? '✓ exact' : '✗'}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="hint" style={{ marginTop: 10 }}>Excluded, as in the repo: {mem.assumptions.excluded}. The full column is a weight-storage floor, not a training total. Model sizes are illustrative, not any released architecture.</p>
    </section>
  );
}

// Deep limits: the bullets under '## Limits named at review' in level-3-deep/README.md.
function parseLimits(md) {
  const sec = md.split('## Limits named at review')[1].split('\n## ')[0];
  return sec.split('\n- ').slice(1).map((b) => b.replace(/\s+/g, ' ').trim());
}
// Verdict: the paragraph after '**Verdict.**' in README.md.
function parseVerdict(md) {
  return md.split('**Verdict.**')[1].split('\n\n')[0].replace(/\s+/g, ' ').trim();
}
function parseRoadmap(md) {
  const sec = md.split('### v1.1 backlog')[1].split('\n### ')[0];
  return sec.split(/\n(?=\d+\. )/).slice(1).map((b) => { const t = b.replace(/\s+/g, ' ').trim(); const [, n, rest] = t.match(/^(\d+)\. (.*)$/); const i = rest.indexOf(': '); return { n: +n, title: i > 0 ? rest.slice(0, i) : rest, body: i > 0 ? rest.slice(i + 2) : '' }; });
}
const RIGOR_STATUS = { 3: 'Partly done: misaligned check in rigor/', 4: 'Done at toy scale: β sweep in rigor/', 5: 'Partly done: 36-setting sweep in rigor/', 8: 'Partly done: best-val selection in rigor/' };

function Limits() {
  const limits = useMemo(() => parseLimits(deepReadme), []);
  const items = useMemo(() => parseRoadmap(report), []);
  const verdict = useMemo(() => parseVerdict(readme), []);
  return (
    <section id="limits" className="card window reveal" data-title="REPORT.md — honest limits">
      <p className="eyebrow">Honest limits and roadmap</p>
      <h2>What this lab does <em>not</em> show yet.</h2>
      <p className="sub">There is no real language model anywhere in this repo: no text, no tokenizer, no open-model run, and no measured GPU memory. The full report, with every hypothesis and miss, is in <a href={blob('REPORT.md')}>REPORT.md</a>.</p>
      <p className="hint" style={{ marginTop: 8 }}><b>Verdict.</b> {verdict}</p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>Deep limits</h3><span className="pill warn">{limits.length} named</span></div>
          <ul className="steps">
            {limits.map((t) => <li key={t}><div>{t}</div></li>)}
          </ul>
          <p className="hint" style={{ marginTop: 10 }}>Parsed from level-3-deep/README.md at build time.</p>
        </div>
        <div>
          <div className="head" style={{ marginBottom: 8 }}><h3>v1.1 backlog</h3><span className="pill info">from REPORT.md</span></div>
          <ol className="steps">
            {items.map((it) => (
              <li key={it.n}><span className="n">{it.n}</span><div><b>{it.title}</b>{RIGOR_STATUS[it.n] && <> <span className="pill good">{RIGOR_STATUS[it.n]}</span></>}{it.body && <div className="hint">{it.body[0].toUpperCase() + it.body.slice(1)}</div>}</div></li>
            ))}
          </ol>
        </div>
      </div>
      <div className="tile" style={{ marginTop: 16 }}>
        <h3>Single seed, now with spread</h3>
        <p className="hint" style={{ margin: '6px 0 0' }}>
          Each recorded number is one seed. Across {rigor.seeds.n_seeds} seeds: SFT test accuracy {f4(rigor.seeds.sft.summary.final_test_accuracy.mean)} (CI {ci(rigor.seeds.sft.summary.final_test_accuracy)}),
          GRPO reward {f4(rigor.seeds.grpo.summary.final_mean_reward.mean)} (CI {ci(rigor.seeds.grpo.summary.final_mean_reward)}), DPO held-out margin {f4(rigor.seeds.dpo.summary.held_out_margin.mean)} (CI {ci(rigor.seeds.dpo.summary.held_out_margin)}).
          The rigor study also found that the Build GRPO gradient does not match its own stated loss; the recorded results are kept as run and a corrected ablation sits beside them.
        </p>
      </div>
    </section>
  );
}

function Reproduce() {
  const docs = [
    ['level-1-learn/README.md', 'Learn', 'LoRA from scratch, GRPO advantages'],
    ['level-2-build/README.md', 'Build', 'SFT pipeline and GRPO track'],
    ['level-3-deep/README.md', 'Deep', 'DPO, retention, probe, memory'],
    ['rigor/README.md', 'Rigor', 'Seeds, baselines, sweeps, ablations'],
    ['REPORT.md', 'Report', 'Hypotheses, misses, open work'],
    ['PROPOSAL.md', 'Proposal', 'Scope and the draft confirmatory re-run'],
  ];
  return (
    <section id="reproduce" className="card window reveal" data-title="Terminal — reproduce">
      <p className="eyebrow">Reproduce it</p>
      <h2>Four scripts, fixed seeds, the same JSON every time.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        <li><span className="n">1</span><div><b>Clone and install</b><div className="hint"><code>git clone {REPO}.git && cd finetune-lab && pip install -r requirements.txt</code></div></div></li>
        <li><span className="n">2</span><div><b>Rerun every level</b><div className="hint"><code>python3 level-1-learn/run_learn.py && python3 level-2-build/run_build.py && python3 level-3-deep/run_deep.py && python3 rigor/run_rigor.py</code></div></div></li>
        <li><span className="n">3</span><div><b>Run the tests</b><div className="hint"><code>python3 -m unittest discover -s tests</code></div></div></li>
        <li><span className="n">4</span><div><b>Rebuild this page</b><div className="hint"><code>cd site && npm ci && npm run build</code> runs the known-answer check first.</div></div></li>
      </ol>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}
