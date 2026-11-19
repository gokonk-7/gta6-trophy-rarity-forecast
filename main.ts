// gta6-trophy-rarity-forecast-01 — forecast each GTA VI trophy's Exophase earn
// rate at the case target, as five quantiles per trophy.
//
// Model (short-window logit-linear ensemble, see MODEL_SPEC.md):
//   * Work in u = ln(1 + h), h = hours since 2026-11-19T00:00:00Z, and fit
//     logit(rate) as a linear function of u over SHORT recent windows.
//     logit(rate) is concave in u, so a short window tracks the current
//     (flatter) slope instead of averaging in the early steep dilution phase.
//   * Median logit = mean of the linear fits on the last 6h and last 12h of
//     reads, evaluated at the target. A window with < 3 points falls back to
//     the last observed logit rate.
//   * Spread: two more candidate medians (linear on the last 24h, and a
//     quadratic with curvature constrained <= 0 on the last 48h) measure
//     disagreement; s = sqrt(0.45^2 + (0.5 * range)^2), clipped to
//     [0.30, 2.5]. Trophies nobody has earned yet (max observed rate
//     <= 0.011%) get s = 2.0 so the upper quantiles cover a late breakthrough.
//   * Story-chain nesting: Ch1 >= Ch2 >= Ch3 >= Ch4 >= final mission >=
//     100% Completion >= Platinum on the medians (isotonic regression).
//   * Frozen-prefix handling: drop the leading run of reads identical to the
//     first read when >= 10 reads remain; otherwise keep everything (a fully
//     frozen history then forecasts as a persistence median).
//   * Quantiles: logistic(logit(median) + z_tau * s), z the standard-normal
//     quantiles, rounded to 4 decimals, clamped to [0.01, 99.99].
//
// Deterministic: no randomness, no clock, no network. Node 22 stdlib only.

import * as fs from "node:fs";
import * as path from "node:path";

const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");
const MS_PER_HOUR = 3600000;

// Standard-normal quantiles at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
// Story chain, trophy ids: Ch1, Ch2, Ch3, Ch4, final mission, 100%, platinum.
const CHAIN_IDS = [3, 4, 5, 6, 7, 14, 1];
const BASE_SPREAD = 0.45;
const MIN_SPREAD = 0.3;
const MAX_SPREAD = 2.5;
const ZERO_RATE_SPREAD = 2.0;
const ZERO_RATE_MAX = 0.011;

interface TrophyDef {
  id: number;
}

interface HistoryRead {
  slot: string;
  players_tracked: number;
  rates: (number | null)[];
}

interface CaseFile {
  at: string;
  target: string;
  trophies: TrophyDef[];
  history: HistoryRead[];
}

interface Read {
  ms: number;
  playersTracked: number;
  rates: (number | null)[];
}

interface FitPoint {
  u: number;
  y: number;
}

function clampRate(rate: number): number {
  if (rate < 0.01) return 0.01;
  if (rate > 99.99) return 99.99;
  return rate;
}

function logitOf(ratePct: number): number {
  const p = clampRate(ratePct);
  return Math.log(p / (100 - p));
}

function logistic(logit: number): number {
  return 100 / (1 + Math.exp(-logit));
}

// Transformed time: u = ln(1 + h), h hours since the release-day epoch.
function uOf(ms: number): number {
  return Math.log(1 + (ms - EPOCH_MS) / MS_PER_HOUR);
}

// Ordinary least squares of y on u; value of the fitted line at uTarget.
// Returns null when there are fewer than 3 points.
function fitLine(points: FitPoint[], uTarget: number): number | null {
  const n = points.length;
  if (n < 3) return null;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const pt of points) {
    sx += pt.u;
    sy += pt.y;
    sxx += pt.u * pt.u;
    sxy += pt.u * pt.y;
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return sy / n;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return intercept + slope * uTarget;
}

// Solve a 3x3 linear system by Gaussian elimination with partial pivoting.
// Returns null when the system is (near-)singular.
function solve3(a: number[][], b: number[]): number[] | null {
  const m: number[][] = [
    [a[0][0], a[0][1], a[0][2], b[0]],
    [a[1][0], a[1][1], a[1][2], b[1]],
    [a[2][0], a[2][1], a[2][2], b[2]],
  ];
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = m[col];
      m[col] = m[pivot];
      m[pivot] = tmp;
    }
    for (let row = col + 1; row < 3; row++) {
      const factor = m[row][col] / m[col][col];
      for (let k = col; k < 4; k++) m[row][k] -= factor * m[col][k];
    }
  }
  const x: number[] = [0, 0, 0];
  for (let row = 2; row >= 0; row--) {
    let sum = m[row][3];
    for (let k = row + 1; k < 3; k++) sum -= m[row][k] * x[k];
    x[row] = sum / m[row][row];
  }
  return x;
}

// Quadratic least-squares fit of y on u (basis 1, u, u^2), value at uTarget.
// Curvature is constrained to <= 0: a positive quadratic coefficient falls
// back to the linear fit. Returns null when there are fewer than 3 points.
function fitQuad(points: FitPoint[], uTarget: number): number | null {
  const n = points.length;
  if (n < 3) return null;
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let s4 = 0;
  let t0 = 0;
  let t1 = 0;
  let t2 = 0;
  for (const pt of points) {
    const u = pt.u;
    const u2 = u * u;
    s0 += 1;
    s1 += u;
    s2 += u2;
    s3 += u2 * u;
    s4 += u2 * u2;
    t0 += pt.y;
    t1 += u * pt.y;
    t2 += u2 * pt.y;
  }
  const coef = solve3(
    [
      [s0, s1, s2],
      [s1, s2, s3],
      [s2, s3, s4],
    ],
    [t0, t1, t2],
  );
  if (coef === null) return fitLine(points, uTarget);
  if (coef[2] > 0) return fitLine(points, uTarget);
  return coef[0] + coef[1] * uTarget + coef[2] * uTarget * uTarget;
}

// Pool-adjacent-violators for a non-increasing sequence (equal weights).
// Returns the adjusted values, same length and order as the input.
function isotonicDecreasing(values: number[]): number[] {
  interface Pool {
    sum: number;
    count: number;
  }
  const pools: Pool[] = [];
  for (const v of values) {
    pools.push({ sum: v, count: 1 });
    while (pools.length >= 2) {
      const n = pools.length;
      const a = pools[n - 2];
      const b = pools[n - 1];
      if (a.sum / a.count >= b.sum / b.count) break;
      pools.pop();
      pools.pop();
      pools.push({ sum: a.sum + b.sum, count: a.count + b.count });
    }
  }
  const out: number[] = [];
  for (const p of pools) {
    const avg = p.sum / p.count;
    for (let k = 0; k < p.count; k++) out.push(avg);
  }
  return out;
}

function sameRead(a: Read, b: Read): boolean {
  if (a.playersTracked !== b.playersTracked) return false;
  if (a.rates.length !== b.rates.length) return false;
  for (let i = 0; i < a.rates.length; i++) {
    if (a.rates[i] !== b.rates[i]) return false;
  }
  return true;
}

function forecastCase(kase: CaseFile): { trophies: { id: number; q: number[] }[] } {
  const atMs = Date.parse(kase.at);
  const uTarget = uOf(Date.parse(kase.target));

  const allReads: Read[] = [];
  for (const h of kase.history) {
    // A read's time is its clock slot: the house reads on a fixed clock and
    // the slot is the read's canonical time (its `at` is the same instant
    // plus sub-second jitter).
    allReads.push({
      ms: Date.parse(h.slot),
      playersTracked: h.players_tracked,
      rates: h.rates,
    });
  }

  // Frozen-prefix handling: drop the leading run identical to the first read
  // when at least 10 reads remain afterwards; otherwise keep everything.
  let reads = allReads;
  if (allReads.length > 0) {
    let run = 0;
    while (run < allReads.length && sameRead(allReads[run], allReads[0])) run++;
    if (allReads.length - run >= 10) reads = allReads.slice(run);
  }

  const trophyCount = kase.trophies.length;
  const medians: number[] = [];
  const spreads: number[] = [];

  for (let i = 0; i < trophyCount; i++) {
    // Observed (time, rate) pairs for this trophy, in chronological order.
    const obs: { ms: number; rate: number }[] = [];
    for (const r of reads) {
      const rate = r.rates[i];
      if (rate !== null && rate !== undefined) obs.push({ ms: r.ms, rate });
    }

    let lastLogit = 0;
    let maxRate = 0;
    if (obs.length > 0) {
      lastLogit = logitOf(obs[obs.length - 1].rate);
      for (const o of obs) {
        if (o.rate > maxRate) maxRate = o.rate;
      }
    }

    const windows: FitPoint[][] = [[], [], [], []];
    const windowHours = [6, 12, 24, 48];
    for (const o of obs) {
      const ageHours = (atMs - o.ms) / MS_PER_HOUR;
      const pt: FitPoint = { u: uOf(o.ms), y: logitOf(o.rate) };
      for (let w = 0; w < 4; w++) {
        if (ageHours <= windowHours[w]) windows[w].push(pt);
      }
    }

    const m6 = fitLine(windows[0], uTarget);
    const m12 = fitLine(windows[1], uTarget);
    const m24 = fitLine(windows[2], uTarget);
    const m48 = fitQuad(windows[3], uTarget);
    const c6 = m6 === null ? lastLogit : m6;
    const c12 = m12 === null ? lastLogit : m12;
    const c24 = m24 === null ? lastLogit : m24;
    const c48 = m48 === null ? lastLogit : m48;

    const medianLogit = (c6 + c12) / 2;
    medians.push(clampRate(logistic(medianLogit)));

    const range = Math.max(c6, c12, c24, c48) - Math.min(c6, c12, c24, c48);
    let s = Math.sqrt(BASE_SPREAD * BASE_SPREAD + (0.5 * range) * (0.5 * range));
    if (s < MIN_SPREAD) s = MIN_SPREAD;
    if (s > MAX_SPREAD) s = MAX_SPREAD;
    if (maxRate <= ZERO_RATE_MAX) s = ZERO_RATE_SPREAD;
    spreads.push(s);
  }

  // Story-chain nesting on the medians (ids 3,4,5,6,7,14,1).
  const indexById = new Map<number, number>();
  for (let i = 0; i < trophyCount; i++) indexById.set(kase.trophies[i].id, i);
  const chainIdx: number[] = [];
  for (const id of CHAIN_IDS) {
    const idx = indexById.get(id);
    if (idx !== undefined) chainIdx.push(idx);
  }
  if (chainIdx.length > 1) {
    const chainValues: number[] = [];
    for (const idx of chainIdx) chainValues.push(medians[idx]);
    const adjusted = isotonicDecreasing(chainValues);
    for (let k = 0; k < chainIdx.length; k++) medians[chainIdx[k]] = adjusted[k];
  }

  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < trophyCount; i++) {
    const medLogit = logitOf(medians[i]);
    const s = spreads[i];
    const q: number[] = [];
    for (const z of Z) {
      const v = logistic(medLogit + z * s);
      let rounded = Math.round(v * 10000) / 10000;
      if (rounded < 0.01) rounded = 0.01;
      if (rounded > 99.99) rounded = 99.99;
      q.push(rounded);
    }
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    rows.push({ id: kase.trophies[i].id, q });
  }
  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const names = fs.readdirSync(inputDir).filter((n) => n.endsWith(".in")).sort();
  for (const name of names) {
    const raw = fs.readFileSync(path.join(inputDir, name), "utf8");
    const kase: CaseFile = JSON.parse(raw);
    const out = forecastCase(kase);
    const outName = name.slice(0, -3) + ".out";
    fs.writeFileSync(path.join(outputDir, outName), JSON.stringify(out));
  }
}

main();
