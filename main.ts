// GTA VI trophy rarity forecaster — gta6-trophy-rarity-forecast-01.
//
// Forecasts, for each of the 55 trophies, the percentage of tracked players
// who will have earned it by the case's target time, as five quantiles
// (0.05 / 0.25 / 0.5 / 0.75 / 0.95).
//
// Model: short-window logit-linear ensemble. Each trophy's rarity is tracked
// in logit space against u = ln(1 + hours since launch). The median is the
// average of two ordinary-least-squares lines fitted over the last 6 and 12
// hours of reads; the spread comes from how much four candidate medians
// (6h, 12h, 24h linear fits plus a 48h quadratic) disagree. A story-chain
// constraint keeps chapter trophies ordered, and a frozen launch-night
// prefix (identical reads) is dropped when enough live reads remain.

import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

// Hours are measured from the start of launch day, 2026-11-19 00:00 UTC.
const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");
const MS_PER_HOUR = 3600000;

// Normal-quantile z-scores for the five requested levels (exact values:
// the 0.05/0.95 and 0.25/0.75 quantiles of the standard normal).
const Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

// The story chain: you cannot earn a later chapter without the earlier one,
// and the platinum requires everything. Medians must not increase along it.
const CHAIN_IDS = [3, 4, 5, 6, 7, 14, 1];

// Calibrated forecast-error scale (in logits) for a 4.46-day horizon:
// the base spread every trophy gets before model disagreement widens it.
const BASE_SPREAD = 0.45;

const RATE_MIN = 0.01;
const RATE_MAX = 99.99;

interface TrophyDef {
  id: number;
}

interface HistoryRead {
  slot: string;
  players_tracked: number;
  rates: number[];
}

interface CaseInput {
  at: string;
  target: string;
  trophies: TrophyDef[];
  history: HistoryRead[];
}

interface Point {
  u: number;
  y: number;
}

interface LineFit {
  intercept: number;
  slope: number;
}

interface QuadFit {
  a: number;
  b: number;
  c: number;
}

function clampRate(rate: number): number {
  if (rate < RATE_MIN) return RATE_MIN;
  if (rate > RATE_MAX) return RATE_MAX;
  return rate;
}

function clampSpread(s: number): number {
  if (s < 0.3) return 0.3;
  if (s > 2.5) return 2.5;
  return s;
}

function logit(rate: number): number {
  const r = clampRate(rate);
  return Math.log(r / (100 - r));
}

function logistic(x: number): number {
  return 100 / (1 + Math.exp(-x));
}

function hoursSinceEpoch(ms: number): number {
  return (ms - EPOCH_MS) / MS_PER_HOUR;
}

// Transformed time coordinate: log-hours since launch.
function uOf(slotMs: number): number {
  return Math.log(1 + hoursSinceEpoch(slotMs));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

function sameRates(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

// Ordinary least squares line through the points; null when the fit is
// degenerate (fewer than 3 points, or no spread in u).
function linearFit(points: Point[]): LineFit | null {
  if (points.length < 3) return null;
  let sumU = 0;
  let sumY = 0;
  for (const p of points) {
    sumU += p.u;
    sumY += p.y;
  }
  const meanU = sumU / points.length;
  const meanY = sumY / points.length;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    const du = p.u - meanU;
    sxx += du * du;
    sxy += du * (p.y - meanY);
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  return { intercept: meanY - slope * meanU, slope };
}

// Quadratic fit y = a + b*u + c*u^2 via the normal equations; null when the
// system is singular.
function quadraticFit(points: Point[]): QuadFit | null {
  if (points.length < 3) return null;
  let n = 0;
  let su = 0;
  let su2 = 0;
  let su3 = 0;
  let su4 = 0;
  let sy = 0;
  let suy = 0;
  let su2y = 0;
  for (const p of points) {
    const u = p.u;
    const u2 = u * u;
    n += 1;
    su += u;
    su2 += u2;
    su3 += u2 * u;
    su4 += u2 * u2;
    sy += p.y;
    suy += u * p.y;
    su2y += u2 * p.y;
  }
  // Solve the 3x3 system by Gaussian elimination with partial pivoting.
  const rows: number[][] = [
    [n, su, su2, sy],
    [su, su2, su3, suy],
    [su2, su3, su4, su2y],
  ];
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let r = col + 1; r < 3; r++) {
      if (Math.abs(rows[r][col]) > Math.abs(rows[pivot][col])) pivot = r;
    }
    if (Math.abs(rows[pivot][col]) < 1e-12) return null;
    if (pivot !== col) {
      const tmp = rows[col];
      rows[col] = rows[pivot];
      rows[pivot] = tmp;
    }
    for (let r = col + 1; r < 3; r++) {
      const factor = rows[r][col] / rows[col][col];
      for (let k = col; k < 4; k++) {
        rows[r][k] -= factor * rows[col][k];
      }
    }
  }
  const c = rows[2][3] / rows[2][2];
  const b = (rows[1][3] - rows[1][2] * c) / rows[1][1];
  const a = (rows[0][3] - rows[0][1] * b - rows[0][2] * c) / rows[0][0];
  return { a, b, c };
}

// Candidate median from one window: linear fit evaluated at uTarget, or the
// last observed logit when the window is too thin to fit.
function linearCandidate(points: Point[], uTarget: number, lastLogit: number): number {
  const fit = linearFit(points);
  if (fit === null) return lastLogit;
  return fit.intercept + fit.slope * uTarget;
}

// Candidate median from the 48h window: a quadratic, but only when its
// curvature is non-positive (rates saturate, they do not accelerate); a
// positively curved quadratic falls back to the linear fit.
function quadCandidate(points: Point[], uTarget: number, lastLogit: number): number {
  const fit = quadraticFit(points);
  if (fit !== null && fit.c <= 0) {
    return fit.a + fit.b * uTarget + fit.c * uTarget * uTarget;
  }
  return linearCandidate(points, uTarget, lastLogit);
}

// Pool-adjacent-violators for a decreasing sequence: scan the values,
// keeping blocks of consecutive entries that share one average; whenever a
// block's average rises above the previous block's, merge the two blocks
// and re-average the whole merged block. The result is non-increasing.
function enforceDecreasing(values: number[]): number[] {
  interface Block {
    start: number;
    end: number;
    sum: number;
    count: number;
  }
  const blocks: Block[] = [];
  for (let i = 0; i < values.length; i++) {
    blocks.push({ start: i, end: i, sum: values[i], count: 1 });
    while (blocks.length > 1) {
      const cur = blocks[blocks.length - 1];
      const prev = blocks[blocks.length - 2];
      if (prev.sum / prev.count < cur.sum / cur.count) {
        blocks.pop();
        blocks.pop();
        blocks.push({
          start: prev.start,
          end: cur.end,
          sum: prev.sum + cur.sum,
          count: prev.count + cur.count,
        });
      } else {
        break;
      }
    }
  }
  const out = values.slice();
  for (const b of blocks) {
    const avg = b.sum / b.count;
    for (let i = b.start; i <= b.end; i++) out[i] = avg;
  }
  return out;
}

function forecastCase(input: CaseInput): { trophies: { id: number; q: number[] }[] } {
  const atMs = Date.parse(input.at);
  const targetMs = Date.parse(input.target);
  const uTarget = Math.log(1 + hoursSinceEpoch(targetMs));

  // Drop the frozen launch-night prefix: the leading run of reads identical
  // to the first one, but only when at least 10 live reads remain afterwards.
  let reads = input.history;
  if (reads.length > 0) {
    const first = reads[0];
    let drop = 0;
    while (drop < reads.length) {
      const r = reads[drop];
      if (r.players_tracked !== first.players_tracked) break;
      if (!sameRates(r.rates, first.rates)) break;
      drop += 1;
    }
    if (reads.length - drop >= 10) {
      reads = reads.slice(drop);
    }
  }

  const trophyCount = input.trophies.length;
  const medianLogits: number[] = [];
  const spreads: number[] = [];
  const medianRates: number[] = [];

  for (let t = 0; t < trophyCount; t++) {
    const series: Point[] = [];
    let maxRate = 0;
    for (const read of reads) {
      const rate = read.rates[t];
      if (rate > maxRate) maxRate = rate;
      series.push({ u: uOf(Date.parse(read.slot)), y: logit(rate) });
    }
    const lastLogit = series.length > 0 ? series[series.length - 1].y : logit(RATE_MIN);

    const window = (hours: number): Point[] => {
      const cutoff = atMs - hours * MS_PER_HOUR;
      const out: Point[] = [];
      for (let i = 0; i < reads.length; i++) {
        if (Date.parse(reads[i].slot) >= cutoff) out.push(series[i]);
      }
      return out;
    };

    const m6 = linearCandidate(window(6), uTarget, lastLogit);
    const m12 = linearCandidate(window(12), uTarget, lastLogit);
    const m24 = linearCandidate(window(24), uTarget, lastLogit);
    const m48 = quadCandidate(window(48), uTarget, lastLogit);

    const medianLogit = (m6 + m12) / 2;
    const range = Math.max(m6, m12, m24, m48) - Math.min(m6, m12, m24, m48);
    let s = clampSpread(Math.sqrt(BASE_SPREAD * BASE_SPREAD + Math.pow(0.5 * range, 2)));
    // No one has earned this trophy yet: no signal at all, so widen the
    // spread enough for the upper quantiles to cover a late breakthrough.
    if (maxRate <= 0.011) s = 2.0;

    // The median rate is clamped to the observable range first, and the
    // quantiles are built from the logit of that clamped median — a raw
    // extrapolated logit past the cap must not leak into the quantiles.
    const medianRate = clampRate(logistic(medianLogit));
    medianLogits.push(logit(medianRate));
    spreads.push(s);
    medianRates.push(medianRate);
  }

  // Enforce the story chain on the medians.
  const chainPositions: number[] = [];
  const chainValues: number[] = [];
  for (const id of CHAIN_IDS) {
    const pos = input.trophies.findIndex((tr) => tr.id === id);
    if (pos >= 0) {
      chainPositions.push(pos);
      chainValues.push(medianRates[pos]);
    }
  }
  if (chainValues.length > 1) {
    const fixed = enforceDecreasing(chainValues);
    for (let k = 0; k < chainPositions.length; k++) {
      const pos = chainPositions[k];
      medianRates[pos] = fixed[k];
      medianLogits[pos] = logit(fixed[k]);
    }
  }

  const rows: { id: number; q: number[] }[] = [];
  for (let t = 0; t < trophyCount; t++) {
    const qs: number[] = [];
    for (let k = 0; k < Z_VALUES.length; k++) {
      qs.push(clampRate(round4(logistic(medianLogits[t] + Z_VALUES[k] * spreads[t]))));
    }
    // Enforce non-decreasing quantiles after rounding.
    for (let k = 1; k < qs.length; k++) {
      if (qs[k] < qs[k - 1]) qs[k] = qs[k - 1];
    }
    rows.push({ id: input.trophies[t].id, q: qs });
  }
  return { trophies: rows };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  if (!existsSync(outputDir)) mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir).filter((f) => f.endsWith(".in")).sort();
  for (const file of files) {
    const input = JSON.parse(readFileSync(join(inputDir, file), "utf8"));
    const result = forecastCase(input);
    const outName = file.slice(0, -3) + ".out";
    writeFileSync(join(outputDir, outName), JSON.stringify(result));
  }
}

main();
