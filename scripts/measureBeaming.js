/**
 * Diagnostic: what redshift factors does the disk actually produce?
 *
 * The shader weights emission by g^4. If g reaches large values the approaching
 * side saturates the whole image, so the display defaults have to be chosen
 * from the real distribution rather than guessed.
 *
 * Run: node scripts/measureBeaming.js
 */

import { momentumFromTetradDirection, integrateHamilton } from "../src/core/geodesics.js";
import { diskInnerRadius } from "../src/core/kerr.js";
import { redshiftFactor } from "../src/core/redshift.js";

/** Trace one pixel ray and collect g at every disk crossing. */
function traceRay(a, camR, camTheta, ndcX, ndcY, fov, diskInner, diskOuter) {
  const t = Math.tan(fov / 2);
  const dir = {
    n1: -1,
    n2: -ndcY * t,
    n3: ndcX * t,
  };
  const norm = Math.hypot(dir.n1, dir.n2, dir.n3);
  dir.n1 /= norm; dir.n2 /= norm; dir.n3 /= norm;

  const p = momentumFromTetradDirection(camR, camTheta, a, dir, 1);
  const state = [0, camR, camTheta, 0, p.pt, p.pr, p.pth, p.pphi];
  const res = integrateHamilton({
    state, a, step: 0.004, maxSteps: 400000, escapeRadius: 200,
  });

  const gs = [];
  for (const c of res.crossings) {
    if (c.r < diskInner || c.r > diskOuter) continue;
    // xi = Lz/E; E is conserved and normalised to 1 at launch.
    const g = redshiftFactor(c.r, a, res.state[7], 1);
    if (g !== null) gs.push(g);
  }
  return gs;
}

const a = 0.9;
const camR = 14;
const camTheta = 1.32;
const fov = (50 * Math.PI) / 180;
const diskInner = diskInnerRadius(a, true);
const diskOuter = 22;

const all = [];
const N = 40;
for (let iy = 0; iy < N; iy++) {
  for (let ix = 0; ix < N; ix++) {
    const ndcX = (2 * (ix + 0.5)) / N - 1;
    const ndcY = 1 - (2 * (iy + 0.5)) / N;
    all.push(...traceRay(a, camR, camTheta, ndcX, ndcY, fov, diskInner, diskOuter));
  }
}

all.sort((x, y) => x - y);
const q = (f) => all[Math.min(all.length - 1, Math.floor(f * (all.length - 1)))];

console.log(`crossings sampled: ${all.length}`);
if (all.length === 0) {
  console.log("no disk crossings — check camera geometry");
  process.exit(1);
}
console.log(`g: min=${all[0].toFixed(4)}  p50=${q(0.5).toFixed(4)}  p90=${q(0.9).toFixed(4)}  p99=${q(0.99).toFixed(4)}  max=${all[all.length - 1].toFixed(4)}`);
console.log(`g^4: max=${Math.pow(all[all.length - 1], 4).toExponential(3)}  p99=${Math.pow(q(0.99), 4).toExponential(3)}`);

// How much of the image is above a given g^4?
for (const threshold of [1, 5, 20, 100, 400]) {
  const n = all.filter((g) => Math.pow(g, 4) > threshold).length;
  console.log(`  g^4 > ${String(threshold).padStart(4)}: ${(100 * n / all.length).toFixed(2)}% of crossings`);
}
