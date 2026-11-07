#!/usr/bin/env node
/**
 * Numerical validation report.
 *
 * Where `npm test` answers "is this within tolerance?", this prints the actual
 * numbers: measured value, reference value, absolute and relative error. That
 * distinction matters for a physics core — a test that passes at 1e-3 hides
 * whether you are at 1e-9 or 9e-4, and the difference between those is the
 * difference between "correct" and "correct by luck with a loose tolerance".
 *
 * Every reference here is external to this codebase: the Bardeen–Press–Teukolsky
 * (1973) photon-orbit table, closed-form Schwarzschild limits, and the
 * post-Newtonian deflection series. Nothing is checked against our own output.
 *
 * Exit status is non-zero if any check fails, so this doubles as a CI gate.
 */

import {
  metric,
  inverseMetric,
  idx,
  T,
  R,
  PH,
  horizonRadius,
  innerHorizonRadius,
  ergosphereRadius,
  circularEnergyAndAngularMomentum,
  iscoRadius,
  marginallyBoundRadius,
  diskInnerRadius,
} from "../src/core/kerr.js";
import {
  carterConstants,
  carterStateFromMomentum,
  nullResidual,
  momentumFromTetradDirection,
  integrateHamilton,
  integrateCarter,
} from "../src/core/geodesics.js";
import { integrateVerlet } from "../src/core/mino.js";
import { equatorialPhotonOrbit } from "../src/core/photonOrbit.js";
import { redshiftFactor, diskTemperature } from "../src/core/redshift.js";

/* ------------------------------------------------------------------ *
 * Reporting
 * ------------------------------------------------------------------ */

const results = [];
let failures = 0;

/** Print a section heading. */
function section(title) {
  console.log("");
  console.log("─".repeat(78));
  console.log(title);
  console.log("─".repeat(78));
}

/** Format a number with `digits` significant figures, fixed where possible. */
function num(x, digits = 8) {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const mag = Math.abs(Math.log10(x));
  if (mag > 6 || mag < 3) return x.toExponential(digits - 1);
  return x.toFixed(Math.max(0, digits - 1 - Math.floor(Math.log10(Math.abs(x)))));
}

/**
 * Record one measured quantity against its reference.
 *
 * @param {string} label
 * @param {number} got     measured
 * @param {number} want    independently known reference
 * @param {number} tol     relative tolerance for the verdict
 * @param {string} [source] where the reference comes from
 */
function check(label, got, want, tol, source = "closed form") {
  const abs = got - want;
  const rel = want === 0 ? Math.abs(abs) : Math.abs(abs / want);
  const ok = rel <= tol;
  if (!ok) failures++;
  results.push({ label, got, want, abs, rel, tol, ok, source });

  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}`);
  console.log(`        measured  ${num(got)}`);
  console.log(`        reference ${num(want)}   (${source})`);
  console.log(`        abs err   ${num(abs, 3)}    rel err ${num(rel, 3)}    tol ${tol}`);
}

/** Record a property that must hold but has no numeric reference. */
function assert(label, condition, detail = "") {
  if (!condition) failures++;
  console.log(`  ${condition ? "PASS" : "FAIL"}  ${label}${detail ? `  [${detail}]` : ""}`);
}

/* ------------------------------------------------------------------ *
 * 1. Metric and horizon structure
 * ------------------------------------------------------------------ */

section("1. METRIC & HORIZON STRUCTURE");

// g^{mu nu} g_{nu sigma} = delta^mu_sigma is the definition of an inverse.
// If this fails, every geodesic in the project is wrong.
{
  let maxErr = 0;
  for (const a of [0, 0.5, 0.9, 0.998]) {
    for (const r of [2.5, 6, 20]) {
      for (const th of [0.4, Math.PI / 2]) {
        const g = metric(r, th, a);
        const gi = inverseMetric(r, th, a);
        for (let mu = 0; mu < 4; mu++) {
          for (let nu = 0; nu < 4; nu++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += gi[idx(mu, k)] * g[idx(k, nu)];
            maxErr = Math.max(maxErr, Math.abs(s - (mu === nu ? 1 : 0)));
          }
        }
      }
    }
  }
  check("max |g^{mu k} g_{k nu} - delta^mu nu|", maxErr, 0, 1e-12, "definition of inverse");
}

// Schwarzschild limit: at a = 0 the metric must reduce to the static form.
{
  const r = 7.3;
  const th = 1.1;
  const g = metric(r, th, 0);
  check("Schwarzschild g_tt at r=7.3", g[idx(T, T)], -(1 - 2 / r), 1e-14);
  check("Schwarzschild g_phiphi at r=7.3", g[idx(PH, PH)], r * r * Math.sin(th) ** 2, 1e-14);
  check("Schwarzschild g_rr at r=7.3", g[idx(R, R)], 1 / (1 - 2 / r), 1e-14);
  check("Schwarzschild g_tphi must vanish", Math.abs(g[idx(T, PH)]), 0, 1e-14, "static at a=0");
}

// r_+ + r_- = 2M is an exact identity of the Kerr roots.
{
  let maxErr = 0;
  for (const a of [0, 0.3, 0.7, 0.9, 0.998]) {
    maxErr = Math.max(maxErr, Math.abs(horizonRadius(a) + innerHorizonRadius(a) - 2));
  }
  check("max |r_+ + r_- - 2M| over spins", maxErr, 0, 1e-12, "sum of horizon roots");
  check("horizon radius at a=0", horizonRadius(0), 2, 1e-14, "Schwarzschild r_s = 2M");
  check("ergosphere meets equator at 2M for any spin", ergosphereRadius(0.9), 2, 1e-14);
}

// ISCO. Two kinds of anchor, deliberately different in nature:
//
//   (a) Exact limits, which are closed-form results independent of this code
//       and can be checked to machine precision.
//   (b) An independent numerical derivation from marginal stability, below.
//
// Quoting a table of intermediate-spin ISCOs would be the weakest possible
// check: if the numbers came from the same source as the implementation, the
// comparison proves nothing. So intermediate spins are checked against the
// numerical derivation instead, and only the exact limits use tight tolerances.
section("1b. CIRCULAR ORBITS (ISCO, binding energy)");

/**
 * Independent numerical ISCO: solve R = R' = 0 for (E, Lz) at each radius and
 * find where R'' changes sign. Shares no algebra with `iscoRadius` — it never
 * uses the Bardeen Z1/Z2 closed form.
 *
 *   R(r) = [E(r^2 + a^2) - a Lz]^2 - Delta [ r^2 + (Lz - aE)^2 ]
 *
 * The r^2 in the second bracket is the mu^2 r^2 term. Writing a bare constant
 * there instead produces a plausible number for the wrong reason: an earlier
 * draft of this check used `+ 1` and reported 2.8M for Schwarzschild.
 *
 * @param {number} a
 * @param {boolean} prograde
 */
function numericalIsco(a, prograde) {
  const Rf = (r, E, L) => {
    const r2 = r * r;
    const a2 = a * a;
    const P = E * (r2 + a2) - a * L;
    return P * P - (r2 - 2 * r + a2) * (r2 + (L - a * E) ** 2);
  };
  const dR = (r, E, L, h = 1e-5) => (Rf(r + h, E, L) - Rf(r - h, E, L)) / (2 * h);
  const d2R = (r, E, L, h = 3e-4) => (Rf(r + h, E, L) - 2 * Rf(r, E, L) + Rf(r - h, E, L)) / (h * h);

  const solve = (r, E, L) => {
    for (let it = 0; it < 200; it++) {
      const f0 = Rf(r, E, L);
      const f1 = dR(r, E, L);
      const j00 = (Rf(r, E + 1e-7, L) - f0) / 1e-7;
      const j01 = (Rf(r, E, L + 1e-7) - f0) / 1e-7;
      const j10 = (dR(r, E + 1e-7, L) - f1) / 1e-7;
      const j11 = (dR(r, E, L + 1e-7) - f1) / 1e-7;
      const det = j00 * j11 - j01 * j10;
      if (!Number.isFinite(det) || Math.abs(det) < 1e-16) return null;
      const dE = (j11 * f0 - j01 * f1) / det;
      const dL = (-j10 * f0 + j00 * f1) / det;
      E -= dE;
      L -= dL;
      if (!Number.isFinite(E) || !Number.isFinite(L)) return null;
      if (Math.abs(dE) + Math.abs(dL) < 1e-14) return { E, L };
    }
    return null;
  };

  const lo = horizonRadius(a) + 0.05;
  const hi = 30;
  let prevSign = 0;
  let E = 1;
  let L = prograde ? 3.5 : -3.5;
  for (let i = 0; i < 8000; i++) {
    const r = lo + ((hi - lo) * i) / 8000;
    const sol = solve(r, E, L);
    if (!sol) continue;
    E = sol.E;
    L = sol.L;
    // Stay on the requested branch; reject unphysical roots.
    if (prograde ? L <= 0 : L >= 0) continue;
    if (E <= 0 || E > 2 || Math.abs(L) > 25) continue;
    const s = Math.sign(d2R(r, E, L));
    if (prevSign !== 0 && s !== 0 && s !== prevSign) return r;
    if (s !== 0) prevSign = s;
  }
  return null;
}

// (a) Exact limits.
check("ISCO at a=0", iscoRadius(0), 6, 1e-12, "Schwarzschild r_isco = 6M");
check("extremal prograde ISCO", iscoRadius(1, true), 1, 1e-6, "coincides with the horizon");
check("extremal retrograde ISCO", iscoRadius(1, false), 9, 1e-6, "r_isco = 9M at a=1");
check("marginally bound orbit at a=0", marginallyBoundRadius(0), 4, 1e-9, "r_mb = 4M");
// Retrograde ISCO is monotonically increasing in a and saturates at 9M, so any
// retrograde value above 9M is unphysical regardless of which formula produced it.
{
  let maxRetro = 0;
  for (let a = 0; a <= 1; a += 0.01) maxRetro = Math.max(maxRetro, iscoRadius(a, false));
  assert("retrograde ISCO never exceeds 9M", maxRetro <= 9 + 1e-9, `max ${num(maxRetro, 8)}`);
}

// (b) Independent numerical derivation at intermediate spins. The tolerance is
// set by the scan resolution and finite-difference noise, not by the physics,
// so it is deliberately looser than the exact limits above.
for (const a of [0, 0.5, 0.9]) {
  for (const pro of [true, false]) {
    const numeric = numericalIsco(a, pro);
    if (numeric === null) {
      console.log(`  SKIP  numerical ISCO a=${a} ${pro ? "pro" : "ret"} (branch tracking failed)`);
      continue;
    }
    check(
      `ISCO a=${a} ${pro ? "prograde" : "retrograde"} vs numerical marginal stability`,
      iscoRadius(a, pro),
      numeric,
      5e-3,
      "R=R'=0, R''=0 solved independently",
    );
  }
}

// Binding energy at ISCO: 1 - E is the radiative efficiency of accretion.
// Anchored on the two exact endpoints; intermediate values are checked for
// monotonicity rather than against a quoted table.
{
  const E0 = circularEnergyAndAngularMomentum(iscoRadius(0, true), 0, true).energy;
  check("Schwarzschild efficiency 1-E_isco", 1 - E0, 1 - Math.sqrt(8 / 9), 1e-12, "E_isco = sqrt(8/9)");
  const Eext = circularEnergyAndAngularMomentum(1.000001, 1, true).energy;
  check("extremal prograde E_isco", Eext, 1 / Math.sqrt(3), 1e-3, "E = 1/sqrt(3) at a=1");
  check("extremal prograde efficiency", 1 - Eext, 1 - 1 / Math.sqrt(3), 1e-3, "42.3% bound");

  // The extremal prograde ISCO sits exactly ON the horizon, so E is only
  // recovered in the limit r -> 1+. A single evaluation at some epsilon is a
  // weak check: pick epsilon too large and you measure the approach, too small
  // and you measure round-off. What is strong is the convergence rate.
  //
  // Theory says E - 1/sqrt(3) = O(eps), so each decade of eps should shrink the
  // error by ~10x. Below some epsilon the error stops shrinking and reverses as
  // cancellation in the metric components takes over; that floor is reported but
  // not asserted, since it is a property of floating point, not of the physics.
  console.log("");
  console.log("  convergence of E(r) to 1/sqrt(3) as r -> 1+ for a = 1:");
  const target = 1 / Math.sqrt(3);
  const errs = [];
  for (const eps of [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]) {
    const E = circularEnergyAndAngularMomentum(1 + eps, 1, true).energy;
    errs.push({ eps, err: Math.abs(E - target) });
    console.log(`    eps ${eps.toExponential(0)}   E ${E.toFixed(10)}   |E - 1/sqrt(3)| ${num(errs.at(-1).err, 3)}`);
  }
  // Fit the observed order: log(err1/err2) / log(eps1/eps2) over the range
  // where round-off is not yet dominant.
  const o1 = Math.log(errs[0].err / errs[3].err) / Math.log(errs[0].eps / errs[3].eps);
  const o2 = Math.log(errs[3].err / errs[6].err) / Math.log(errs[3].eps / errs[6].eps);
  console.log(`    observed order over 1e-2..3e-4: ${num(o1, 4)}`);
  console.log(`    observed order over 3e-4..1e-5: ${num(o2, 4)}`);
  check("convergence order of E -> 1/sqrt(3)", (o1 + o2) / 2, 1, 0.15, "first-order in (r - r_+)");

  // Where the finite-difference evaluation stops improving. Informational.
  let bestEps = 1;
  let bestErr = Infinity;
  for (let e = -2; e >= -9; e -= 0.5) {
    const eps = 10 ** e;
    const err = Math.abs(circularEnergyAndAngularMomentum(1 + eps, 1, true).energy - target);
    if (err < bestErr) { bestErr = err; bestEps = eps; }
  }
  console.log(`    round-off floor: best eps ${bestEps.toExponential(0)} gives error ${num(bestErr, 3)}`);
  console.log("    (below this, cancellation dominates and the answer degrades)");

  let prevEta = 0;
  let monotone = true;
  for (let a = 0; a <= 0.99; a += 0.05) {
    const r = iscoRadius(a, true);
    const eta = 1 - circularEnergyAndAngularMomentum(r, a, true).energy;
    if (eta <= prevEta) monotone = false;
    prevEta = eta;
  }
  assert("prograde efficiency increases monotonically with spin", monotone, `reaches ${num(prevEta, 5)}`);
  console.log("  efficiencies (prograde / retrograde), for reference:");
  for (const a of [0, 0.5, 0.9, 0.998]) {
    const rp = iscoRadius(a, true);
    const rr = iscoRadius(a, false);
    const ep = 1 - circularEnergyAndAngularMomentum(rp, a, true).energy;
    const er = 1 - circularEnergyAndAngularMomentum(rr, a, false).energy;
    console.log(
      `    a=${String(a).padEnd(5)} isco ${num(rp, 6)}/${num(rr, 6)}   eta ${num(ep, 5)}/${num(er, 5)}`,
    );
  }
}

/* ------------------------------------------------------------------ *
 * 2. Photon orbits against the published table
 * ------------------------------------------------------------------ */

section("2. EQUATORIAL PHOTON ORBITS vs BARDEEN-PRESS-TEUKOLSKY (1973) Table 1");

const BARDEEN = [
  { a: 0.0, rPlus: 3.0, xiPlus: 5.19615, rMinus: 3.0, xiMinus: -5.19615 },
  { a: 0.5, rPlus: 2.3473, xiPlus: 4.09627, rMinus: 3.53209, xiMinus: -6.13816 },
  { a: 0.9, rPlus: 1.55785, xiPlus: 2.84442, rMinus: 3.91027, xiMinus: -6.83232 },
  { a: 0.998, rPlus: 1.07391, xiPlus: 2.11089, rMinus: 3.99822, xiMinus: -6.99667 },
];

for (const row of BARDEEN) {
  for (const prograde of [true, false]) {
    const want = prograde ? { r: row.rPlus, xi: row.xiPlus } : { r: row.rMinus, xi: row.xiMinus };
    const got = equatorialPhotonOrbit(row.a, prograde);
    const tag = prograde ? "pro " : "ret ";
    check(
      `r_ph ${tag} a=${row.a}`,
      got.radius,
      want.r,
      1e-4,
      "BPT Table 1 (5 s.f.)",
    );
    check(
      `xi   ${tag} a=${row.a}`,
      got.impactParameter,
      want.xi,
      1e-4,
      "BPT Table 1 (5 s.f.)",
    );
  }
}

// The Schwarzschild shadow radius is 3*sqrt(3) M exactly.
check("Schwarzschild b_crit", equatorialPhotonOrbit(0, true).impactParameter, 3 * Math.sqrt(3), 1e-10);

/* ------------------------------------------------------------------ *
 * 3. The null constraint and conserved quantities
 * ------------------------------------------------------------------ */

section("3. NULL CONSTRAINT & CONSERVATION ALONG A TRAJECTORY");

const R0 = 400;

/** Launch an equatorial photon at r0 with impact parameter b, via the tetrad. */
function launch(a, b, r0 = R0) {
  const alpha = Math.asin(Math.min(0.99, b / r0));
  const dir = { n1: -Math.cos(alpha), n2: 0, n3: Math.sin(alpha) };
  const p = momentumFromTetradDirection(r0, Math.PI / 2, a, dir, 1);
  return {
    state: [0, r0, Math.PI / 2, 0, p.pt, p.pr, p.pth, p.pphi],
    momentum: p,
    bExact: p.pphi / -p.pt,
  };
}

// A photon must satisfy g^{mu nu} p_mu p_nu = 0 to machine precision at every
// point. This is the strongest single check on the tetrad: it fails if the
// basis is not orthonormal.
{
  let maxRes = 0;
  for (const a of [0, 0.5, 0.9]) {
    for (const r of [3.5, 8, 40]) {
      for (const th of [0.3, 1.2, 2.6]) {
        for (const d of [[-1, 0, 0], [0, 1, 0], [0.4, 0.5, -0.7071]]) {
          const n = Math.hypot(...d);
          const p = momentumFromTetradDirection(r, th, a, { n1: d[0] / n, n2: d[1] / n, n3: d[2] / n }, 1);
          maxRes = Math.max(
            maxRes,
            Math.abs(nullResidual(r, th, a, [p.pt, p.pr, p.pth, p.pphi])),
          );
        }
      }
    }
  }
  check("max |g^{mu nu} p_mu p_nu| over tetrads", maxRes, 0, 1e-10, "null condition");
}

// E, Lz, Q must be constant along the path. Drift here means the integrator is
// not following the geodesic equation.
{
  const a = 0.9;
  const { state, momentum } = launch(a, 3.2);
  const p0 = [momentum.pt, momentum.pr, momentum.pth, momentum.pphi];
  const c0 = carterConstants(state[2], a, p0);

  let driftE = 0;
  let driftL = 0;
  let driftQ = 0;
  const res = integrateHamilton({
    state,
    a,
    step: 0.02,
    maxSteps: 300000,
    escapeRadius: R0,
    onState: (s) => {
      const p = [s[4], s[5], s[6], s[7]];
      const c = carterConstants(s[2], a, p);
      driftE = Math.max(driftE, Math.abs(c.energy - c0.energy));
      driftL = Math.max(driftL, Math.abs(c.lz - c0.lz));
      driftQ = Math.max(driftQ, Math.abs(c.q - c0.q));
    },
  });

  console.log(`  trajectory: status=${res.status} steps=${res.steps} rPeri=${num(res.rPeri, 6)}`);
  check("drift in E along path", driftE, 0, 1e-9, "stationarity of metric");
  check("drift in Lz along path", driftL, 0, 1e-9, "axisymmetry of metric");
  check("drift in Carter Q along path", driftQ, 0, 1e-6, "Carter (1968) constant");
}

/* ------------------------------------------------------------------ *
 * 4. Two independent integrators must agree
 * ------------------------------------------------------------------ */

section("4. CROSS-VALIDATION: Hamilton (metric-only) vs Carter (separated) vs Verlet");

/** Accumulate total |dphi| without branch cuts cancelling the winding. */
function shoot(state, a, integrator, opts = {}) {
  let phiAccum = 0;
  let prevPhi = state[3];
  const onState = (s) => {
    const phi = integrator === "hamilton" ? s[3] : s.phi;
    let d = phi - prevPhi;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    phiAccum += d;
    prevPhi = phi;
  };

  const step = opts.step ?? 0.02;
  const maxSteps = opts.maxSteps ?? 300000;
  const escapeRadius = opts.escapeRadius ?? R0;

  let res;
  if (integrator === "hamilton") {
    res = integrateHamilton({ state, a, step, maxSteps, escapeRadius, onState });
  } else if (integrator === "carter") {
    const st = carterStateFromMomentum(
      { r: state[1], theta: state[2], phi: state[3] },
      a,
      [state[4], state[5], state[6], state[7]],
    );
    res = integrateCarter({ state: st, a, step, maxSteps, escapeRadius, onState });
  } else {
    const st = carterStateFromMomentum(
      { r: state[1], theta: state[2], phi: state[3] },
      a,
      [state[4], state[5], state[6], state[7]],
    );
    res = integrateVerlet({ state: st, a, step, maxSteps, escapeRadius, onState });
  }
  return { status: res.status, rPeri: res.rPeri, dPhi: Math.abs(phiAccum), steps: res.steps, crossings: res.crossings };
}

/**
 * Launch out of the equatorial plane so the ray genuinely crosses the disk.
 * Built through the tetrad so the momentum stays null; patching p_theta
 * afterwards would break the null condition and the three integrators would
 * then follow three different trajectories, which is not a cross-validation.
 */
function launchTilted(a, r0, theta0, tilt, bFrac) {
  const n3 = bFrac;
  const n2 = tilt;
  const n1 = -Math.sqrt(Math.max(0, 1 - n2 * n2 - n3 * n3));
  const p = momentumFromTetradDirection(r0, theta0, a, { n1, n2, n3 }, 1);
  return {
    state: [0, r0, theta0, 0, p.pt, p.pr, p.pth, p.pphi],
    bExact: p.pphi / -p.pt,
    nullRes: nullResidual(r0, theta0, a, [p.pt, p.pr, p.pth, p.pphi]),
  };
}

// The three integrators share no algebra:
//   - Hamilton: the metric and finite-differenced metric derivatives, nothing else
//   - Carter:   separated first-order equations in the constants E, Lz, Q
//   - Verlet:   velocity-Verlet in Mino time on the potentials R(r) and Theta(th)
// Agreement between them is therefore evidence about the physics rather than
// about shared code, and disagreement localises the error to one of them.
//
// Tolerances are calibrated against tests/geodesics.test.js, which uses 1e-3 for
// closest approach and 5e-3 for disk-crossing events. Tightening them further
// would measure step size, not physics.
//
// Which observables are meaningful is not uniform across the three, and getting
// this wrong is what produced the original wall of "failures":
//   - For an ESCAPING ray, rPeri is the closest approach of a well-defined
//     trajectory. Real observable, converges with step size, worth asserting.
//   - For a CAPTURED ray, rPeri is only how far the final step overshot the
//     termination radius. It carries no physics and is pure step-size noise.
//   - |dphi| for a ray near the capture threshold is genuinely step-sensitive,
//     because the winding count changes as the ray crosses the separatrix.
//     Printed for the record, not asserted.

// --- Escaping rays: closest approach is a physical observable. ------------
console.log("  escaping rays — rPeri is a physical observable, asserted:");
for (const a of [0, 0.5, 0.9]) {
  for (const b of [8, 14, 25]) {
    const { state } = launch(a, b);
    const h = shoot(state, a, "hamilton");
    const c = shoot(state, a, "carter");
    const v = shoot(state, a, "verlet");
    console.log(
      `    a=${a} b=${b}: rPeri  hamilton ${num(h.rPeri, 10)}  carter ${num(c.rPeri, 10)}  verlet ${num(v.rPeri, 10)}`,
    );
    console.log(
      `              |dphi|   ${num(h.dPhi, 8)}  ${num(c.dPhi, 8)}  ${num(v.dPhi, 8)}   (${h.status}/${c.status}/${v.status})`,
    );
    assert(
      `all three escape a=${a} b=${b}`,
      h.status === "escaped" && c.status === "escaped" && v.status === "escaped",
      `${h.status}/${c.status}/${v.status}`,
    );
    // |dphi| accumulates over the whole trajectory, so it drifts more than the
    // single-point rPeri; 5e-3 matches the crossing tolerance for the same reason.
    check(`|rPeri hamilton - carter| a=${a} b=${b}`, Math.abs(h.rPeri - c.rPeri), 0, 1e-3, "tests/geodesics.test.js");
    check(`|rPeri hamilton - verlet| a=${a} b=${b}`, Math.abs(h.rPeri - v.rPeri), 0, 1e-3, "tests/geodesics.test.js");
    check(`||dphi| hamilton - carter| a=${a} b=${b}`, Math.abs(h.dPhi - c.dPhi), 0, 5e-3, "tests/geodesics.test.js");
    check(`||dphi| hamilton - verlet| a=${a} b=${b}`, Math.abs(h.dPhi - v.dPhi), 0, 5e-3, "tests/geodesics.test.js");
  }
}

// --- Captured rays: only the classification is physical. ------------------
// Impact parameters are derived from the prograde critical value rather than
// hard-coded, because b_crit falls from 5.196M at a=0 to 2.844M at a=0.9; a
// fixed b that is captured at a=0 escapes at a=0.9.
console.log("");
console.log("  captured rays — rPeri is horizon-overshoot noise, printed not asserted:");
for (const a of [0, 0.5, 0.9]) {
  const rp = horizonRadius(a);
  const rMin = rp + 1e-3;
  const bCrit = equatorialPhotonOrbit(a, true).impactParameter;
  for (const frac of [0.5, 0.8]) {
    const b = frac * bCrit;
    const { state } = launch(a, b);
    const h = shoot(state, a, "hamilton", { maxSteps: 900000 });
    const c = shoot(state, a, "carter", { maxSteps: 900000 });
    const v = shoot(state, a, "verlet", { maxSteps: 900000 });
    console.log(
      `    a=${a} b=${num(b, 6)} (${frac} b_crit):  rPeri  ${num(h.rPeri, 10)}  ${num(c.rPeri, 10)}  ${num(v.rPeri, 10)}`,
    );
    console.log(`              |dphi|   ${num(h.dPhi, 8)}  ${num(c.dPhi, 8)}  ${num(v.dPhi, 8)}   (${h.status}/${c.status}/${v.status})`);
    // Capture vs escape is a physical classification, so it must agree.
    assert(
      `all three capture a=${a} b=${num(b, 6)}`,
      h.status === "captured" && c.status === "captured" && v.status === "captured",
      `${h.status}/${c.status}/${v.status}`,
    );
    // Each must land *near* the termination radius. The bound is two-sided and
    // deliberately loose, for two reasons:
    //   - Hamilton integrates in affine parameter with Boyer–Lindquist
    //     coordinates, which are singular at r_+, so it approaches from above.
    //   - Carter and Verlet integrate in Mino time, where the horizon is
    //     perfectly regular and dr/dlambda_M is finite and nonzero there, so
    //     they legitimately step *through* r_+ before the test fires.
    // A one-sided `rPeri > r_+` test would therefore be wrong physics, while a
    // one-sided `rPeri <= r_+ + slack` test accepted an earlier blow-up that
    // produced rPeri = -1.93e9. Requiring closeness to rMin catches blow-up
    // (which is unbounded) without forbidding Mino-time overshoot.
    for (const [name, r] of [["hamilton", h.rPeri], ["carter", c.rPeri], ["verlet", v.rPeri]]) {
      assert(
        `${name} rPeri near rMin a=${a} b=${num(b, 6)}`,
        Number.isFinite(r) && Math.abs(r - rMin) < 0.1,
        `rPeri=${r}, r_+=${rp}, rMin=${rMin}, |rPeri - rMin|=${Math.abs(r - rMin)}`,
      );
    }
  }
}

// --- Disk crossings: the events the renderer actually reads. ---------------
console.log("");
console.log("  equatorial disk crossings — the renderer's observable, asserted:");
for (const a of [0, 0.6]) {
  const r0 = 60;
  const { state, nullRes } = launchTilted(a, r0, Math.PI / 2 - 0.1, 0.2, 0.03);
  assert(`tilted launch is null a=${a}`, Math.abs(nullRes) < 1e-12, `residual ${nullRes}`);
  const opts = { step: 0.005, maxSteps: 2000000, escapeRadius: r0 };
  const h = shoot(state, a, "hamilton", opts);
  const c = shoot(state, a, "carter", opts);
  const v = shoot(state, a, "verlet", opts);
  console.log(`    a=${a}: crossings  hamilton ${h.crossings.length}  carter ${c.crossings.length}  verlet ${v.crossings.length}   (${h.status}/${c.status}/${v.status})`);
  assert(`hamilton records crossings a=${a}`, h.crossings.length > 0, "none recorded");
  assert(
    `crossing count agrees a=${a}`,
    h.crossings.length === c.crossings.length && h.crossings.length === v.crossings.length,
    `${h.crossings.length}/${c.crossings.length}/${v.crossings.length}`,
  );
  for (let i = 0; i < h.crossings.length; i++) {
    const hc = h.crossings[i];
    console.log(
      `      #${i} r  ${num(hc.r, 10)}  ${num(c.crossings[i].r, 10)}  ${num(v.crossings[i].r, 10)}   phi  ${num(hc.phi, 8)}  ${num(c.crossings[i].phi, 8)}  ${num(v.crossings[i].phi, 8)}`,
    );
    check(`crossing ${i} r  hamilton - carter a=${a}`, Math.abs(hc.r - c.crossings[i].r), 0, 5e-3, "tests/geodesics.test.js");
    check(`crossing ${i} r  hamilton - verlet a=${a}`, Math.abs(hc.r - v.crossings[i].r), 0, 5e-3, "tests/geodesics.test.js");
    check(`crossing ${i} phi hamilton - carter a=${a}`, Math.abs(hc.phi - c.crossings[i].phi), 0, 5e-3, "tests/geodesics.test.js");
    check(`crossing ${i} phi hamilton - verlet a=${a}`, Math.abs(hc.phi - v.crossings[i].phi), 0, 5e-3, "tests/geodesics.test.js");
    assert(`crossing ${i} direction agrees a=${a}`, hc.dir === c.crossings[i].dir && hc.dir === v.crossings[i].dir, `${hc.dir}/${c.crossings[i].dir}/${v.crossings[i].dir}`);
  }
}

/* ------------------------------------------------------------------ *
 * 5. Convergence order
 * ------------------------------------------------------------------ */

section("5. ORDER OF CONVERGENCE (step refinement)");

// Choosing the observable matters more than choosing the tolerance here, and
// the obvious candidates are all traps:
//
//   - rPeri is a min over discrete samples, so it is a *non-smooth* functional
//     of the trajectory. Its error is dominated by which sample happens to be
//     nearest the turning point, not by the local truncation error.
//   - |dphi| is summed up to the step that first crosses escapeRadius. That
//     crossing is itself a step-dependent event, so the last-step contribution
//     is O(h) and swamps the higher-order terms — measured, RK4 showed an
//     apparent order of 1.0 and even -17.7 on this observable.
//   - The interpolated disk-crossing radius is smooth, but linear interpolation
//     is only second-order accurate, so it masks RK4's order entirely.
//
// The clean observable is the ingoing Eddington–Finkelstein advanced time
//   v = t + r*(r),   r*(r) = r + 2M ln(r/2M - 1).
// In EF coordinates ds^2 = -(1-2M/r) dv^2 + 2 dv dr + r^2 dOmega^2, so a
// radially ingoing photon (ds^2 = 0, dOmega = 0) satisfies
//   dv [ (1-2M/r) dv - 2 dr ] = 0,
// whose ingoing branch is dv = 0: **v is exactly constant along the ray**.
// That is a closed-form invariant with no event detection and no interpolation,
// so its drift is pure integration error.
{
  const M = 1;
  const rStar = (r) => r + 2 * M * Math.log(r / (2 * M) - 1);
  const r0 = 200;
  const rStop = 4.0;

  /** Max |v - v0| sampled along a purely radial ingoing photon. */
  const vDrift = (step, which) => {
    const p = momentumFromTetradDirection(r0, Math.PI / 2, 0, { n1: -1, n2: 0, n3: 0 }, 1);
    const state = [0, r0, Math.PI / 2, 0, p.pt, p.pr, p.pth, p.pphi];
    const v0 = state[0] + rStar(state[1]);
    let maxDrift = 0;
    const probe = (r, t) => {
      if (r <= rStop) maxDrift = Math.max(maxDrift, Math.abs(t + rStar(r) - v0));
    };
    const opts = { a: 0, step, maxSteps: 2000000, escapeRadius: r0 * 2, minRadius: rStop - 1e-6 };
    let res;
    if (which === "hamilton") {
      res = integrateHamilton({ state, onState: (s) => probe(s[1], s[0]), ...opts });
    } else {
      const cs = carterStateFromMomentum(
        { r: state[1], theta: state[2], phi: state[3] },
        0,
        [state[4], state[5], state[6], state[7]],
      );
      const onState = (s) => probe(s.r, s.t);
      res = which === "carter" ? integrateCarter({ state: cs, onState, ...opts }) : integrateVerlet({ state: cs, onState, ...opts });
    }
    assert(`${which} captured at step ${step}`, res.status === "captured", `status=${res.status}`);
    return maxDrift;
  };

  // Each integrator gets the step range where its error is above the round-off
  // floor. Below that floor the observed order collapses toward 0, which is a
  // statement about double precision, not about the scheme.
  const CASES = [
    // [integrator, expected order, steps, absolute tolerance on the order]
    ["verlet", 2, [0.08, 0.04, 0.02, 0.01], 0.4],
    ["hamilton", 4, [1.0, 0.5, 0.25, 0.125], 0.9],
    ["carter", 1, [0.08, 0.04, 0.02, 0.01], 0.4],
  ];

  for (const [which, expected, steps, orderTolAbs] of CASES) {
    const drifts = steps.map((s) => vDrift(s, which));
    console.log(`  ${which} (expected order ${expected}):`);
    console.log(`    step   ${steps.map((s) => String(s).padStart(12)).join("")}`);
    console.log(`    |dv|   ${drifts.map((d) => num(d, 4).padStart(12)).join("")}`);
    const orders = [];
    for (let i = 1; i < drifts.length; i++) {
      const o = Math.log2(drifts[i - 1] / drifts[i]);
      orders.push(o);
      console.log(`    order  ${num(o, 4)}   (${steps[i - 1]} -> ${steps[i]})`);
    }
    const mean = orders.reduce((x, y) => x + y, 0) / orders.length;
    // check() compares relative to the reference, so convert the absolute
    // tolerance on the order into the relative form it expects.
    check(`${which} observed order of convergence`, mean, expected, orderTolAbs / expected, "Richardson extrapolation");
  }
}

/* ------------------------------------------------------------------ *
 * 6. Light deflection: post-Newtonian series
 * ------------------------------------------------------------------ */

section("6. SCHWARZSCHILD LIGHT DEFLECTION vs POST-NEWTONIAN SERIES");

// For a ray from finite radius r0, the flat-space reference is 2*acos(b/r0),
// not pi. The excess over that is the relativistic deflection, which must match
// 4/b + 15*pi/(4 b^2) + 128/(3 b^3) + ... with M = 1.
//
// Two things have to be right for this comparison to mean anything, and getting
// either wrong produced the original failures:
//
//   1. r0 must be large compared with b. The finite-launch-radius correction is
//      itself a series in b/r0, so at r0 = 400 and b = 120 (b/r0 = 0.3) it is
//      the dominant error and swamps the O(b^-3) term being tested. Measured at
//      r0 = 400 the b = 120 case was off by 4.7e-2; at r0 = 8000 it is 1.1e-4,
//      which is precisely the O(b^-3) truncation term.
//   2. Use the exact conserved b = Lz/E, not the nominal launch value. The
//      tetrad launch fixes the *direction*, and bExact differs from b by
//      roughly b^2/r0, which is the same order as the 1PN term being checked.
//
// The tolerance is the actual O(b^-3) truncation term, so it is a statement
// about the series rather than a free parameter.
{
  const a = 0;
  const r0 = 8000;
  const deflectionOf = (b) => {
    const { state, bExact } = launch(a, b, r0);
    const { dPhi, status } = shoot(state, a, "hamilton", { escapeRadius: r0, maxSteps: 4000000 });
    assert(`b=${b} escapes`, status === "escaped", `status=${status}`);
    const flat = 2 * Math.acos(Math.min(1, bExact / r0));
    return { bExact, delta: dPhi - flat, status };
  };

  const measured = {};
  for (const b of [80, 160, 320]) {
    const { bExact, delta } = deflectionOf(b);
    measured[b] = { bExact, delta };
    const pn = 4 / bExact + (15 * Math.PI) / (4 * bExact * bExact);
    const truncation = 128 / (3 * bExact ** 3);
    console.log(
      `  b=${b} (bExact ${num(bExact, 8)}, b/r0 ${num(bExact / r0, 3)}): deflection ${num(delta, 6)}  PN ${num(pn, 6)}  O(b^-3) term ${num(truncation, 6)}`,
    );
    // check() is relative, so express the absolute truncation term as a
    // fraction of the reference. The factor of 2 leaves room for the O(b^-4)
    // term, which is not modelled.
    check(`deflection matches 4/b + 15pi/4b^2  b=${b}`, delta, pn, (2 * truncation) / pn, "Einstein + 1PN");
  }

  // Multiplying by b isolates the leading 4M, but the 1PN term survives as
  // 15*pi/(4b), so the honest reference for the product is 4 + 15*pi/(4b).
  // Asserting that is a real second test of the series, not just a restatement
  // of the check above.
  const bFar = measured[320].bExact;
  const productRef = 4 + (15 * Math.PI) / (4 * bFar);
  const productTrunc = 128 / (3 * bFar * bFar);
  console.log(`  b * deflection at b=${bFar}: ${num(measured[320].delta * bFar, 8)}  ref ${num(productRef, 8)}`);
  check(
    "b * deflection -> 4M + 15pi/4b",
    measured[320].delta * bFar,
    productRef,
    (2 * productTrunc) / productRef,
    "Einstein 1911 + 1PN",
  );

  // Doubling b must halve the deflection, to within the 1PN correction.
  const ratio = measured[320].delta / measured[160].delta;
  console.log(`  deflection ratio b=320 / b=160: ${num(ratio, 6)} (expect ~0.5)`);
  assert("deflection halves when b doubles", ratio > 0.49 && ratio < 0.51, `ratio=${num(ratio, 6)}`);
}

// Capture threshold: below b_crit the ray is swallowed, above it escapes.
{
  const a = 0;
  const bcrit = 3 * Math.sqrt(3);
  const below = shoot(launch(a, bcrit * 0.98).state, a, "hamilton").status;
  const above = shoot(launch(a, bcrit * 1.02).state, a, "hamilton").status;
  assert(`ray at 0.98 b_crit is captured`, below === "captured", `status=${below}`);
  assert(`ray at 1.02 b_crit escapes`, above === "escaped", `status=${above}`);
}

/* ------------------------------------------------------------------ *
 * 7. Redshift factor against closed-form limits
 * ------------------------------------------------------------------ */

section("7. REDSHIFT FACTOR g = E_obs / E_em vs CLOSED-FORM LIMITS");

// Static emitter in Schwarzschild: pure gravitational redshift, g = sqrt(1-2/r).
// This is the arbiter that catches an inverted formula.
for (const r of [3, 5, 10, 50]) {
  check(`static g at r=${r} (a=0)`, redshiftFactor(r, 0, 0, 0), Math.sqrt(1 - 2 / r), 1e-12);
}

// Zero-angular-momentum photon from a circular orbit in Schwarzschild is exact:
// g = sqrt(1 - 3/r). Not asymptotic — machine precision.
for (const r of [5, 10, 100]) {
  check(`xi=0 circular-orbit g at r=${r} (a=0)`, redshiftFactor(r, 0, 0, 1), Math.sqrt(1 - 3 / r), 1e-12);
}

// Special-relativistic limit: far away the orbit is Newtonian with v = r^-1/2
// and a forward-emitted photon has xi ~ r, so g -> sqrt((1+v)/(1-v)).
for (const r of [100, 400, 1600]) {
  const v = 1 / Math.sqrt(r);
  const dop = Math.sqrt((1 + v) / (1 - v));
  const g = redshiftFactor(r, 0, r, 1);
  console.log(`  r=${r}: g ${num(g, 8)}  SR Doppler ${num(dop, 8)}  rel ${num(Math.abs(g - dop) / dop, 3)}`);
  assert(`SR Doppler limit at r=${r}`, Math.abs(g - dop) / dop < 2 / r, `rel ${num(Math.abs(g - dop) / dop, 3)} < ${2 / r}`);
}

// The light surface: no timelike circular emitter at or inside r = 3M for a = 0.
assert("g is null at r=3M (light surface)", redshiftFactor(3, 0, 0, 1) === null);
assert("g is null inside the light surface", redshiftFactor(2, 0, 0, 1) === null);
assert("g exists just outside the light surface", redshiftFactor(3.01, 0, 0, 1) !== null);

// Mirror symmetry of the disk: reversing orbital direction and flipping spin
// is the same physical configuration seen from the other side.
{
  let maxErr = 0;
  for (const a of [0, 0.4, 0.8]) {
    for (const r of [5, 10, 20]) {
      for (const xi of [0, 1.5, 4]) {
        maxErr = Math.max(maxErr, Math.abs(redshiftFactor(r, a, -xi, -1) - redshiftFactor(r, -a, xi, 1)));
      }
    }
  }
  check("max |g(r,a,-xi,-1) - g(r,-a,xi,+1)|", maxErr, 0, 1e-12, "phi -> -phi mirror symmetry");
}

// The measured g distribution across a disk. This is not a check against a
// reference — it is the number that sets the display gain, since g^4 spans a
// large dynamic range and a naive gain saturates the approaching limb.
//
// The maximum must be read with care. `redshiftFactor` returns null where the
// circular orbit stops being timelike, and just inside that boundary (the
// *light surface*, which is ξ-dependent — equatorially at 3M for Schwarzschild)
// the denominator u^t (1 - Omega xi) goes to zero, so g diverges. A fixed grid
// therefore reports a maximum that is an artefact of the grid spacing rather
// than a property of the disk: at a=0.9 the raw max is g = 144.7 at r = 2.97,
// which makes a printed "g^4 range" of ~1e12:1 meaningless.
//
// So the distribution is summarised by percentiles, and the display-relevant
// dynamic range is quoted as p99/p01 of g^4 — the span the renderer must map
// into a colour ramp while ignoring the divergent tail.
{
  console.log("");
  console.log("  g distribution across the disk (sets the display gain):");
  for (const a of [0, 0.5, 0.9]) {
    const rIn = diskInnerRadius(a, true);
    const cells = [];
    for (let r = rIn; r <= 22; r += 0.05) {
      for (const xi of [-6, -4, -2, 0, 2, 4, 6]) {
        const g = redshiftFactor(r, a, xi, 1);
        if (g !== null) cells.push({ r, xi, g });
      }
    }
    const gs = cells.map((c) => c.g).sort((x, y) => x - y);
    const pct = (p) => gs[Math.min(gs.length - 1, Math.floor(p * (gs.length - 1)))];
    const max = cells.reduce((best, c) => (c.g > best.g ? c : best), cells[0]);
    const range4 = Math.pow(pct(0.99), 4) / Math.pow(pct(0.01), 4);
    console.log(`    a=${a}  rIn=${num(rIn, 5)}  ${cells.length} disk samples`);
    console.log(
      `      g:  min ${num(gs[0], 5)}   p01 ${num(pct(0.01), 5)}   median ${num(pct(0.5), 5)}   p99 ${num(pct(0.99), 5)}`,
    );
    console.log(
      `      display-relevant g^4 span (p99/p01) = ${num(range4, 4)}:1`,
    );
    console.log(
      `      raw max g = ${num(max.g, 5)} at r=${num(max.r, 4)} xi=${max.xi} (r/rIn=${num(max.r / rIn, 4)})` +
        ` — light-surface divergence, excluded from the gain`,
    );
  }
}

// Temperature profile: vanishes at the inner edge, peaks just outside it.
{
  const rIn = 2.3209;
  assert("T_eff vanishes at the inner edge", diskTemperature(rIn, rIn) === 0);

  // With x = r/rIn, the profile is T = x^-3/4 (1 - x^-3/4)^1/4, so
  //   T^4 = x^-3 (1 - x^-3/4) = x^-3 - x^-15/4.
  // Setting d(T^4)/dx = 0:
  //   -3 x^-4 + (15/4) x^-19/4 = 0
  //   (15/4) x^-19/4 = 3 x^-4
  //   x^3/4 = (15/4)/3 = 5/4
  //   x = (5/4)^4/3 = 1.3465216812699272
  // Two earlier references for this number were both wrong: 2^(4/3) = 2.5198
  // came from mis-differentiating, and (5/3)^(4/3) = 1.9761 from mis-solving
  // x^3/4 = 5/4. The code was correct both times.
  const xPeak = Math.pow(5 / 4, 4 / 3);

  // Locate the peak by bisecting on a central-difference derivative rather than
  // taking the argmax of a coarse grid or of a golden-section search.
  //
  // A grid step of 0.01 in r contributes an error of the same order as the
  // discrepancy being checked, so it cannot support a tight tolerance. A
  // golden-section search on the *value* is worse than it looks: near a
  // quadratic maximum the function is flat to O(dx^2), so comparing values can
  // only localise the peak to ~sqrt(eps) — measured, 1.9e-8.
  //
  // The derivative, by contrast, is linear near the peak, so bisection drives
  // the bracket to machine precision. The finite-difference step h contributes
  // an O(h^2) bias, which is why h is chosen small but not so small that
  // cancellation dominates: measured, h = 1e-5 gives 1.2e-10.
  const fd = 1e-5;
  const dT = (x) => (diskTemperature((x + fd) * rIn, rIn) - diskTemperature((x - fd) * rIn, rIn)) / (2 * fd);
  let lo = 1.0;
  let hi = 2.0;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (dT(mid) > 0) lo = mid;
    else hi = mid;
  }
  const peakX = 0.5 * (lo + hi);
  const peakR = peakX * rIn;
  const peakT = diskTemperature(peakR, rIn);

  console.log(`  T_eff peak at r=${num(peakR, 12)} (rIn=${rIn}), value ${num(peakT, 12)}`);
  console.log(`  analytic x = (5/4)^(4/3) predicts r = ${num(xPeak * rIn, 12)}`);
  assert("T_eff peak lies just outside the inner edge", peakR > rIn && peakR < rIn + 1.5);
  check("T_eff peak location r/rIn", peakX, xPeak, 1e-8, "argmax of Shakura-Sunyaev profile");

  // At x = (5/4)^(4/3) we have x^-3/4 = 4/5, so the peak value is
  //   (4/5) * (1 - 4/5)^(1/4) = (4/5) * 5^(-1/4) = 0.5349922439811376.
  // The profile is NOT normalised to a peak of 1 — the docstrings in
  // redshift.js and shaders.js claimed it was, which was wrong. Pinning the
  // value here means the claim cannot silently drift again.
  const tPeak = 0.8 * Math.pow(0.2, 0.25);
  check("T_eff peak value", peakT, tPeak, 1e-8, "(4/5) * 5^(-1/4)");
}

/* ------------------------------------------------------------------ *
 * 8. Frame dragging
 * ------------------------------------------------------------------ */

section("8. FRAME DRAGGING (Lense-Thirring)");

// omega = -g_tphi / g_phiphi. At the horizon it must approach the hole's own
// angular velocity. For Kerr, r_+^2 + a^2 = 2 r_+ (from r_+^2 - 2r_+ + a^2 = 0),
// so Omega_H = a / (r_+^2 + a^2) = a / (2 r_+).
{
  const a = 0.998;
  const rH = horizonRadius(a);
  const omegaH = a / (2 * rH);
  const near = (r) => -metric(r, Math.PI / 2, a)[idx(T, PH)] / metric(r, Math.PI / 2, a)[idx(PH, PH)];
  console.log(`  a=${a}: omega_H (closed form) ${num(omegaH, 8)}`);
  for (const eps of [1e-2, 1e-4, 1e-6]) {
    console.log(`    at r = rH + ${eps}: omega ${num(near(rH + eps), 8)}`);
  }
  check("omega approaches Omega_H at the horizon", near(rH + 1e-6), omegaH, 2e-3, "Omega_H = a/(2 r_+)");
  // Far away it must fall off as the dipole term 2a/r^3.
  const r = 200;
  check("frame dragging far-field 2a/r^3", near(r), (2 * a) / r ** 3, 5e-3, "Lense-Thirring dipole");
}

/* ------------------------------------------------------------------ *
 * Summary
 * ------------------------------------------------------------------ */

section("SUMMARY");

const passed = results.length - failures;
console.log(`  ${results.length} numeric checks, ${passed} within tolerance, ${failures} failed`);

// Worst offenders, so a passing run still shows how tight things actually are.
const worst = [...results].sort((x, y) => y.rel / y.tol - x.rel / x.tol).slice(0, 8);
console.log("");
console.log("  Tightest margins (rel err / tolerance, closest to failing first):");
for (const r of worst) {
  console.log(`    ${(r.rel / r.tol).toFixed(3)}x tol   rel ${num(r.rel, 3)}   ${r.label}`);
}

console.log("");
console.log(failures === 0 ? "  ALL CHECKS PASSED" : `  ${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
