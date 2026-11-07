/**
 * Null geodesic integration in Kerr spacetime.
 *
 * This module deliberately contains TWO independent integrators:
 *
 *  1. `integrateHamilton` — the "reference" integrator. It uses only the
 *     inverse metric and its derivatives:
 *
 *         dx^mu/dlambda =  g^{mu nu} p_nu
 *         dp_mu/dlambda = -1/2 d_mu( g^{alpha beta} ) p_alpha p_beta
 *
 *     It knows nothing about the Carter constant. It is slow (metric
 *     derivatives are finite-differenced) but structurally simple and
 *     therefore trustworthy.
 *
 *  2. `integrateCarter` — the "fast" integrator, mirroring what the GPU shader
 *     does. It uses the separated Carter form with constants of motion E, Lz, Q.
 *
 * Because the two share no algebra, tests/ can assert they agree to high
 * precision. If the Carter constant or any sign convention is wrong, the
 * disagreement surfaces immediately instead of silently producing a
 * plausible-looking but wrong shadow.
 *
 * Carter separated equations for a NULL geodesic (mu = 0), M = 1:
 *
 *   P(r)      = E (r^2 + a^2) - a Lz
 *   R(r)      = P^2 - Delta [ (Lz - a E)^2 + Q ]
 *   Th(theta) = Q + a^2 E^2 cos^2 - Lz^2 cos^2 / sin^2
 *   Sigma dr/dlambda    = s_r sqrt(R)
 *   Sigma dtheta/dlambda= s_t sqrt(Th)
 *   Sigma dt/dlambda    = (A E - 2 a r Lz) / Delta
 *   Sigma dphi/dlambda  = 2 a r E / Delta + Lz (1/sin^2 - a^2/Delta)
 *
 * with A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta). The t and phi equations are
 * derived from xdot^mu = g^{mu nu} p_nu and are verified numerically against
 * `integrateHamilton` in tests/geodesics.test.js.
 */

import {
  inverseMetric,
  metric,
  idx,
  contract,
  horizonRadius,
} from "./kerr.js";

export const T = 0;
export const R = 1;
export const TH = 2;
export const PH = 3;

/** Affine-parameter state: [t, r, theta, phi, pt, pr, pth, pphi]. */
export const STATE_LEN = 8;

/** Floor so 1/sin^2(theta) never explodes at the poles. */
const SIN2_FLOOR = 1e-12;

/* ------------------------------------------------------------------ *
 * Constants of motion and separated potentials
 * ------------------------------------------------------------------ */

/**
 * Convert a covariant momentum into Carter constants of motion.
 *
 *   E  = -p_t
 *   Lz = p_phi
 *   Q  = p_theta^2 + cos^2(theta) [ Lz^2 / sin^2(theta) - a^2 E^2 ]
 *
 * @param {number} theta
 * @param {number} a
 * @param {ArrayLike<number>} pCov [pt, pr, pth, pphi]
 */
export function carterConstants(theta, a, pCov) {
  const energy = -pCov[0];
  const lz = pCov[3];
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const sin2 = Math.max(sinT * sinT, SIN2_FLOOR);
  const q =
    pCov[2] * pCov[2] +
    cosT * cosT * ((lz * lz) / sin2 - a * a * energy * energy);
  return { energy, lz, q };
}

/** Radial potential R(r) of the separated null geodesic equation. */
export function radialPotential(r, a, energy, lz, q) {
  const r2 = r * r;
  const a2 = a * a;
  const p = energy * (r2 + a2) - a * lz;
  const delta = r2 - 2.0 * r + a2;
  const lza = lz - a * energy;
  return p * p - delta * (lza * lza + q);
}

/** Polar potential Theta(theta) of the separated null geodesic equation. */
export function polarPotential(theta, a, energy, lz, q) {
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);
  const cos2 = cosT * cosT;
  const sin2 = Math.max(sinT * sinT, SIN2_FLOOR);
  return q + a * a * energy * energy * cos2 - (lz * lz * cos2) / sin2;
}

/**
 * Residual of the null condition g^{mu nu} p_mu p_nu = 0.
 * Should be ~0 for a correct photon; the tests assert this.
 */
export function nullResidual(r, theta, a, pCov) {
  const ginv = inverseMetric(r, theta, a);
  return contract(ginv, pCov, pCov);
}

/** Sigma = r^2 + a^2 cos^2(theta). */
export function sigmaOf(r, theta, a) {
  return r * r + a * a * Math.cos(theta) * Math.cos(theta);
}

/* ------------------------------------------------------------------ *
 * Reference integrator: Hamilton's equations, metric-derivative only.
 * ------------------------------------------------------------------ */

const FD = 1e-5;
const _ginv = new Float64Array(16);
const _dgR = new Float64Array(16);
const _dgT = new Float64Array(16);
const _plus = new Float64Array(16);
const _minus = new Float64Array(16);

/** Central-difference derivative of the inverse metric wrt r or theta. */
function dinverseMetric(r, theta, a, which, out) {
  if (which === R) {
    inverseMetric(r + FD, theta, a, _plus);
    inverseMetric(r - FD, theta, a, _minus);
  } else {
    inverseMetric(r, theta + FD, a, _plus);
    inverseMetric(r, theta - FD, a, _minus);
  }
  for (let i = 0; i < 16; i++) out[i] = (_plus[i] - _minus[i]) / (2 * FD);
  return out;
}

/**
 * Right-hand side of Hamilton's equations for a photon.
 * @param {ArrayLike<number>} s state [t,r,th,ph,pt,pr,pth,pph]
 * @param {number} a
 * @param {Float64Array} out
 */
export function hamiltonDerivative(s, a, out = new Float64Array(STATE_LEN)) {
  const r = s[1];
  const th = s[2];
  const ginv = inverseMetric(r, th, a, _ginv);

  // dx^mu/dlambda = g^{mu nu} p_nu
  for (let i = 0; i < 4; i++) {
    out[i] =
      ginv[idx(i, 0)] * s[4] +
      ginv[idx(i, 1)] * s[5] +
      ginv[idx(i, 2)] * s[6] +
      ginv[idx(i, 3)] * s[7];
  }

  // dp_mu/dlambda = -1/2 (d_mu g^{ab}) p_a p_b
  dinverseMetric(r, th, a, R, _dgR);
  dinverseMetric(r, th, a, TH, _dgT);
  let accR = 0;
  let accT = 0;
  for (let A = 0; A < 4; A++) {
    for (let B = 0; B < 4; B++) {
      const pp = s[4 + A] * s[4 + B];
      accR += _dgR[idx(A, B)] * pp;
      accT += _dgT[idx(A, B)] * pp;
    }
  }
  out[4] = 0; // p_t conserved (stationarity)
  out[5] = -0.5 * accR;
  out[6] = -0.5 * accT;
  out[7] = 0; // p_phi conserved (axisymmetry)
  return out;
}

const _k1 = new Float64Array(STATE_LEN);
const _k2 = new Float64Array(STATE_LEN);
const _k3 = new Float64Array(STATE_LEN);
const _k4 = new Float64Array(STATE_LEN);
const _stage = new Float64Array(STATE_LEN);
const _next = new Float64Array(STATE_LEN);

function stage(s, k, h) {
  for (let i = 0; i < STATE_LEN; i++) _stage[i] = s[i] + h * k[i];
  return _stage;
}

/**
 * One classical RK4 step of the reference integrator.
 *
 * @param {ArrayLike<number>} s
 * @param {number} a
 * @param {number} h
 * @param {Float64Array} out
 * @param {Float64Array} [stageR] optional 1-element scratch array; if given it
 *   receives the smallest `r` touched by any of the four *stages*. Callers use
 *   it to reject a step whose derivative evaluations left the physical region
 *   (see `stepHorizonSafe`). The accepted result is deliberately excluded: a
 *   captured ray is supposed to step past the termination radius.
 * @returns {Float64Array} out
 */
export function rk4Step(s, a, h, out = _next, stageR = null) {
  hamiltonDerivative(s, a, _k1);
  let rWorst = s[1] + 0.5 * h * _k1[1];
  hamiltonDerivative(stage(s, _k1, 0.5 * h), a, _k2);
  rWorst = Math.min(rWorst, s[1] + 0.5 * h * _k2[1]);
  hamiltonDerivative(stage(s, _k2, 0.5 * h), a, _k3);
  rWorst = Math.min(rWorst, s[1] + h * _k3[1]);
  hamiltonDerivative(stage(s, _k3, h), a, _k4);
  rWorst = Math.min(rWorst, s[1] + h * _k4[1]);
  for (let i = 0; i < STATE_LEN; i++) {
    out[i] = s[i] + (h / 6) * (_k1[i] + 2 * _k2[i] + 2 * _k3[i] + _k4[i]);
  }
  if (stageR) stageR[0] = rWorst;
  return out;
}

/**
 * Beyond this many halvings we stop subdividing and accept the step. Reaching
 * it would mean sitting essentially on the horizon, which the caller's
 * termination test should have caught first; the cap only bounds the work.
 */
const MAX_SUBDIVISIONS = 40;

/**
 * Take one RK4 step, rejecting and halving it whenever any stage would
 * evaluate the metric at or inside the horizon.
 *
 * Boyer–Lindquist coordinates are singular at `r_+`: `Delta` changes sign
 * there, and `g^{tt}`, `g^{tphi}`, `g^{phiph}` all carry `Delta` in the
 * denominator, so they diverge. An infalling ray's trajectory is perfectly
 * regular and `dr/dlambda` stays finite at the horizon, but `p_r` diverges, so
 * a fixed-size step can evaluate its *later stages* below `r_+`. Those stages
 * then feed a huge spurious `dp_r/dlambda` back into the accepted step.
 *
 * Measured, a = 0, b = 4, step 0.02: the step from r = 2.001125 evaluates its
 * fourth stage at r = 1.9837 — inside the horizon — and lands at r = 9.37,
 * then at r = -1.93e9. The same ray at step 0.005 terminates cleanly at 2.0008.
 *
 * Halving is cheap because only `p_r` diverges, so the required sub-step
 * shrinks roughly linearly with the distance to `r_+`. Escaping rays never
 * trip the test, so this costs four `Math.min` calls and no extra derivative
 * evaluations on the hot path.
 *
 * The guard must be the horizon itself, never the (larger) termination radius:
 * guarding above the singularity but below the termination test makes the loop
 * Zeno, because the accepted decrement eventually falls under one ulp of `r`
 * and the ray freezes just above the guard without ever satisfying it.
 *
 * @param {Float64Array} s state, read only
 * @param {number} a
 * @param {number} h requested step
 * @param {number} rGuard every stage must stay strictly above this
 * @param {Float64Array} out written on success
 * @param {Float64Array} scratch 1-element array for `rk4Step`'s `stageR`
 * @returns {number} the step size actually taken
 */
function stepHorizonSafe(s, a, h, rGuard, out, scratch) {
  let sub = h;
  for (let attempt = 0; ; attempt++) {
    rk4Step(s, a, sub, out, scratch);
    // A non-finite worst radius means a stage divided by Delta ~ 0; treat that
    // as a crossing so the step is rejected rather than accepted as NaN.
    const rWorst = scratch[0];
    if (Number.isFinite(rWorst) && rWorst > rGuard) return sub;
    if (attempt >= MAX_SUBDIVISIONS) return sub;
    sub *= 0.5;
  }
}

/**
 * Record the equatorial-plane crossing (theta = pi/2) between two states by
 * linear interpolation. A thin accretion disk lives in that plane, so this is
 * the event the renderer actually cares about.
 */
function equatorCrossing(prev, now) {
  const thPrev = prev[2];
  const thNow = now[2];
  if ((thPrev - Math.PI / 2) * (thNow - Math.PI / 2) >= 0) return null;
  const w = (Math.PI / 2 - thPrev) / (thNow - thPrev);
  return {
    r: prev[1] + w * (now[1] - prev[1]),
    phi: prev[3] + w * (now[3] - prev[3]),
    dir: thNow > thPrev ? 1 : -1,
  };
}

/**
 * Integrate a photon with the reference integrator until it escapes, is
 * captured, or the step budget is exhausted.
 *
 * @param {Object} opts
 * @param {ArrayLike<number>} opts.state initial [t,r,th,ph,pt,pr,pth,pph]
 * @param {number} opts.a dimensionless spin
 * @param {number} [opts.step=0.05] base affine step
 * @param {number} [opts.maxSteps=200000]
 * @param {number} [opts.escapeRadius=60]
 * @param {number} [opts.minRadius] defaults to just outside the horizon
 * @param {(r:number, theta:number, phi:number)=>boolean} [opts.stopWhen]
 * @param {(state:Float64Array)=>void} [opts.onState] called after each step
 * @returns {{status:string, state:Float64Array, steps:number, crossings:Array}}
 */
export function integrateHamilton(opts) {
  const {
    state,
    a,
    step = 0.05,
    maxSteps = 200000,
    escapeRadius = 60,
    minRadius,
    stopWhen,
    onState,
  } = opts;

  const s = Float64Array.from(state);
  const rMin = minRadius ?? horizonRadius(a) + 1e-3;
  // The coordinate singularity is at the outer horizon itself, not at the
  // termination radius. Guarding at `rMin` instead would be a Zeno trap: the
  // accepted decrement shrinks with the step until it is smaller than one ulp
  // of `r`, at which point `r` stops changing and the loop never terminates.
  const rGuard = horizonRadius(a);
  const crossings = [];
  const prev = new Float64Array(STATE_LEN);
  const _stageR = new Float64Array(1);
  let rPeri = s[1];

  // Termination is evaluated after each step: a ray starts at the camera
  // radius, so checking before the first step would report "escaped" at once.
  for (let i = 0; i < maxSteps; i++) {
    if (s[1] <= rMin) {
      return { status: "captured", state: s, steps: i, crossings, rPeri };
    }

    // Curvature is strongest near the hole: shrink the step there.
    const h = step * Math.min(1, Math.max(0.1, (s[1] - rMin) / 4));
    prev.set(s);
    stepHorizonSafe(s, a, h, rGuard, _next, _stageR);
    s.set(_next);
    rPeri = Math.min(rPeri, s[1]);

    const cross = equatorCrossing(prev, s);
    if (cross) crossings.push(cross);

    if (onState) onState(s);

    if (stopWhen && stopWhen(s[1], s[2], s[3])) {
      return { status: "hit", state: s, steps: i, crossings, rPeri };
    }
    if (s[1] >= escapeRadius) {
      return { status: "escaped", state: s, steps: i, crossings, rPeri };
    }
  }
  return { status: "budget", state: s, steps: maxSteps, crossings, rPeri };
}

/* ------------------------------------------------------------------ *
 * Fast integrator: separated Carter form (mirrors the GLSL shader).
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} CarterState
 * @property {number} r
 * @property {number} theta
 * @property {number} t
 * @property {number} phi
 * @property {number} energy
 * @property {number} lz
 * @property {number} q
 * @property {number} signR +1 or -1
 * @property {number} signT +1 or -1
 */

/**
 * Derive a Carter state from a covariant momentum, including the signs of the
 * radial and polar motion.
 * @param {{r:number, theta:number, t?:number, phi?:number}} init
 * @param {number} a
 * @param {ArrayLike<number>} pCov [pt, pr, pth, pphi]
 * @returns {CarterState}
 */
export function carterStateFromMomentum(init, a, pCov) {
  const { energy, lz, q } = carterConstants(init.theta, a, pCov);
  return {
    r: init.r,
    theta: init.theta,
    t: init.t ?? 0,
    phi: init.phi ?? 0,
    energy,
    lz,
    q,
    signR: pCov[1] >= 0 ? 1 : -1,
    signT: pCov[2] >= 0 ? 1 : -1,
  };
}

/**
 * Derivatives of the separated equations.
 * @param {CarterState} s
 * @param {number} a
 */
export function carterDerivative(s, a) {
  const { r, theta, energy, lz, q } = s;
  const r2 = r * r;
  const a2 = a * a;
  const delta = r2 - 2.0 * r + a2;
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  const sin2 = Math.max(sinT * sinT, SIN2_FLOOR);
  const sigma = r2 + a2 * cosT * cosT;

  const R_ = Math.max(0.0, radialPotential(r, a, energy, lz, q));
  const TH_ = Math.max(0.0, polarPotential(theta, a, energy, lz, q));
  const A = (r2 + a2) * (r2 + a2) - a2 * delta * sin2;

  return {
    dr: (s.signR * Math.sqrt(R_)) / sigma,
    dtheta: (s.signT * Math.sqrt(TH_)) / sigma,
    dt: (A * energy - 2.0 * a * r * lz) / (sigma * delta),
    dphi:
      (2.0 * a * r * energy) / (sigma * delta) +
      (lz * (1.0 / sin2 - a2 / delta)) / sigma,
  };
}

/**
 * Advance a Carter state by h.
 *
 * Two pathologies have to be handled, and both are easy to get wrong:
 *
 *  1. Numerical flicker at a turning point. For a ray exactly in the
 *     equatorial plane, Theta is identically zero analytically but flickers
 *     around zero numerically (cos(pi/2) is ~6e-17, not 0). A strict
 *     `Theta >= 0` test then fails and bisection collapses to zero progress.
 *     A scale-aware tolerance is required.
 *
 *  2. Zero velocity at a turning point. There sqrt(R) = 0, so dr/dlambda = 0
 *     and *no* first-order step can leave, whatever sign is chosen — the
 *     integrator stalls and burns its step budget. Writing
 *
 *         rdot^2 = V(r),   V = R / Sigma^2
 *
 *     the motion leaving a turning point from rest is
 *
 *         r = r_turn + (1/4) V'(r_turn) t^2
 *
 *     (differentiate rdot^2 = V to get rddot = V'/2, then integrate from rest).
 *     This is exact to second order and automatically departs outward at
 *     periastron (V' > 0) and inward at apoastron (V' < 0).
 *
 * t and phi have no turning points, so they advance over the full remaining
 * step with their current (smooth, nonzero) rate.
 *
 * @param {CarterState} s mutated in place
 * @param {number} a
 * @param {number} h requested affine step
 * @returns {{turnedR:number, turnedT:number}} turning points crossed
 */
export function carterStep(s, a, h) {
  let turnedR = 0;
  let turnedT = 0;
  let remaining = h;
  const eps = 1e-6;

  // A photon has |dr/dlambda| ~ E far from the hole, so comparing the local
  // velocity against E gives a dimensionless "are we stuck" test.
  const vFloor = 1e-7 * Math.max(1e-30, Math.abs(s.energy));
  const slopeFloor = 1e-9 * Math.max(1, s.energy * s.energy);
  const tTol = 1e-9 * Math.max(1, a * a * s.energy * s.energy, s.lz * s.lz, Math.abs(s.q));

  const sigmaAt = (r, th) => r * r + a * a * Math.cos(th) * Math.cos(th);
  const vRad = (r) => radialPotential(r, a, s.energy, s.lz, s.q) / sigmaAt(r, s.theta) ** 2;
  const vPol = (th) => polarPotential(th, a, s.energy, s.lz, s.q) / sigmaAt(s.r, th) ** 2;

  for (let guard = 0; guard < 16 && remaining > 1e-12; guard++) {
    const vr = Math.sqrt(Math.max(0, vRad(s.r)));
    const vt = Math.sqrt(Math.max(0, vPol(s.theta)));
    const slopeR = (vRad(s.r + eps) - vRad(s.r - eps)) / (2 * eps);
    const slopeT = (vPol(s.theta + eps) - vPol(s.theta - eps)) / (2 * eps);

    // A coordinate is at a *genuine* turning point only when its velocity
    // vanishes AND the potential has nonzero slope there. A ray confined to
    // the equatorial plane has Theta = -Lz^2 cos^2/sin^2 <= 0, which vanishes
    // at theta = pi/2 with zero slope: that coordinate should simply stay put
    // while the other one keeps moving, not be treated as stalled.
    const turnR = vr <= vFloor && Math.abs(slopeR) > slopeFloor;
    const turnT = vt <= vFloor && Math.abs(slopeT) > slopeFloor;

    if (turnR || turnT) {
      const d = carterDerivative(s, a);
      s.t += d.dt * remaining;
      s.phi += d.dphi * remaining;

      if (turnR) {
        s.r += 0.25 * slopeR * remaining * remaining;
        s.signR = slopeR > 0 ? 1 : -1;
        turnedR++;
      }
      if (turnT) {
        s.theta += 0.25 * slopeT * remaining * remaining;
        s.signT = slopeT > 0 ? 1 : -1;
        turnedT++;
      }
      remaining = 0;
      continue;
    }

    const d = carterDerivative(s, a);
    const rTrial = s.r + d.dr * remaining;
    const thTrial = s.theta + d.dtheta * remaining;

    const allowed =
      radialPotential(rTrial, a, s.energy, s.lz, s.q) >= -tTol &&
      polarPotential(thTrial, a, s.energy, s.lz, s.q) >= -tTol;

    if (allowed) {
      s.r = rTrial;
      s.theta = thTrial;
      s.t += d.dt * remaining;
      s.phi += d.dphi * remaining;
      remaining = 0;
      break;
    }

    // Bisect for the largest fraction of the step that stays allowed, advance
    // to the turning point, and loop so the next iteration departs from it.
    let lo = 0;
    let hi = remaining;
    for (let i = 0; i < 40; i++) {
      const mid = 0.5 * (lo + hi);
      const ok =
        radialPotential(s.r + d.dr * mid, a, s.energy, s.lz, s.q) >= -tTol &&
        polarPotential(s.theta + d.dtheta * mid, a, s.energy, s.lz, s.q) >= -tTol;
      if (ok) lo = mid;
      else hi = mid;
    }
    s.r += d.dr * lo;
    s.theta += d.dtheta * lo;
    s.t += d.dt * lo;
    s.phi += d.dphi * lo;
    remaining -= lo;
  }

  return { turnedR, turnedT };
}

/**
 * Integrate with the separated Carter equations, recording equatorial
 * crossings. Mirrors the shader's inner loop so CPU/GPU agreement can be
 * asserted in tests.
 *
 * @param {Object} opts
 * @param {CarterState} opts.state
 * @param {number} opts.a
 * @param {number} [opts.step=0.05]
 * @param {number} [opts.maxSteps=200000]
 * @param {number} [opts.escapeRadius=60]
 * @param {number} [opts.minRadius]
 * @param {(r:number, theta:number, phi:number)=>boolean} [opts.stopWhen]
 * @param {(state:CarterState)=>void} [opts.onState] called after each step
 */
export function integrateCarter(opts) {
  const {
    state,
    a,
    step = 0.05,
    maxSteps = 200000,
    escapeRadius = 60,
    minRadius,
    stopWhen,
    onState,
  } = opts;

  const s = { ...state };
  const rMin = minRadius ?? horizonRadius(a) + 1e-3;
  const crossings = [];
  let rPeri = s.r;

  for (let i = 0; i < maxSteps; i++) {
    if (s.r <= rMin) {
      return { status: "captured", state: s, steps: i, crossings, rPeri };
    }

    const prev = { r: s.r, theta: s.theta, phi: s.phi };
    const h = step * Math.min(1, Math.max(0.1, (s.r - rMin) / 4));
    carterStep(s, a, h);
    rPeri = Math.min(rPeri, s.r);

    const cross = equatorCrossing(
      [0, prev.r, prev.theta, prev.phi],
      [0, s.r, s.theta, s.phi],
    );
    if (cross) crossings.push(cross);

    if (onState) onState(s);

    if (stopWhen && stopWhen(s.r, s.theta, s.phi)) {
      return { status: "hit", state: s, steps: i, crossings, rPeri };
    }
    if (s.r >= escapeRadius) {
      return { status: "escaped", state: s, steps: i, crossings, rPeri };
    }
  }
  return { status: "budget", state: s, steps: maxSteps, crossings, rPeri };
}

/* ------------------------------------------------------------------ *
 * Ray initialisation: screen direction -> covariant momentum
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} TetradMomentum
 * @property {number} pt
 * @property {number} pr
 * @property {number} pth
 * @property {number} pphi
 * @property {number} alpha lapse
 * @property {number} betaPhi shift (phi component)
 */

/**
 * Build a covariant photon momentum at (r, theta, phi) for a ray travelling
 * along a unit 3-direction measured in the locally non-rotating (ZAMO)
 * orthonormal frame.
 *
 * The ZAMO tetrad comes from the 3+1 split of the metric:
 *
 *   alpha     = 1 / sqrt(-g^{tt})          lapse
 *   beta^phi  = g_{tphi} / g_{phiphi}      shift (from g_{ti} = gamma_ij beta^j)
 *   gamma_rr = g_rr,  gamma_th = g_th,  gamma_ph = g_phiphi
 *
 *   p = E ( e_hat_t + n^(1) e_hat_r + n^(2) e_hat_th + n^(3) e_hat_ph )
 *
 * which is null by construction when |n| = 1. The covariant components follow
 * from p_mu = g_{mu nu} p^nu, so no sign convention is hard-coded.
 *
 * @param {number} r
 * @param {number} theta
 * @param {number} a
 * @param {{n1:number,n2:number,n3:number}} dir unit 3-vector in the ZAMO frame
 * @param {number} [energy=1]
 * @returns {TetradMomentum}
 */
export function momentumFromTetradDirection(r, theta, a, dir, energy = 1) {
  const ginv = inverseMetric(r, theta, a);
  const g = metric(r, theta, a);

  // Lapse from g^{tt} = -1/alpha^2; shift from g_{ti} = gamma_ij beta^j, so
  // beta^phi = g_{tphi}/g_{phiphi} = -omega (minus the frame-dragging rate).
  const alpha = 1 / Math.sqrt(Math.max(1e-12, -ginv[idx(T, T)]));
  const betaPhi = g[idx(T, PH)] / g[idx(PH, PH)];

  // Contravariant components: p = E (e_hat_t + n^i e_hat_i).
  const pUpT = energy / alpha;
  const pUpPh = -betaPhi * pUpT + (energy * dir.n3) / Math.sqrt(g[idx(PH, PH)]);
  const pUpR = (energy * dir.n1) / Math.sqrt(g[idx(R, R)]);
  const pUpTh = (energy * dir.n2) / Math.sqrt(g[idx(TH, TH)]);

  // Lower the index with the covariant metric.
  const pt = g[idx(T, T)] * pUpT + g[idx(T, PH)] * pUpPh;
  const pphi = g[idx(PH, T)] * pUpT + g[idx(PH, PH)] * pUpPh;
  const pr = g[idx(R, R)] * pUpR;
  const pth = g[idx(TH, TH)] * pUpTh;

  return { pt, pr, pth, pphi, alpha, betaPhi };
}

/**
 * Recover the covariant momentum of a Carter state from its constants of
 * motion. p_t = -E and p_phi = Lz are exact. For the remaining two, note
 *
 *   p^r     = dr/dlambda = signR sqrt(R) / Sigma,  g_rr = Sigma / Delta
 *   => p_r  = g_rr p^r = signR sqrt(R) / Delta
 *   p^th    = dtheta/dlambda = signT sqrt(Theta) / Sigma,  g_thth = Sigma
 *   => p_th = signT sqrt(Theta)
 *
 * @param {CarterState} s
 * @param {number} a
 * @returns {{pt:number, pr:number, pth:number, pphi:number}}
 */
export function carterMomentum(s, a) {
  const { r, theta, energy, lz, q, signR, signT } = s;
  const delta = r * r - 2.0 * r + a * a;

  const pr = (signR * Math.sqrt(Math.max(0, radialPotential(r, a, energy, lz, q)))) / delta;
  const pth = signT * Math.sqrt(Math.max(0, polarPotential(theta, a, energy, lz, q)));
  return { pt: -energy, pr, pth, pphi: lz };
}
