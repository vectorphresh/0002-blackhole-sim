/**
 * Mino-time geodesic integration — the algorithm mirrored by the GLSL shader.
 *
 * Carter's separated equations are written with the affine parameter lambda,
 * but the affine parameter couples r and theta through the common factor
 * Sigma = r^2 + a^2 cos^2(theta). Introducing Mino's parameter
 *
 *     dlambda = Sigma dlambda_M
 *
 * removes it and the equations decouple completely:
 *
 *     dr/dlambda_M   = s_r sqrt(R(r))          depends only on r
 *     dtheta/dlambda_M = s_t sqrt(Theta(theta)) depends only on theta
 *     dt/dlambda_M   = (A E - 2 a r Lz) / Delta                (r only)
 *     dphi/dlambda_M = 2 a r E / Delta + Lz (1/sin^2 - a^2/Delta)  (r and theta)
 *
 * Because r and theta are now independent one-dimensional motions in
 * potentials, each obeys
 *
 *     d2r/dlambda_M2 = (1/2) dR/dr
 *
 * (differentiate v = s_r sqrt(R): dv/dlambda_M = s_r R' v / (2 s_r sqrt(R)) = R'/2,
 * independent of the sign s_r). So velocity-Verlet applies directly, and it
 * handles turning points for free: at a turning point v = 0 but the
 * acceleration (1/2)R' is not, so the integrator accelerates the particle back
 * into the allowed region with the correct sign automatically. No sign
 * bookkeeping, no bisection, no stalling.
 *
 * Mino time is a pure reparametrisation of affine parameter, so the path
 * through spacetime is identical to `integrateHamilton`'s; only the parameter
 * differs. That is what makes the two cross-checkable in tests.
 *
 * t and phi are quadratures along the path and are accumulated with the
 * trapezoid rule using the endpoint rates.
 */

import { radialPotential, polarPotential } from "./geodesics.js";
import { horizonRadius } from "./kerr.js";

const SIN2_FLOOR = 1e-12;
const TINY = 1e-300;
/** Smallest Mino-time step, so the controller can never stall the ray. */
export const H_MIN = 1e-9;
/** Largest Mino-time step, used where the potentials are flat and the
 *  adaptive rule would otherwise allow an arbitrarily large step. */
export const H_MAX = 0.25;

/**
 * (1/2) dR/dr — the radial acceleration in Mino time, in closed form.
 *
 *   R   = P^2 - Delta K,   P = E(r^2 + a^2) - a Lz,   K = (Lz - aE)^2 + Q
 *   dR/dr = 2 P P' - Delta' K,   P' = 2Er,   Delta' = 2r - 2
 *
 * Exact rather than finite-differenced: cheaper in the shader and free of the
 * cancellation that plagues a difference of two ~E^2 r^4 quantities.
 */
export function radialAccel(r, a, energy, lz, q) {
  const r2 = r * r;
  const a2 = a * a;
  const p = energy * (r2 + a2) - a * lz;
  const k = (lz - a * energy) * (lz - a * energy) + q;
  return 2.0 * energy * r * p - (r - 1.0) * k;
}

/**
 * (1/2) dTheta/dtheta — the polar acceleration in Mino time, in closed form.
 *
 *   Theta = Q + a^2 E^2 cos^2 - Lz^2 cot^2
 *   dTheta/dtheta = -2 a^2 E^2 cos sin + 2 Lz^2 cos / sin^3
 */
export function polarAccel(theta, a, energy, lz) {
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  const sin2 = Math.max(sinT * sinT, SIN2_FLOOR);
  return -a * a * energy * energy * cosT * sinT + (lz * lz * cosT) / (sin2 * sinT);
}

/** dt/dlambda_M. */
export function dtDlambdaM(r, theta, a, energy, lz) {
  const r2 = r * r;
  const a2 = a * a;
  const delta = r2 - 2.0 * r + a2;
  const sin2 = Math.max(Math.sin(theta) ** 2, SIN2_FLOOR);
  const A = (r2 + a2) * (r2 + a2) - a2 * delta * sin2;
  return (A * energy - 2.0 * a * r * lz) / delta;
}

/** dphi/dlambda_M. */
export function dphiDlambdaM(r, theta, a, energy, lz) {
  const r2 = r * r;
  const a2 = a * a;
  const delta = r2 - 2.0 * r + a2;
  const sin2 = Math.max(Math.sin(theta) ** 2, SIN2_FLOOR);
  return (2.0 * a * r * energy) / delta + lz * (1.0 / sin2 - a2 / delta);
}

/**
 * Adaptive Mino-time step for the current state.
 *
 * A *fixed* Mino step is unusable. In Mino time the radial speed is
 * `dr/dlambda_M = sqrt(R) ~ E r^2`, which grows without bound, so a step that
 * resolves the near-horizon region leaps thousands of units per iteration out
 * at the launch radius — the ray sails straight past the hole. The step must
 * therefore scale with the local dynamics.
 *
 * Four limits, each expressed so that halving `step` halves `h` everywhere
 * (which is what preserves the second-order convergence of Verlet):
 *
 *   r / |v_r|      relative radial change per step is bounded
 *   1 / |v_t|      polar change per step is bounded (theta is O(1))
 *   sqrt(r / |a_r|) the acceleration term stays subdominant, so a step can
 *                    never jump across a turning point
 *   sqrt(1 / |a_t|) same for the polar turning points
 *
 * At a turning point `v -> 0`, so the velocity limits relax and the
 * acceleration limits take over — which is exactly the regime where the
 * integrator needs to be careful.
 *
 * @param {MinoState} s
 * @param {number} a
 * @param {number} step tolerance multiplier; smaller = more accurate
 * @returns {number} the Mino-time step
 */
export function adaptiveStep(s, a, step) {
  const { r, theta, energy, lz, q } = s;
  const vr = Math.sqrt(Math.max(0, radialPotential(r, a, energy, lz, q)));
  const vt = Math.sqrt(Math.max(0, polarPotential(theta, a, energy, lz, q)));
  const ar = Math.abs(radialAccel(r, a, energy, lz, q));
  const at = Math.abs(polarAccel(theta, a, energy, lz));
  const h = Math.min(
    r / Math.max(vr, TINY),
    1.0 / Math.max(vt, TINY),
    Math.sqrt(r / Math.max(ar, TINY)),
    Math.sqrt(1.0 / Math.max(at, TINY)),
  );
  return Math.min(H_MAX, Math.max(H_MIN, step * h));
}

/**
 * One velocity-Verlet step in Mino time.
 *
 * @param {MinoState} s mutated in place
 * @param {number} a spin
 * @param {number} h Mino-time step
 */
export function verletStep(s, a, h) {
  const { r, theta, energy, lz, q } = s;

  const R = radialPotential(r, a, energy, lz, q);
  const TH = polarPotential(theta, a, energy, lz, q);
  const vr = Math.sign(s.signR) * Math.sqrt(Math.max(0, R));
  const vt = Math.sign(s.signT) * Math.sqrt(Math.max(0, TH));
  const ar = radialAccel(r, a, energy, lz, q);
  const at = polarAccel(theta, a, energy, lz);

  const rNew = r + vr * h + 0.5 * ar * h * h;
  const thNew = theta + vt * h + 0.5 * at * h * h;

  const aRnew = radialAccel(rNew, a, energy, lz, q);
  const aTnew = polarAccel(thNew, a, energy, lz);
  const vrNew = vr + 0.5 * (ar + aRnew) * h;
  const vtNew = vt + 0.5 * (at + aTnew) * h;

  // Quadratures for t and phi: trapezoid over the endpoint rates.
  const t0 = dtDlambdaM(r, theta, a, energy, lz);
  const p0 = dphiDlambdaM(r, theta, a, energy, lz);
  const t1 = dtDlambdaM(rNew, thNew, a, energy, lz);
  const p1 = dphiDlambdaM(rNew, thNew, a, energy, lz);

  s.t += 0.5 * (t0 + t1) * h;
  s.phi += 0.5 * (p0 + p1) * h;
  s.r = rNew;
  s.theta = thNew;

  // The Verlet velocity carries the direction; keep the sign fields in sync so
  // a caller can reconstruct the momentum.
  s.signR = vrNew >= 0 ? 1 : -1;
  s.signT = vtNew >= 0 ? 1 : -1;
  s.vr = vrNew;
  s.vt = vtNew;
}

/**
 * Integrate with velocity-Verlet in Mino time, recording equatorial crossings.
 *
 * @param {Object} opts
 * @param {MinoState} opts.state
 * @param {number} opts.a
 * @param {number} [opts.step=0.02] step tolerance (see `adaptiveStep`)
 * @param {number} [opts.maxSteps=200000]
 * @param {number} [opts.escapeRadius=60]
 * @param {number} [opts.minRadius]
 * @param {(r:number, theta:number, phi:number)=>boolean} [opts.stopWhen]
 * @param {(state:MinoState)=>void} [opts.onState]
 */
export function integrateVerlet(opts) {
  const {
    state,
    a,
    step = 0.02,
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
    if (s.r <= rMin) return { status: "captured", state: s, steps: i, crossings, rPeri };

    const prev = { r: s.r, theta: s.theta, phi: s.phi };
    verletStep(s, a, adaptiveStep(s, a, step));
    rPeri = Math.min(rPeri, s.r);

    if ((prev.theta - Math.PI / 2) * (s.theta - Math.PI / 2) < 0) {
      const w = (Math.PI / 2 - prev.theta) / (s.theta - prev.theta);
      crossings.push({
        r: prev.r + w * (s.r - prev.r),
        phi: prev.phi + w * (s.phi - prev.phi),
        dir: s.theta > prev.theta ? 1 : -1,
      });
    }

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
