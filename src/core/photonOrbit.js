/**
 * Photon orbit and shadow-size calculations.
 *
 * These are derived numerically from the separated radial potential R(r)
 * rather than quoted from a formula, so a sign error in the potential shows up
 * as a failed test instead of a subtly wrong shadow.
 *
 * For an equatorial photon (theta = pi/2, Q = 0) with E = 1 and axial angular
 * momentum xi = Lz:
 *
 *   R(r) = P^2 - Delta (xi - a)^2,   P = r^2 + a^2 - a xi
 *
 * An unstable circular photon orbit satisfies R = 0 and dR/dr = 0. Eliminating
 * xi gives the orbit radius; the critical impact parameter is then b = xi.
 *
 * Known limits, asserted in tests/photonOrbit.test.js:
 *   a = 0        : r_ph = 3M,        b_crit = 3 sqrt(3) M = 5.19615
 *   a -> 1       : prograde r_ph -> M,  retrograde r_ph -> 4M
 */

import { findRoot, horizonRadius } from "./kerr.js";
import { radialPotential } from "./geodesics.js";

/**
 * R(r) for an equatorial photon with E = 1, expressed via the Carter potential.
 * @param {number} r
 * @param {number} a
 * @param {number} xi
 */
export function equatorialR(r, a, xi) {
  return radialPotential(r, a, 1, xi, 0);
}

/**
 * dR/dr by central difference.
 */
export function equatorialRPrime(r, a, xi, h = 1e-6) {
  return (equatorialR(r + h, a, xi) - equatorialR(r - h, a, xi)) / (2 * h);
}

/**
 * For a given orbit radius r, solve R(r) = 0 for the impact parameter xi.
 * R is quadratic in xi, so this has a closed form; we take the root that
 * corresponds to the requested direction of motion.
 *
 *   (r^2 + a^2 - a xi)^2 = Delta (xi - a)^2
 *   r^2 + a^2 - a xi = s sqrt(Delta) (xi - a)
 *   =>  xi = (r^2 + a^2 + s a sqrt(Delta)) / (a + s sqrt(Delta))
 *
 * Two sign traps live here, and neither is caught by the a = 0 test:
 *
 *  1. `prograde` is a boolean. Writing `prograde * sq` evaluates to 0 for
 *     retrograde, which collapses the denominator to `a` and returns NaN for
 *     every radius when a = 0.
 *
 *  2. The branch choice. s = +1 gives xi > 0, which for a > 0 is the
 *     *corotating* (prograde) orbit sitting inside 3M; s = -1 gives xi < 0,
 *     the counter-rotating orbit outside 3M. Getting this backwards silently
 *     swaps prograde and retrograde for every a > 0 while leaving a = 0
 *     perfectly correct, because the two branches are mirror images there.
 *
 * @param {number} r
 * @param {number} a
 * @param {boolean} prograde
 */
export function xiAtRadius(r, a, prograde = true) {
  const s = prograde ? 1 : -1;
  const delta = r * r - 2.0 * r + a * a;
  if (delta <= 0) return NaN;
  const sq = Math.sqrt(delta);
  const den = a + s * sq;
  if (Math.abs(den) < 1e-14) return NaN;
  return (r * r + a * a + s * a * sq) / den;
}

/**
 * Solve R = dR/dr = 0 for the equatorial photon orbit radius by root-finding
 * on the function xi(r) from R = 0: a circular orbit is a stationary point of
 * xi(r), i.e. d(xi)/dr = 0.
 *
 * @param {number} a
 * @param {boolean} prograde
 * @returns {{radius:number, impactParameter:number}}
 */
export function equatorialPhotonOrbit(a, prograde = true) {
  const rH = horizonRadius(a);
  // Bracket the stationary point of xi(r) outside the horizon.
  const lo = rH + 1e-4;
  const hi = 12.0;
  const dxi = (r) => (xiAtRadius(r + 1e-5, a, prograde) - xiAtRadius(r - 1e-5, a, prograde)) / 2e-5;

  // Scan for a sign change in dxi, then refine.
  let prevR = lo;
  let prevD = dxi(prevR);
  let bracket = null;
  const n = 400;
  for (let i = 1; i <= n; i++) {
    const r = lo + ((hi - lo) * i) / n;
    const d = dxi(r);
    if (Number.isFinite(prevD) && Number.isFinite(d) && prevD * d < 0) {
      bracket = [prevR, r];
      break;
    }
    prevR = r;
    prevD = d;
  }
  if (!bracket) {
    throw new Error(`equatorialPhotonOrbit: no stationary point found (a=${a}, prograde=${prograde})`);
  }
  const radius = findRoot(dxi, bracket[0], bracket[1], { tol: 1e-11 });
  return { radius, impactParameter: xiAtRadius(radius, a, prograde) };
}

/**
 * Critical impact parameter for a photon aimed at the hole: the boundary
 * between capture and escape, found by bisection on the actual geodesic
 * integrator. This is the physically meaningful "shadow radius" and is
 * independent of the circular-orbit algebra above.
 *
 * @param {Object} opts
 * @param {number} opts.a
 * @param {boolean} [opts.prograde=true]
 * @param {number} [opts.startRadius=200]
 * @param {number} [opts.tol=1e-6]
 * @param {(b:number)=>boolean} opts.captures
 */
export function criticalImpactParameter(opts) {
  const { a, prograde = true, tol = 1e-6, captures } = opts;
  // Seed bracket from the circular-orbit estimate, then bisect on capture.
  const est = equatorialPhotonOrbit(a, prograde).impactParameter;
  let lo = Math.abs(est) * 0.5;
  let hi = Math.abs(est) * 1.5;
  if (!captures(lo)) {
    // lo escapes: widen downward until captured.
    let found = false;
    for (let i = 0; i < 40 && !found; i++) {
      lo *= 0.7;
      found = captures(lo);
    }
    if (!found) throw new Error("criticalImpactParameter: could not bracket a captured ray");
  }
  if (captures(hi)) {
    let found = false;
    for (let i = 0; i < 40 && !found; i++) {
      hi *= 1.3;
      found = !captures(hi);
    }
    if (!found) throw new Error("criticalImpactParameter: could not bracket an escaping ray");
  }
  return findRoot((b) => (captures(b) ? 1 : -1), lo, hi, { tol, maxIter: 200 });
}
