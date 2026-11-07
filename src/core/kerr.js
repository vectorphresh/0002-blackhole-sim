/**
 * Kerr spacetime in Boyer–Lindquist coordinates (t, r, theta, phi).
 *
 * Signature (-, +, +, +), geometrised units G = c = M = 1.
 *
 *   Sigma = r^2 + a^2 cos^2(theta)
 *   Delta = r^2 - 2 r + a^2
 *
 *   ds^2 = -(1 - 2r/Sigma) dt^2 - (2 a r sin^2(theta)/Sigma) dt dphi
 *          + (Sigma/Delta) dr^2 + Sigma dtheta^2
 *          + ((r^2 + a^2)^2 - a^2 Delta sin^2(theta)) sin^2(theta)/Sigma dphi^2
 *
 * Every quantity here is derived from the metric or its inverse; nothing is
 * hard-coded from a table of results. Where a closed form is used (ISCO,
 * photon orbit radii) the derivation is cited and the result is asserted by
 * the numerical test-suite in tests/.
 */

import { TWO_PI } from "./constants.js";

/** Coordinate indices used throughout: x^0=t, x^1=r, x^2=theta, x^3=phi. */
export const T = 0;
export const R = 1;
export const TH = 2;
export const PH = 3;

/**
 * @typedef {Object} KerrFrame
 * @property {number} sigma  Sigma = r^2 + a^2 cos^2(theta)
 * @property {number} delta  Delta = r^2 - 2r + a^2
 * @property {number} sinT
 * @property {number} cosT
 * @property {number} sin2   sin^2(theta)
 * @property {number} cos2   cos^2(theta)
 */

/**
 * Precompute the recurring metric building blocks at a point.
 * @param {number} r
 * @param {number} theta
 * @param {number} a  dimensionless spin a/M, 0 <= a < 1
 * @returns {KerrFrame}
 */
export function frame(r, theta, a) {
  const sinT = Math.sin(theta);
  const cosT = Math.cos(theta);
  const sin2 = sinT * sinT;
  const cos2 = cosT * cosT;
  const r2 = r * r;
  const a2 = a * a;
  return {
    sigma: r2 + a2 * cos2,
    delta: r2 - 2.0 * r + a2,
    sinT,
    cosT,
    sin2,
    cos2,
    r2,
    a2,
  };
}

/**
 * Covariant metric components g_{mu nu} as a flat 4x4 row-major array.
 * @param {number} r
 * @param {number} theta
 * @param {number} a
 * @param {Float64Array} [out]
 * @returns {Float64Array}
 */
export function metric(r, theta, a, out = new Float64Array(16)) {
  const f = frame(r, theta, a);
  const { sigma, delta, sin2, r2, a2 } = f;
  const A = (r2 + a2) * (r2 + a2) - a2 * delta * sin2;

  out.fill(0);
  out[idx(T, T)] = -(1.0 - 2.0 * r / sigma);
  out[idx(T, PH)] = -(2.0 * a * r * sin2) / sigma;
  out[idx(PH, T)] = out[idx(T, PH)];
  out[idx(R, R)] = sigma / delta;
  out[idx(TH, TH)] = sigma;
  out[idx(PH, PH)] = (A * sin2) / sigma;
  return out;
}

/**
 * Contravariant metric components g^{mu nu} (flat 4x4, row-major).
 *
 *   g^{tt}  = -A / (Sigma Delta)
 *   g^{tphi}= -2 a r / (Sigma Delta)
 *   g^{rr}  =  Delta / Sigma
 *   g^{thth}=  1 / Sigma
 *   g^{phph}= (Delta - a^2 sin^2) / (Sigma Delta sin^2)
 *
 * @param {number} r
 * @param {number} theta
 * @param {number} a
 * @param {Float64Array} [out]
 * @returns {Float64Array}
 */
export function inverseMetric(r, theta, a, out = new Float64Array(16)) {
  const f = frame(r, theta, a);
  const { sigma, delta, sin2, r2, a2 } = f;
  const A = (r2 + a2) * (r2 + a2) - a2 * delta * sin2;
  const sd = sigma * delta;

  out.fill(0);
  out[idx(T, T)] = -A / sd;
  out[idx(T, PH)] = -(2.0 * a * r) / sd;
  out[idx(PH, T)] = out[idx(T, PH)];
  out[idx(R, R)] = delta / sigma;
  out[idx(TH, TH)] = 1.0 / sigma;
  out[idx(PH, PH)] = (delta - a2 * sin2) / (sd * sin2);
  return out;
}

/** Flat-array index for a 4x4 matrix. */
export function idx(i, j) {
  return i * 4 + j;
}

/**
 * Raise an index: p^mu = g^{mu nu} p_nu.
 * @param {Float64Array} ginv flat 4x4
 * @param {ArrayLike<number>} pcov covariant momentum
 * @param {Float64Array} [out]
 */
export function raise(ginv, pcov, out = new Float64Array(4)) {
  for (let i = 0; i < 4; i++) {
    out[i] =
      ginv[idx(i, 0)] * pcov[0] +
      ginv[idx(i, 1)] * pcov[1] +
      ginv[idx(i, 2)] * pcov[2] +
      ginv[idx(i, 3)] * pcov[3];
  }
  return out;
}

/**
 * Contract two covectors with the inverse metric: g^{mu nu} p_mu q_nu.
 * @param {Float64Array} ginv
 * @param {ArrayLike<number>} p
 * @param {ArrayLike<number>} q
 */
export function contract(ginv, p, q) {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      s += ginv[idx(i, j)] * p[i] * q[j];
    }
  }
  return s;
}

/**
 * Outer horizon radius r_+ = M + sqrt(M^2 - a^2) (M = 1).
 * @param {number} a
 */
export function horizonRadius(a) {
  return 1.0 + Math.sqrt(Math.max(0.0, 1.0 - a * a));
}

/**
 * Inner horizon radius r_- = M - sqrt(M^2 - a^2).
 * @param {number} a
 */
export function innerHorizonRadius(a) {
  return 1.0 - Math.sqrt(Math.max(0.0, 1.0 - a * a));
}

/**
 * Ergosphere radius at polar angle theta: r = M + sqrt(M^2 - a^2 cos^2 theta).
 * At the equator this is 2M; on the axis it coincides with the horizon.
 * @param {number} a
 * @param {number} [theta=HALF_PI]
 */
export function ergosphereRadius(a, theta = Math.PI / 2) {
  const c = Math.cos(theta);
  return 1.0 + Math.sqrt(Math.max(0.0, 1.0 - a * a * c * c));
}

/**
 * Frame-dragging angular velocity of zero-angular-momentum observers,
 * omega = -g_{t phi} / g_{phi phi}. This is the rate at which spacetime itself
 * is dragged around the hole and is the natural reference frame for the disk.
 * @param {number} r
 * @param {number} theta
 * @param {number} a
 */
export function frameDragging(r, theta, a) {
  const g = metric(r, theta, a);
  return -g[idx(T, PH)] / g[idx(PH, PH)];
}

/**
 * Coordinate angular velocity of a circular equatorial geodesic:
 *
 *   Omega = +/- 1 / (r^{3/2} +/- a)
 *
 * upper sign prograde. Follows from factoring the radial Euler-Lagrange
 * stationarity condition (r^3 - a^2) Omega^2 + 2 a Omega - 1 = 0, using
 * r^3 - a^2 = (r^{3/2} - a)(r^{3/2} + a).
 *
 * Returns positive for prograde and negative for retrograde. Note the
 * denominator uses `a`, not `sqrt(a)` — an easy mistake that the a = 0
 * (Keplerian) limit cannot catch.
 *
 * @param {number} r
 * @param {number} a
 * @param {boolean} [prograde=true]
 */
export function circularOmega(r, a, prograde = true) {
  const den = prograde ? r * Math.sqrt(r) + a : r * Math.sqrt(r) - a;
  if (den === 0) return NaN;
  return (prograde ? 1.0 : -1.0) / den;
}

/**
 * Normalisation u^t of a circular equatorial geodesic with angular velocity
 * Omega, from the line element with dr = dtheta = 0:
 *
 *   -1 = g_tt (u^t)^2 + 2 g_tphi u^t u^phi + g_phiphi (u^phi)^2
 *      = (u^t)^2 (g_tt + 2 g_tphi Omega + g_phiphi Omega^2)
 *
 * @param {number} r
 * @param {number} a
 * @param {number} Omega
 */
export function circularU_t(r, a, Omega) {
  const g = metric(r, Math.PI / 2, a);
  const norm =
    g[idx(T, T)] + 2.0 * g[idx(T, PH)] * Omega + g[idx(PH, PH)] * Omega * Omega;
  if (norm >= 0) return NaN; // not a timelike orbit (inside ergosphere / horizon)
  return 1.0 / Math.sqrt(-norm);
}

/**
 * Specific energy E and axial angular momentum Lz of a circular equatorial
 * geodesic, derived from the metric rather than quoted.
 *
 * At the equator (Sigma = r^2, sin theta = 1):
 *
 *   g_tt   = -(1 - 2/r)
 *   g_tphi = -2a/r
 *   g_pphi = r^2 + a^2 + 2a^2/r
 *
 * The radial Euler-Lagrange condition dU/dr = 0 with
 * U = g_tt + 2 g_tphi Omega + g_phiphi Omega^2 becomes
 *
 *   (r^3 - a^2) Omega^2 + 2 a Omega - 1 = 0
 *
 * and since r^3 - a^2 = (r^{3/2} - a)(r^{3/2} + a), this factors to
 *
 *   Omega = 1 / (r^{3/2} +/- a)          (matches `circularOmega`)
 *
 * Then with u^t = 1/sqrt(-U):
 *
 *   E  = -u_t = -u^t (g_tt + g_tphi Omega)
 *   Lz =  u_phi =  u^t (g_tphi + g_phiphi Omega)
 *
 * Verified against a finite-difference derivation from the metric in
 * scripts/checkCircular.js and asserted in tests/kerr.test.js.
 *
 * @param {number} r
 * @param {number} a
 * @param {boolean} [prograde=true]
 */
export function circularEnergyAndAngularMomentum(r, a, prograde = true) {
  const omega = circularOmega(r, a, prograde);
  if (!Number.isFinite(omega)) return { energy: NaN, angularMomentum: NaN };

  const gtt = -(1.0 - 2.0 / r);
  const gtph = (-2.0 * a) / r;
  const gpp = r * r + a * a + (2.0 * a * a) / r;

  const U = gtt + 2.0 * gtph * omega + gpp * omega * omega;
  if (U >= 0) return { energy: NaN, angularMomentum: NaN }; // not timelike

  const ut = 1.0 / Math.sqrt(-U);
  return {
    energy: -ut * (gtt + gtph * omega),
    angularMomentum: ut * (gtph + gpp * omega),
    omega,
    u_t: ut,
  };
}

/**
 * Innermost stable circular orbit for a test particle (Bardeen, Press &
 * Teukolsky 1972):
 *
 *   Z1 = 1 + (1 - a^2)^{1/3} [ (1+a)^{1/3} + (1-a)^{1/3} ]
 *   Z2 = sqrt(3 a^2 + Z1^2)
 *   r_isco = 3 + Z2 -/+ sqrt( (3 - Z1)(3 + Z1 + 2 Z2) )
 *
 * minus for prograde, plus for retrograde. Limits (asserted in tests/):
 *   a = 0        : 6M both directions
 *   a = 1        : 1M prograde (coincident with the horizon), 9M retrograde
 *
 * Do not confuse the extremal prograde ISCO (1M) with the extremal prograde
 * photon orbit (1M) or the Schwarzschild photon orbit (3M).
 *
 * @param {number} a
 * @param {boolean} [prograde=true]
 */
export function iscoRadius(a, prograde = true) {
  const ac = Math.min(Math.max(a, 0), 1);
  const z1 =
    1.0 +
    Math.pow(1.0 - ac * ac, 1.0 / 3.0) *
      (Math.pow(1.0 + ac, 1.0 / 3.0) + Math.pow(1.0 - ac, 1.0 / 3.0));
  const z2 = Math.sqrt(3.0 * ac * ac + z1 * z1);
  const term = Math.sqrt(Math.max(0.0, (3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
  return prograde ? 3.0 + z2 - term : 3.0 + z2 + term;
}

/**
 * Marginally bound orbit radius (where the specific energy of a circular
 * orbit equals 1). a=0 -> 4M.
 * @param {number} a
 * @param {boolean} [prograde=true]
 */
export function marginallyBoundRadius(a, prograde = true) {
  const ac = Math.min(Math.max(a, 0), 1);
  const s = prograde ? 1.0 : -1.0;
  const sq = Math.sqrt(ac);
  return 2.0 - s * sq + 2.0 * Math.sqrt(Math.max(0.0, 1.0 - s * sq));
}

/**
 * Radius of the last marginally stable orbit used as the inner edge of the
 * disk. Falls back to the horizon if a numerical edge case pushes it inside.
 * @param {number} a
 * @param {boolean} prograde
 */
export function diskInnerRadius(a, prograde = true) {
  return Math.max(iscoRadius(a, prograde), horizonRadius(a) + 1e-6);
}

/**
 * Solve f(x) = 0 on [lo, hi] with a bracketing bisection refined by secant
 * steps. Deliberately simple and dependency-free: the physics tests only need
 * robust root finding on smooth, monotonic-ish functions.
 *
 * @param {(x:number)=>number} f
 * @param {number} lo
 * @param {number} hi
 * @param {Object} [opts]
 * @returns {number}
 */
export function findRoot(f, lo, hi, opts = {}) {
  const tol = opts.tol ?? 1e-12;
  const maxIter = opts.maxIter ?? 200;
  let flo = f(lo);
  let fhi = f(hi);
  if (flo === 0) return lo;
  if (fhi === 0) return hi;
  if (flo * fhi > 0) {
    throw new Error(
      `findRoot: no bracket on [${lo}, ${hi}] (f=${flo}, ${fhi})`,
    );
  }
  let x = lo;
  let fx = flo;
  for (let i = 0; i < maxIter; i++) {
    const xm = 0.5 * (lo + hi);
    const fm = f(xm);
    if (fm === 0 || (hi - lo) / 2 < tol) return xm;
    // Secant step from the bracketing pair when it stays inside the bracket.
    let xs = xm;
    const denom = fx - fm;
    if (denom !== 0) {
      const cand = x - fx * ((xm - x) / denom);
      if (cand > Math.min(lo, hi) && cand < Math.max(lo, hi)) xs = cand;
    }
    if (fm * fx < 0) {
      hi = xm;
      fhi = fm;
    } else {
      lo = xm;
      flo = fm;
      x = xm;
      fx = fm;
    }
    if (Math.abs(hi - lo) < tol) return 0.5 * (lo + hi);
    if (xs !== xm) {
      const fxs = f(xs);
      if (fxs * fx < 0) {
        hi = xs;
      } else {
        lo = xs;
        x = xs;
        fx = fxs;
      }
    }
  }
  return 0.5 * (lo + hi);
}

export { TWO_PI };
