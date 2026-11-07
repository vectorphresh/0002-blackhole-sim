/**
 * Relativistic emission from a thin accretion disk.
 *
 * This is the physics behind `diskEmission` in the trace shader, kept here so
 * it can be tested. The shader is a port of this module; if the two disagree,
 * the shader is wrong.
 *
 * Geometry: the disk is a thin equatorial slab whose material follows circular
 * geodesics. A photon crossing the plane at radius r with axial angular
 * momentum Lz was, at that crossing, emitted by material moving with angular
 * velocity Omega.
 */

/**
 * Coordinate angular velocity of a circular equatorial geodesic.
 * @param {number} r
 * @param {number} a
 * @param {number} direction +1 prograde, -1 retrograde
 */
export function orbitalOmega(r, a, direction = 1) {
  const den = r * Math.sqrt(r) + direction * a;
  if (den === 0) return NaN;
  return direction / den;
}

/**
 * Total redshift factor g = E_observed / E_emitted for a photon with impact
 * parameter xi = Lz/E crossing the disk at radius r.
 *
 *   E_em  = -u^mu p_mu = u^t (E - Omega Lz)
 *   E_obs = E                       (the observer at infinity is static)
 *   g     = E_obs / E_em = 1 / [ u^t (1 - Omega xi) ]
 *
 * with u^t = (-g_tt - 2 g_tphi Omega - g_phiphi Omega^2)^(-1/2).
 *
 * The reciprocal is the whole point and is easy to get backwards. Writing
 * g = u^t (1 - Omega xi) reports light climbing out of the potential well as
 * *blueshifted*: at Omega = 0 it gives u^t = 1/sqrt(1-2/r) > 1, whereas the
 * correct Schwarzschild gravitational redshift is sqrt(1-2/r) < 1.
 *
 * Returns null where no circular orbit exists (inside the light surface) or
 * where the photon would have to be emitted against the flow.
 *
 * @param {number} r crossing radius
 * @param {number} a spin
 * @param {number} xi impact parameter Lz/E of the photon
 * @param {number} [direction=1] +1 prograde, -1 retrograde
 * @returns {number|null}
 */
export function redshiftFactor(r, a, xi, direction = 1) {
  const omega = orbitalOmega(r, a, direction);
  if (!Number.isFinite(omega)) return null;

  // Equatorial metric components: Sigma = r^2 at theta = pi/2.
  const gtt = -1.0 + 2.0 / r;
  const gtphi = -2.0 * a / r;
  const gpp = r * r + a * a + (2.0 * a * a) / r;

  const norm = gtt + 2.0 * gtphi * omega + gpp * omega * omega;
  if (norm >= 0) return null; // not a timelike orbit: inside the light surface
  const ut = 1.0 / Math.sqrt(-norm);

  const denom = ut * (1.0 - omega * xi);
  if (denom <= 1e-6) return null; // infinitely redshifted
  return 1.0 / denom;
}

/**
 * Shakura–Sunyaev effective temperature profile.
 *
 *   T_eff^4 ~ (1 - sqrt(r_in / r)) / r^3
 *
 * The flux factor vanishes at the inner edge, which is where the disk stops
 * being thin and the geodesic assumption breaks down.
 *
 * With x = r/rInner the profile is T^4 = x^-3 - x^-15/4, so the peak is at
 * x = (5/4)^(4/3) = 1.3465216... where T = (4/5) * 5^(-1/4) = 0.5349922...
 * It is NOT normalised to a peak of 1; the caller's display gain absorbs the
 * scale. See scripts/validate.js section 7, which pins both numbers.
 *
 * @param {number} r
 * @param {number} rInner inner disk edge (typically the ISCO)
 */
export function diskTemperature(r, rInner) {
  if (r < rInner) return 0;
  const x = r / rInner;
  const f = Math.max(0, 1 - Math.pow(rInner / r, 0.75));
  return Math.pow(x, -0.75) * Math.pow(f, 0.25);
}

/**
 * Observed colour temperature: the emitted temperature multiplied by the
 * redshift factor. Blueshifted approaching material looks hotter, redshifted
 * receding material cooler.
 */
export function observedTemperature(r, a, xi, rInner, direction = 1) {
  const g = redshiftFactor(r, a, xi, direction);
  if (g === null) return null;
  return g * diskTemperature(r, rInner);
}
