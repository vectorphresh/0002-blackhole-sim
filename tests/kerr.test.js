/**
 * Metric algebra and Kerr geometry limits.
 *
 * These tests are the foundation: if g and g^{-1} are not mutually inverse,
 * or if the Schwarzschild limits are wrong, every downstream result (geodesics,
 * shadow size, disk appearance) is meaningless.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  metric,
  inverseMetric,
  idx,
  horizonRadius,
  innerHorizonRadius,
  ergosphereRadius,
  iscoRadius,
  marginallyBoundRadius,
  circularOmega,
  circularEnergyAndAngularMomentum,
  frameDragging,
} from "../src/core/kerr.js";

const SPINS = [0, 0.2, 0.5, 0.9, 0.998];
const TOL = 1e-9;

test("g^{mu alpha} g_{alpha nu} = delta^mu_nu for a range of spins and points", () => {
  for (const a of SPINS) {
    for (const r of [2.5, 5, 10, 40]) {
      for (const theta of [0.3, Math.PI / 2, Math.PI - 0.3]) {
        const g = metric(r, theta, a);
        const gi = inverseMetric(r, theta, a);
        for (let mu = 0; mu < 4; mu++) {
          for (let nu = 0; nu < 4; nu++) {
            let acc = 0;
            for (let al = 0; al < 4; al++) {
              acc += gi[idx(mu, al)] * g[idx(al, nu)];
            }
            const expected = mu === nu ? 1 : 0;
            assert.ok(
              Math.abs(acc - expected) < 1e-6,
              `a=${a} r=${r} th=${theta}: g^-1 g [${mu},${nu}] = ${acc}, expected ${expected}`,
            );
          }
        }
      }
    }
  }
});

test("metric is symmetric", () => {
  for (const a of SPINS) {
    const g = metric(7, 1.1, a);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        assert.ok(Math.abs(g[idx(i, j)] - g[idx(j, i)]) < TOL);
      }
    }
  }
});

test("Schwarzschild limit: a=0 reproduces the standard metric", () => {
  const r = 10;
  const theta = 1.0;
  const g = metric(r, theta, 0);
  const f = 1 - 2 / r;
  assert.ok(Math.abs(g[idx(0, 0)] + f) < TOL, "g_tt");
  assert.ok(Math.abs(g[idx(1, 1)] - 1 / f) < TOL, "g_rr");
  assert.ok(Math.abs(g[idx(2, 2)] - r * r) < TOL, "g_thth");
  assert.ok(Math.abs(g[idx(3, 3)] - r * r * Math.sin(theta) ** 2) < TOL, "g_phiphi");
  assert.ok(Math.abs(g[idx(0, 3)]) < TOL, "off-diagonal g_tphi must vanish at a=0");
});

test("horizon and ergosphere radii", () => {
  assert.ok(Math.abs(horizonRadius(0) - 2) < TOL, "Schwarzschild horizon at 2M");
  assert.ok(Math.abs(horizonRadius(1) - 1) < TOL, "extremal horizon at M");
  assert.ok(Math.abs(innerHorizonRadius(0) - 0) < TOL);
  // Equatorial ergosphere is 2M for any spin; on the axis it meets the horizon.
  for (const a of SPINS) {
    assert.ok(Math.abs(ergosphereRadius(a, Math.PI / 2) - 2) < TOL, `ergosphere equator a=${a}`);
    assert.ok(
      Math.abs(ergosphereRadius(a, 0) - horizonRadius(a)) < TOL,
      `ergosphere on axis meets horizon a=${a}`,
    );
    assert.ok(ergosphereRadius(a, Math.PI / 2) >= horizonRadius(a) - TOL);
  }
});

test("Delta = 0 exactly at the horizon", () => {
  for (const a of SPINS) {
    const rp = horizonRadius(a);
    const delta = rp * rp - 2 * rp + a * a;
    assert.ok(Math.abs(delta) < 1e-9, `Delta(r_+) a=${a} = ${delta}`);
  }
});

test("ISCO limits: 6M non-spinning, 1M extremal prograde, 9M extremal retrograde", () => {
  assert.ok(Math.abs(iscoRadius(0) - 6) < 1e-9, `a=0 ISCO = ${iscoRadius(0)}`);
  assert.ok(Math.abs(iscoRadius(1, true) - 1) < 1e-6, `extremal prograde = ${iscoRadius(1, true)}`);
  assert.ok(Math.abs(iscoRadius(1, false) - 9) < 1e-6, `extremal retrograde = ${iscoRadius(1, false)}`);
  // Tabulated intermediate values (Bardeen 1972).
  assert.ok(Math.abs(iscoRadius(0.5, true) - 4.2331) < 1e-3, `a=0.5 pro = ${iscoRadius(0.5, true)}`);
  assert.ok(Math.abs(iscoRadius(0.9, true) - 2.3209) < 1e-3, `a=0.9 pro = ${iscoRadius(0.9, true)}`);
  assert.ok(Math.abs(iscoRadius(0.5, false) - 7.5546) < 1e-3, `a=0.5 retro = ${iscoRadius(0.5, false)}`);
  // Prograde ISCO must shrink monotonically with spin; retrograde must grow.
  let prevPro = Infinity;
  let prevRetro = -Infinity;
  for (const a of [0, 0.25, 0.5, 0.75, 0.9, 0.99]) {
    const pro = iscoRadius(a, true);
    const retro = iscoRadius(a, false);
    assert.ok(pro < prevPro, `prograde ISCO must decrease with spin at a=${a}`);
    assert.ok(retro > prevRetro, `retrograde ISCO must increase with spin at a=${a}`);
    assert.ok(pro >= horizonRadius(a) - 1e-9, `prograde ISCO outside horizon at a=${a}`);
    prevPro = pro;
    prevRetro = retro;
  }
});

test("marginally bound orbit: 4M non-spinning", () => {
  assert.ok(Math.abs(marginallyBoundRadius(0) - 4) < 1e-9);
});

test("circular orbit energy reproduces Schwarzschild E = (1-2/r)/sqrt(1-3/r)", () => {
  for (const r of [6, 8, 12, 30]) {
    const { energy } = circularEnergyAndAngularMomentum(r, 0, true);
    const expected = (1 - 2 / r) / Math.sqrt(1 - 3 / r);
    assert.ok(
      Math.abs(energy - expected) < 1e-9,
      `a=0 r=${r}: E=${energy}, expected ${expected}`,
    );
  }
});

test("circular orbit angular momentum reproduces Schwarzschild L = sqrt(r/(1-3/r))", () => {
  for (const r of [6, 8, 12, 30]) {
    const { angularMomentum } = circularEnergyAndAngularMomentum(r, 0, true);
    const expected = Math.sqrt(r / (1 - 3 / r));
    assert.ok(
      Math.abs(angularMomentum - expected) < 1e-9,
      `a=0 r=${r}: Lz=${angularMomentum}, expected ${expected}`,
    );
  }
});

test("energy at ISCO matches the known binding energies", () => {
  // a=0: E_isco = 2 sqrt(2)/3 = 0.94281 (5.7% radiative efficiency).
  const { energy } = circularEnergyAndAngularMomentum(6, 0, true);
  assert.ok(Math.abs(energy - (2 * Math.SQRT2) / 3) < 1e-9, `E_isco a=0 = ${energy}`);

  // Extremal prograde ISCO sits exactly at the horizon r = M, so the known
  // limit E = 1/sqrt(3) (42.3% efficiency) must be approached from outside.
  let prev = Infinity;
  for (const eps of [0.5, 0.2, 0.05, 0.01, 0.001]) {
    const c = circularEnergyAndAngularMomentum(1 + eps, 1, true);
    assert.ok(
      c.energy < prev,
      `E must decrease monotonically toward the extremal ISCO (eps=${eps})`,
    );
    prev = c.energy;
  }
  const near = circularEnergyAndAngularMomentum(1.0001, 1, true);
  assert.ok(
    Math.abs(near.energy - 1 / Math.sqrt(3)) < 1e-4,
    `E -> 1/sqrt(3) at extremal ISCO, got ${near.energy}`,
  );
});

test("Newtonian limit far from the hole", () => {
  const r = 5000;
  const { energy, angularMomentum } = circularEnergyAndAngularMomentum(r, 0, true);

  // Energy: E = (1 - 2/r)/sqrt(1 - 3/r) = 1 - 1/(2r) + O(1/r^2).
  assert.ok(Math.abs(energy - (1 - 1 / (2 * r))) < 1e-6, `E = ${energy}`);

  // Angular momentum is the exact Schwarzschild value sqrt(r/(1-3/r)), which
  // approaches the Keplerian sqrt(r) only at first order: the relative
  // correction is 3/(2r). Assert the exact value tightly and the Newtonian
  // approach with a tolerance that reflects the actual O(1/r) error.
  const exact = Math.sqrt(r / (1 - 3 / r));
  assert.ok(Math.abs(angularMomentum - exact) < 1e-9, `Lz = ${angularMomentum} vs exact ${exact}`);
  const newtonian = Math.sqrt(r);
  const relError = Math.abs(angularMomentum - newtonian) / newtonian;
  assert.ok(relError < 1e-3, `Lz should approach sqrt(r) at O(1/r), rel error ${relError}`);
  assert.ok(Math.abs(relError - 3 / (2 * r)) < 1e-6, `leading correction should be 3/(2r) = ${3 / (2 * r)}, got ${relError}`);
});

test("Keplerian omega matches Schwarzschild sqrt(1/r^3) at a=0", () => {
  for (const r of [6, 10, 20]) {
    const omega = circularOmega(r, 0, true);
    assert.ok(Math.abs(omega - Math.sqrt(1 / r ** 3)) < 1e-12, `r=${r}: ${omega}`);
  }
});

test("frame dragging is positive, vanishes at a=0, and peaks near the hole", () => {
  assert.ok(Math.abs(frameDragging(10, Math.PI / 2, 0)) < 1e-12, "a=0 must not drag");
  const near = frameDragging(3, Math.PI / 2, 0.9);
  const far = frameDragging(50, Math.PI / 2, 0.9);
  assert.ok(near > 0, "prograde frame dragging is positive");
  assert.ok(near > far, "frame dragging falls off with radius");
});

/* ------------------------------------------------------------------ *
 * Regression guard for the circular-orbit closed form.
 *
 * A circular geodesic must make the timelike radial potential vanish with
 * zero slope. This check derives E and Lz from the metric by finite
 * difference (via the radial Euler-Lagrange condition), so it shares no
 * algebra with the closed form above — exactly the comparison that caught a
 * wrong Bardeen expression here.
 * ------------------------------------------------------------------ */

/** Timelike radial potential (mu = 1, particle mass = 1). */
function radialPotentialTimelike(r, a, energy, lz) {
  const r2 = r * r;
  const a2 = a * a;
  const p = energy * (r2 + a2) - a * lz;
  const delta = r2 - 2 * r + a2;
  return p * p - delta * (r2 + (lz - a * energy) ** 2);
}

/** E, Lz derived only from the metric: solve dU/dr = 0 for Omega. */
function circularFromMetricOnly(r, a, prograde = true) {
  const h = 1e-6;
  const plus = metric(r + h, Math.PI / 2, a);
  const minus = metric(r - h, Math.PI / 2, a);
  const mid = metric(r, Math.PI / 2, a);
  const gtt = mid[idx(0, 0)];
  const gtph = mid[idx(0, 3)];
  const gpp = mid[idx(3, 3)];
  const dgtt = (plus[idx(0, 0)] - minus[idx(0, 0)]) / (2 * h);
  const dgtph = (plus[idx(0, 3)] - minus[idx(0, 3)]) / (2 * h);
  const dgpp = (plus[idx(3, 3)] - minus[idx(3, 3)]) / (2 * h);

  // 0.5 g_tt' + g_tphi' Omega + 0.5 g_pp' Omega^2 = 0
  const A = 0.5 * dgpp;
  const B = dgtph;
  const C = 0.5 * dgtt;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return null;
  const sq = Math.sqrt(disc);
  const omega = prograde
    ? Math.max((-B + sq) / (2 * A), (-B - sq) / (2 * A))
    : Math.min((-B + sq) / (2 * A), (-B - sq) / (2 * A));

  const U = gtt + 2 * gtph * omega + gpp * omega * omega;
  if (U >= 0) return null;
  const ut = 1 / Math.sqrt(-U);
  return { energy: -ut * (gtt + gtph * omega), lz: ut * (gtph + gpp * omega), omega };
}

test("circular orbit E and Lz satisfy R(r) = 0 and R'(r) = 0", () => {
  const h = 1e-5;
  for (const a of [0, 0.25, 0.5, 0.75, 0.9, 0.998]) {
    for (const prograde of [true, false]) {
      const rIsco = iscoRadius(a, prograde);
      for (const r of [rIsco + 0.5, rIsco + 2, 12, 40]) {
        const { energy, angularMomentum } = circularEnergyAndAngularMomentum(r, a, prograde);
        assert.ok(Number.isFinite(energy), `E finite at a=${a} r=${r}`);
        assert.ok(energy < 1.0001, `bound orbit must have E <= 1 at a=${a} r=${r}, got ${energy}`);

        const residual = radialPotentialTimelike(r, a, energy, angularMomentum);
        const slope =
          (radialPotentialTimelike(r + h, a, energy, angularMomentum) -
            radialPotentialTimelike(r - h, a, energy, angularMomentum)) /
          (2 * h);
        assert.ok(
          Math.abs(residual) < 1e-6,
          `R(r) must vanish: a=${a} pro=${prograde} r=${r} -> ${residual}`,
        );
        assert.ok(
          Math.abs(slope) < 1e-4,
          `R'(r) must vanish: a=${a} pro=${prograde} r=${r} -> ${slope}`,
        );
      }
    }
  }
});

test("closed-form circular orbits agree with the metric-only derivation", () => {
  for (const a of [0, 0.5, 0.9, 0.998]) {
    for (const prograde of [true, false]) {
      for (const r of [iscoRadius(a, prograde) + 0.5, 6, 15]) {
        const closed = circularEnergyAndAngularMomentum(r, a, prograde);
        const derived = circularFromMetricOnly(r, a, prograde);
        assert.ok(derived, `derivation failed a=${a} r=${r}`);
        assert.ok(
          Math.abs(closed.energy - derived.energy) < 1e-6,
          `E mismatch a=${a} pro=${prograde} r=${r}: ${closed.energy} vs ${derived.energy}`,
        );
        assert.ok(
          Math.abs(closed.angularMomentum - derived.lz) < 1e-6,
          `Lz mismatch a=${a} pro=${prograde} r=${r}: ${closed.angularMomentum} vs ${derived.lz}`,
        );
      }
    }
  }
});
