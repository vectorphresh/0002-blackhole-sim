/**
 * Photon orbit and shadow size.
 *
 * The header of `photonOrbit.js` claims these limits are asserted here; before
 * this file existed, they were not. Two real bugs were hiding in
 * `xiAtRadius` as a result:
 *
 *   - `prograde * sq` with a boolean `prograde` evaluates to 0 for retrograde,
 *     so every retrograde call returned NaN at a = 0.
 *   - the +/- branch was chosen backwards, which swaps prograde and retrograde
 *     for every a > 0 while leaving a = 0 exactly correct.
 *
 * Both are invisible to an a = 0 test, which is the point of testing the
 * spinning cases against the published Bardeen–Press–Teukolsky table.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  equatorialR,
  xiAtRadius,
  equatorialPhotonOrbit,
  criticalImpactParameter,
} from "../src/core/photonOrbit.js";
import { horizonRadius } from "../src/core/kerr.js";
import { integrateHamilton, momentumFromTetradDirection } from "../src/core/geodesics.js";

/**
 * Bardeen, Press & Teukolsky (1973), Table 1: equatorial prograde (+) and
 * retrograde (-) photon orbits, radius r_ph and impact parameter xi.
 */
const BARDEEN = [
  { a: 0.0, rPlus: 3.0, xiPlus: 5.19615, rMinus: 3.0, xiMinus: -5.19615 },
  { a: 0.5, rPlus: 2.3473, xiPlus: 4.09627, rMinus: 3.53209, xiMinus: -6.13816 },
  { a: 0.9, rPlus: 1.55785, xiPlus: 2.84442, rMinus: 3.91027, xiMinus: -6.83232 },
  { a: 0.998, rPlus: 1.07391, xiPlus: 2.11089, rMinus: 3.99822, xiMinus: -6.99667 },
];

test("equatorial photon orbits match the Bardeen–Press–Teukolsky table", () => {
  for (const row of BARDEEN) {
    for (const prograde of [true, false]) {
      const want = prograde ? { r: row.rPlus, xi: row.xiPlus } : { r: row.rMinus, xi: row.xiMinus };
      const got = equatorialPhotonOrbit(row.a, prograde);
      assert.ok(
        Math.abs(got.radius - want.r) < 1e-3,
        `r_ph a=${row.a} ${prograde ? "pro" : "ret"}: ${got.radius} vs ${want.r}`,
      );
      assert.ok(
        Math.abs(got.impactParameter - want.xi) < 1e-3,
        `xi a=${row.a} ${prograde ? "pro" : "ret"}: ${got.impactParameter} vs ${want.xi}`,
      );
    }
  }
});

test("prograde orbits lie inside 3M and retrograde outside for a > 0", () => {
  // Frame dragging helps corotating photons orbit closer in. At a = 0 the two
  // coincide, so this is a genuinely spinning-spacetime statement.
  for (const a of [0.3, 0.7, 0.95]) {
    const pro = equatorialPhotonOrbit(a, true);
    const retro = equatorialPhotonOrbit(a, false);
    assert.ok(pro.radius < 3, `prograde r_ph should be < 3M, got ${pro.radius} at a=${a}`);
    assert.ok(retro.radius > 3, `retrograde r_ph should be > 3M, got ${retro.radius} at a=${a}`);
    assert.ok(pro.radius > horizonRadius(a), "photon orbit must sit outside the horizon");
    // Prograde corotates with the hole, so xi has the sign of a.
    assert.ok(pro.impactParameter > 0, "prograde xi must be positive for a > 0");
    assert.ok(retro.impactParameter < 0, "retrograde xi must be negative for a > 0");
  }
});

test("extremal limits: prograde r -> M and b -> 2M, retrograde r -> 4M and b -> -7M", () => {
  // The limit must be *approached*, not evaluated. At a = 1 exactly, the
  // prograde photon orbit sits on the horizon itself, so xi(r) has no
  // stationary point strictly outside r_+ and the solver correctly reports
  // none. Sampling a sequence approaching extremality is the honest test.
  const seq = [0.9, 0.99, 0.999, 0.9999];

  const pro = seq.map((a) => equatorialPhotonOrbit(a, true));
  for (let i = 1; i < pro.length; i++) {
    assert.ok(pro[i].radius < pro[i - 1].radius, "prograde r_ph must decrease toward M");
    assert.ok(
      Math.abs(pro[i].impactParameter - 2) < Math.abs(pro[i - 1].impactParameter - 2),
      "prograde b must converge to 2M",
    );
  }
  assert.ok(Math.abs(pro[pro.length - 1].radius - 1) < 2e-2, `prograde r_ph -> M, got ${pro[pro.length - 1].radius}`);
  assert.ok(Math.abs(pro[pro.length - 1].impactParameter - 2) < 3e-2, `prograde b -> 2M, got ${pro[pro.length - 1].impactParameter}`);

  const retro = seq.map((a) => equatorialPhotonOrbit(a, false));
  for (let i = 1; i < retro.length; i++) {
    assert.ok(retro[i].radius > retro[i - 1].radius, "retrograde r_ph must increase toward 4M");
    assert.ok(
      Math.abs(retro[i].impactParameter + 7) < Math.abs(retro[i - 1].impactParameter + 7),
      "retrograde b must converge to -7M",
    );
  }
  assert.ok(Math.abs(retro[retro.length - 1].radius - 4) < 2e-2, `retrograde r_ph -> 4M, got ${retro[retro.length - 1].radius}`);
  assert.ok(Math.abs(retro[retro.length - 1].impactParameter + 7) < 3e-2, `retrograde b -> -7M, got ${retro[retro.length - 1].impactParameter}`);
});

test("a circular photon orbit really is a double root of R(r)", () => {
  // The arbiter: the orbit found by root-finding on xi(r) must satisfy R = 0
  // and dR/dr = 0 in the Carter potential, which is independent algebra.
  for (const a of [0, 0.4, 0.8, 0.99]) {
    for (const prograde of [true, false]) {
      const { radius, impactParameter } = equatorialPhotonOrbit(a, prograde);
      const r = radius;
      const xi = impactParameter;
      const R = equatorialR(r, a, xi);
      const scale = Math.max(1, r ** 4);
      assert.ok(Math.abs(R) / scale < 1e-6, `R must vanish at a=${a} ${prograde}: ${R}`);

      const h = 1e-5 * Math.max(1, r);
      const dR = (equatorialR(r + h, a, xi) - equatorialR(r - h, a, xi)) / (2 * h);
      assert.ok(
        Math.abs(dR) / scale < 1e-4,
        `dR/dr must vanish at a=${a} ${prograde}: ${dR}`,
      );
    }
  }
});

test("xiAtRadius is finite for retrograde rays at a = 0", () => {
  // Regression test for `prograde * sq` with a boolean: at a = 0 the buggy
  // denominator became exactly 0 and every retrograde radius returned NaN.
  for (const r of [3.1, 5, 12, 40]) {
    assert.ok(Number.isFinite(xiAtRadius(r, 0, false)), `retrograde xi must be finite at r=${r}`);
    assert.ok(
      Math.abs(xiAtRadius(r, 0, false) + xiAtRadius(r, 0, true)) < 1e-9,
      "at a = 0 the two branches must be mirror images",
    );
  }
});

test("critical impact parameter from the integrator agrees with the orbit algebra", () => {
  // b_crit is found by bisecting on actual capture/escape, so this cross-checks
  // the closed-form orbit radius against the geodesic integrator itself.
  const r0 = 400;
  for (const a of [0, 0.5]) {
    const est = Math.abs(equatorialPhotonOrbit(a, true).impactParameter);

    const captures = (b) => {
      const alpha = Math.asin(Math.min(0.99, b / r0));
      const dir = { n1: -Math.cos(alpha), n2: 0, n3: Math.sin(alpha) };
      const p = momentumFromTetradDirection(r0, Math.PI / 2, a, dir, 1);
      const state = [0, r0, Math.PI / 2, 0, p.pt, p.pr, p.pth, p.pphi];
      const res = integrateHamilton({
        state,
        a,
        step: 0.01,
        maxSteps: 400000,
        escapeRadius: r0,
      });
      return res.status === "captured";
    };

    const bCrit = criticalImpactParameter({ a, prograde: true, captures, tol: 1e-5 });
    // A ray launched from finite r0 sees a redshifted shadow: the measured
    // threshold is b_crit * sqrt(1 - 2/r0), not b_crit.
    const expected = est * Math.sqrt(1 - 2 / r0);
    assert.ok(
      Math.abs(bCrit - expected) < 2e-3,
      `b_crit a=${a}: measured ${bCrit}, expected ${expected}`,
    );
  }
});
