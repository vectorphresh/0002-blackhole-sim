/**
 * Mino-time / velocity-Verlet validation.
 *
 * This is the algorithm the GLSL shader implements, so it must agree with the
 * metric-only reference integrator. Mino time is a pure reparametrisation of
 * affine parameter, so the *path* through spacetime — and therefore every
 * equatorial crossing event — must match exactly; only the parameter differs.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  integrateVerlet,
  radialAccel,
  polarAccel,
  verletStep,
  adaptiveStep,
} from "../src/core/mino.js";
import {
  carterStateFromMomentum,
  integrateHamilton,
  momentumFromTetradDirection,
  radialPotential,
  polarPotential,
} from "../src/core/geodesics.js";

const R0 = 400;

function launch(a, b, r0 = R0) {
  const alpha = Math.asin(Math.min(0.99, b / r0));
  const dir = { n1: -Math.cos(alpha), n2: 0, n3: Math.sin(alpha) };
  const p = momentumFromTetradDirection(r0, Math.PI / 2, a, dir, 1);
  return { state: [0, r0, Math.PI / 2, 0, p.pt, p.pr, p.pth, p.pphi], p };
}

function launchTilted(a, r0, theta0, tilt, bFrac) {
  const n3 = bFrac;
  const n2 = tilt;
  const n1 = -Math.sqrt(Math.max(0, 1 - n2 * n2 - n3 * n3));
  const p = momentumFromTetradDirection(r0, theta0, a, { n1, n2, n3 }, 1);
  return [0, r0, theta0, 0, p.pt, p.pr, p.pth, p.pphi];
}

/** Build a MinoState from a reference-integrator state vector. */
function minoFromState(state, a) {
  const cs = carterStateFromMomentum(
    { r: state[1], theta: state[2], t: state[0], phi: state[3] },
    a,
    [state[4], state[5], state[6], state[7]],
  );
  return cs;
}

test("Verlet acceleration is the analytic half-derivative of the potential", () => {
  // d2r/dlambda_M2 = (1/2) dR/dr must hold, and it must be independent of the
  // direction of motion — that is what makes Verlet sign-free at turning points.
  //
  // The closed forms are compared against a central difference. The step is
  // scaled with r because R ~ E^2 r^4: a fixed h would make the difference
  // pure cancellation noise out at large radius. The tolerance is likewise
  // scaled, since the FD error is O(h^2 R''') and the round-off is
  // O(eps R / h), both relative to the acceleration itself.
  for (const a of [0, 0.5, 0.9]) {
    for (const r of [4, 8, 20, 60]) {
      const E = 1;
      const L = 10;
      const Q = 3;
      const h = 1e-4 * r;
      const fd = (radialPotential(r + h, a, E, L, Q) - radialPotential(r - h, a, E, L, Q)) / (2 * h) * 0.5;
      const exact = radialAccel(r, a, E, L, Q);
      assert.ok(
        Math.abs(exact - fd) < 1e-5 * (1 + Math.abs(exact)),
        `radialAccel a=${a} r=${r}: exact=${exact} fd=${fd}`,
      );

      const th = 1.0;
      const ht = 1e-4;
      const fdT = (polarPotential(th + ht, a, E, L, Q) - polarPotential(th - ht, a, E, L, Q)) / (2 * ht) * 0.5;
      const exactT = polarAccel(th, a, E, L);
      assert.ok(
        Math.abs(exactT - fdT) < 1e-5 * (1 + Math.abs(exactT)),
        `polarAccel a=${a}: exact=${exactT} fd=${fdT}`,
      );
    }
  }
});

test("Verlet leaves a radial turning point with the correct sign, no bookkeeping", () => {
  // Start exactly at periastron (R = 0, v = 0). A first-order method cannot
  // move; Verlet must accelerate outward because R' > 0 there.
  const a = 0.5;
  const E = 1;
  const L = 12;
  const Q = 0;

  // Scan inward from outside the orbit: R > 0 in the allowed region and turns
  // negative at periastron, so the first + -> - crossing is the turning point.
  // The grid is coarse — R' is O(10^4) here, so a 1e-4 bracket leaves a
  // residual of several units — and the bracket is bisected afterwards.
  let lo = null;
  let hi = 60.0;
  let prevR = radialPotential(hi, a, E, L, Q);
  assert.ok(prevR > 0, "launch point must be in the allowed region");
  for (let r = 60.0 - 1e-4; r > 3.0; r -= 1e-4) {
    const cur = radialPotential(r, a, E, L, Q);
    if (prevR > 0 && cur <= 0) {
      lo = r;
      hi = r + 1e-4;
      break;
    }
    prevR = cur;
  }
  assert.ok(lo !== null, "expected a radial turning point");

  // Bisect: R > 0 at hi (allowed), R <= 0 at lo (forbidden).
  for (let i = 0; i < 80; i++) {
    const mid = 0.5 * (lo + hi);
    if (radialPotential(mid, a, E, L, Q) > 0) hi = mid;
    else lo = mid;
  }
  const rTurn = 0.5 * (lo + hi);
  assert.ok(Math.abs(radialPotential(rTurn, a, E, L, Q)) < 1e-3, "should sit on R = 0");
  assert.ok(radialAccel(rTurn, a, E, L, Q) > 0, "acceleration must point outward at periastron");

  const s = { r: rTurn, theta: Math.PI / 2, t: 0, phi: 0, energy: E, lz: L, q: Q, signR: -1, signT: 1 };
  assert.ok(Math.abs(radialPotential(s.r, a, E, L, Q)) < 1e-3);

  // SignR deliberately points inward (-1) even though this is periastron.
  // Verlet should override it, because the acceleration is outward.
  //
  // The step comes from the adaptive controller rather than being fixed. That
  // is not cosmetic: here R' ~ 1175, so a fixed step of 0.05 overshoots to
  // r = 12.4 in one iteration, and since R ~ E^2 r^4 the overshoot compounds —
  // six steps later r is 5e12 and the state is NaN. The controller limits the
  // step by sqrt(r / |a_r|), which keeps the acceleration term subdominant.
  for (let i = 0; i < 40; i++) {
    verletStep(s, a, adaptiveStep(s, a, 0.02));
    assert.ok(Number.isFinite(s.r), `state must stay finite at step ${i}`);
    assert.ok(radialPotential(s.r, a, E, L, Q) >= -1e-6, `r must stay in the allowed region: ${s.r}`);
  }
  assert.ok(s.r > rTurn, `must move outward from periastron: r=${s.r}, rTurn=${rTurn}`);
  assert.ok(s.signR > 0, "signR must flip to outward");
});

test("Verlet trajectory matches the metric-only reference integrator", () => {
  for (const a of [0, 0.4, 0.8]) {
    for (const b of [8, 14, 25]) {
      const { state } = launch(a, b);
      const ref = integrateHamilton({
        state,
        a,
        step: 0.02,
        maxSteps: 400000,
        escapeRadius: R0,
      });
      const ver = integrateVerlet({
        state: minoFromState(state, a),
        a,
        step: 0.02,
        maxSteps: 400000,
        escapeRadius: R0,
      });

      assert.equal(
        ver.status,
        ref.status,
        `status mismatch a=${a} b=${b}: verlet=${ver.status} ref=${ref.status}`,
      );
      assert.ok(
        Math.abs(ver.rPeri - ref.rPeri) < 1e-3,
        `closest approach mismatch a=${a} b=${b}: verlet=${ver.rPeri} ref=${ref.rPeri}`,
      );
    }
  }
});

test("Verlet converges to the reference trajectory as the step shrinks", () => {
  // Second-order convergence: halving the step should quarter the error.
  const a = 0.6;
  const b = 10;
  const { state } = launch(a, b);
  const ref = integrateHamilton({
    state: Float64Array.from(state),
    a,
    step: 0.001,
    maxSteps: 4000000,
    escapeRadius: R0,
  });

  const errors = [];
  for (const step of [0.04, 0.02, 0.01]) {
    const ver = integrateVerlet({
      state: minoFromState(state, a),
      a,
      step,
      maxSteps: 4000000,
      escapeRadius: R0,
    });
    errors.push(Math.abs(ver.rPeri - ref.rPeri));
  }
  assert.ok(errors[0] > errors[1], `error must decrease: ${errors}`);
  assert.ok(errors[1] > errors[2], `error must decrease: ${errors}`);
  const ratio = (errors[1] / errors[2] + errors[0] / errors[1]) / 2;
  assert.ok(ratio > 2.5, `expected ~4x error reduction per halving, got ${ratio}`);
});

test("Verlet equatorial crossings match the reference integrator", () => {
  const r0 = 60;
  for (const a of [0, 0.6]) {
    const state = launchTilted(a, r0, Math.PI / 2 - 0.1, 0.2, 0.03);
    const ref = integrateHamilton({
      state: Float64Array.from(state),
      a,
      step: 0.005,
      maxSteps: 2000000,
      escapeRadius: r0,
    });
    const ver = integrateVerlet({
      state: minoFromState(state, a),
      a,
      step: 0.005,
      maxSteps: 2000000,
      escapeRadius: r0,
    });

    assert.equal(ver.status, ref.status, `status mismatch a=${a}`);
    assert.ok(ref.crossings.length > 0, "expected crossings");
    assert.equal(
      ver.crossings.length,
      ref.crossings.length,
      `crossing count a=${a}: ${ver.crossings.length} vs ${ref.crossings.length}`,
    );
    for (let i = 0; i < ref.crossings.length; i++) {
      assert.ok(
        Math.abs(ref.crossings[i].r - ver.crossings[i].r) < 5e-3,
        `crossing ${i} radius a=${a}: ${ref.crossings[i].r} vs ${ver.crossings[i].r}`,
      );
      assert.ok(
        Math.abs(ref.crossings[i].phi - ver.crossings[i].phi) < 5e-3,
        `crossing ${i} phi a=${a}: ${ref.crossings[i].phi} vs ${ver.crossings[i].phi}`,
      );
    }
  }
});

test("Verlet handles rays that wind around the hole many times", () => {
  // Near-critical rays cross the disk plane repeatedly; the renderer depends on
  // getting every one of them.
  const a = 0;
  const { state } = launch(a, 5.25);
  const ref = integrateHamilton({
    state: Float64Array.from(state),
    a,
    step: 0.005,
    maxSteps: 2000000,
    escapeRadius: R0,
  });
  const ver = integrateVerlet({
    state: minoFromState(state, a),
    a,
    step: 0.005,
    maxSteps: 2000000,
    escapeRadius: R0,
  });
  assert.equal(ver.status, ref.status);
  assert.ok(ref.rPeri < 3.6, "should approach the photon sphere");
  assert.ok(
    Math.abs(ver.rPeri - ref.rPeri) < 1e-2,
    `winding ray closest approach: verlet=${ver.rPeri} ref=${ref.rPeri}`,
  );
});
