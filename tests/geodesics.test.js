/**
 * Geodesic integration validation.
 *
 * The central idea: `integrateHamilton` derives motion from the metric alone,
 * while `integrateCarter` uses the separated Carter constants. They share no
 * algebra, so agreement between them is strong evidence both are right — and
 * disagreement localises the error to one of them.
 *
 * Independent physical anchors used here (no reference to our own code):
 *   - Schwarzschild light deflection   delta = 4M / b
 *   - Schwarzschild capture threshold  b_crit = 3 sqrt(3) M
 *   - Schwarzschild photon sphere      r_ph = 3M
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  carterConstants,
  carterStateFromMomentum,
  carterMomentum,
  integrateHamilton,
  integrateCarter,
  momentumFromTetradDirection,
  nullResidual,
} from "../src/core/geodesics.js";
import { findRoot, horizonRadius } from "../src/core/kerr.js";
import { equatorialPhotonOrbit } from "../src/core/photonOrbit.js";

const R0 = 400; // launch radius, far enough to be asymptotically flat

/**
 * Launch an equatorial photon at r = R0, phi = 0 with (approximately) the
 * requested impact parameter. Returns the initial state and the exact
 * conserved b = Lz / E.
 */
function launch(a, b, r0 = R0) {
  const alpha = Math.asin(Math.min(0.99, b / r0));
  const dir = { n1: -Math.cos(alpha), n2: 0, n3: Math.sin(alpha) };
  const p = momentumFromTetradDirection(r0, Math.PI / 2, a, dir, 1);
  const state = [0, r0, Math.PI / 2, 0, p.pt, p.pr, p.pth, p.pphi];
  return { state, momentum: p, bExact: p.pphi / -p.pt };
}

/** Integrate a prepared state and report its fate plus total |dphi|. */
function shootState(state, a, opts = {}) {
  let phiAccum = 0;
  let prevPhi = state[3];
  const res = integrateHamilton({
    state,
    a,
    step: opts.step ?? 0.05,
    maxSteps: opts.maxSteps ?? 400000,
    escapeRadius: opts.escapeRadius ?? R0,
    onState: (s) => {
      // Accumulate |dphi| without letting branch cuts cancel the winding.
      let d = s[3] - prevPhi;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      phiAccum += d;
      prevPhi = s[3];
    },
  });
  return {
    status: res.status,
    rPeri: res.rPeri,
    steps: res.steps,
    dPhi: Math.abs(phiAccum),
    crossings: res.crossings,
  };
}

/**
 * Launch a photon from `theta0` with an out-of-equatorial direction, built
 * through the tetrad so the momentum stays null. Patching p_theta after the
 * fact would break the null condition and make the two integrators follow
 * different physics.
 *
 * `bFrac` sets the azimuthal share of the motion (hence the impact parameter),
 * `tilt` the polar share, and the remainder is inward radial.
 */
function launchTilted(a, r0, theta0, tilt, bFrac) {
  const n3 = bFrac;
  const n2 = tilt;
  const n1 = -Math.sqrt(Math.max(0, 1 - n2 * n2 - n3 * n3));
  const p = momentumFromTetradDirection(r0, theta0, a, { n1, n2, n3 }, 1);
  const state = [0, r0, theta0, 0, p.pt, p.pr, p.pth, p.pphi];
  return {
    state,
    momentum: p,
    bExact: p.pphi / -p.pt,
    nullRes: nullResidual(r0, theta0, a, [p.pt, p.pr, p.pth, p.pphi]),
  };
}

/** Launch an equatorial photon and integrate it. */
function shoot(a, b, opts = {}) {
  const { state } = launch(a, b);
  return shootState(state, a, opts);
}

/* ------------------------------------------------------------------ *
 * Ray initialisation
 * ------------------------------------------------------------------ */

test("ZAMO tetrad construction yields a null momentum", () => {
  for (const a of [0, 0.5, 0.9, 0.998]) {
    for (const r of [3.5, 8, 25, 120]) {
      for (const theta of [0.6, Math.PI / 2, Math.PI - 0.6]) {
        for (const dir of [
          { n1: 1, n2: 0, n3: 0 },
          { n1: 0, n2: 1, n3: 0 },
          { n1: 0, n2: 0, n3: 1 },
          { n1: 1 / Math.sqrt(3), n2: 1 / Math.sqrt(3), n3: 1 / Math.sqrt(3) },
        ]) {
          const p = momentumFromTetradDirection(r, theta, a, dir, 1);
          const res = nullResidual(r, theta, a, [p.pt, p.pr, p.pth, p.pphi]);
          assert.ok(
            Math.abs(res) < 1e-9,
            `null condition failed a=${a} r=${r} th=${theta} dir=${JSON.stringify(dir)}: ${res}`,
          );
        }
      }
    }
  }
});

test("outward and inward radial rays are both null and point the right way", () => {
  for (const a of [0, 0.9]) {
    const out = momentumFromTetradDirection(20, Math.PI / 2, a, { n1: 1, n2: 0, n3: 0 }, 1);
    const inn = momentumFromTetradDirection(20, Math.PI / 2, a, { n1: -1, n2: 0, n3: 0 }, 1);
    assert.ok(out.pr > 0, "outgoing ray must have p_r > 0");
    assert.ok(inn.pr < 0, "incoming ray must have p_r < 0");
    assert.ok(Math.abs(nullResidual(20, Math.PI / 2, a, [out.pt, out.pr, out.pth, out.pphi])) < 1e-9);
    assert.ok(Math.abs(nullResidual(20, Math.PI / 2, a, [inn.pt, inn.pr, inn.pth, inn.pphi])) < 1e-9);
  }
});

/* ------------------------------------------------------------------ *
 * Conservation laws along the reference trajectory
 * ------------------------------------------------------------------ */

test("E and Lz are conserved by the reference integrator", () => {
  for (const a of [0, 0.3, 0.7, 0.95]) {
    const { state } = launch(a, 12);
    const e0 = -state[4];
    const l0 = state[7];
    let maxE = 0;
    let maxL = 0;
    integrateHamilton({
      state,
      a,
      step: 0.02,
      maxSteps: 300000,
      escapeRadius: R0,
      onState: (s) => {
        maxE = Math.max(maxE, Math.abs(-s[4] - e0));
        maxL = Math.max(maxL, Math.abs(s[7] - l0));
      },
    });
    assert.ok(maxE < 1e-9, `E drifted by ${maxE} at a=${a}`);
    assert.ok(maxL < 1e-9, `Lz drifted by ${maxL} at a=${a}`);
  }
});

test("Carter constant Q is conserved along a reference trajectory", () => {
  // The reference integrator never sees Q. If our definition of Q is wrong, or
  // the integrator is wrong, Q drifts. Either way this fails.
  for (const a of [0, 0.3, 0.7, 0.95]) {
    const { state } = launch(a, 12);
    state[6] = 0.35; // tilt out of the equatorial plane so Q is non-trivial
    const q0 = carterConstants(state[2], a, [state[4], state[5], state[6], state[7]]).q;
    assert.ok(Number.isFinite(q0), "Q must be finite");

    let maxDrift = 0;
    integrateHamilton({
      state,
      a,
      step: 0.02,
      maxSteps: 300000,
      escapeRadius: R0,
      onState: (s) => {
        const q = carterConstants(s[2], a, [s[4], s[5], s[6], s[7]]).q;
        maxDrift = Math.max(maxDrift, Math.abs(q - q0));
      },
    });
    const scale = Math.max(1, Math.abs(q0));
    assert.ok(
      maxDrift / scale < 1e-5,
      `Q drifted by ${maxDrift} (Q0=${q0}) at a=${a} — Carter constant or integrator is wrong`,
    );
  }
});

test("null condition is preserved along a reference trajectory", () => {
  for (const a of [0, 0.5, 0.9]) {
    const { state } = launch(a, 10);
    let maxRes = 0;
    integrateHamilton({
      state,
      a,
      step: 0.02,
      maxSteps: 200000,
      escapeRadius: R0,
      onState: (s) => {
        maxRes = Math.max(maxRes, Math.abs(nullResidual(s[1], s[2], a, [s[4], s[5], s[6], s[7]])));
      },
    });
    assert.ok(maxRes < 1e-6, `null condition violated by ${maxRes} at a=${a}`);
  }
});

/* ------------------------------------------------------------------ *
 * Schwarzschild physical anchors
 * ------------------------------------------------------------------ */

test("Schwarzschild light deflection matches the post-Newtonian series", () => {
  // A ray that starts at finite r0 sweeps 2*arccos(b/r0) in flat space; pi is
  // only the r0 -> infinity limit. Gravity adds the deflection on top of that.
  // r0 must be large compared with the largest b tested, otherwise the
  // finite-radius correction dominates the post-Newtonian term we are checking.
  const a = 0;
  const r0 = 2000;
  const measurements = [];
  for (const b of [80, 160, 320, 640]) {
    const { state, bExact } = launch(a, b, r0);
    const r = shootState(state, a, { step: 0.02, escapeRadius: r0, maxSteps: 2000000 });
    assert.equal(r.status, "escaped", `b=${b} should escape, got ${r.status}`);

    const flat = 2 * Math.acos(Math.min(1, bExact / r0));
    const delta = r.dPhi - flat;
    // Post-Newtonian: delta = 4M/b + 15 pi M^2 / (4 b^2) + O(M^3/b^3), M = 1.
    const predicted = 4 / bExact + (15 * Math.PI) / (4 * bExact * bExact);
    measurements.push({ b: bExact, delta, predicted });
    assert.ok(
      Math.abs(delta - predicted) / predicted < 0.01,
      `b=${b}: measured delta=${delta}, expected ${predicted}`,
    );
  }
  // Leading order: delta * b -> 4M as b grows.
  const last = measurements[measurements.length - 1];
  assert.ok(
    Math.abs(last.delta * last.b - 4) / 4 < 0.02,
    `delta*b should approach 4M, got ${last.delta * last.b}`,
  );
  // Deflection must fall off like 1/b: doubling b halves it.
  for (let i = 1; i < measurements.length; i++) {
    const ratio = measurements[i].delta / measurements[i - 1].delta;
    assert.ok(ratio < 0.56 && ratio > 0.44, `deflection should halve when b doubles, ratio=${ratio}`);
  }
});

test("Schwarzschild capture threshold is b_crit = 3 sqrt(3) M", () => {
  const a = 0;
  const bCrit = 3 * Math.sqrt(3); // 5.19615

  const captures = (b, r0) => {
    const { state } = launch(a, b, r0);
    const res = integrateHamilton({
      state,
      a,
      step: 0.02,
      maxSteps: 2000000,
      escapeRadius: r0,
    });
    return res.status === "captured";
  };

  // Sanity: well below must capture, well above must escape.
  assert.ok(captures(3.0, R0), "b=3 must be captured");
  assert.ok(!captures(9.0, R0), "b=9 must escape");

  // A ray launched at finite r0 escapes iff its local angle exceeds the
  // shadow half-angle, whose sine is b_crit sqrt(1 - 2M/r0) / r0. The launch
  // sets b = r0 sin(alpha), so the measured threshold is
  //
  //   b_measured(r0) = b_crit sqrt(1 - 2M/r0)
  //
  // Asserting this exact relation validates the integrator *and* the
  // gravitational redshift factor baked into the ZAMO tetrad.
  for (const r0 of [400, 1200]) {
    const measured = findRoot((b) => (captures(b, r0) ? 1 : -1), 4.9, 5.6, {
      tol: 1e-5,
      maxIter: 90,
    });
    const predicted = bCrit * Math.sqrt(1 - 2 / r0);
    assert.ok(
      Math.abs(measured - predicted) < 2e-4,
      `r0=${r0}: b_measured=${measured}, expected ${predicted}`,
    );
  }

  // The r0 -> infinity limit must recover 3 sqrt(3).
  const far = findRoot((b) => (captures(b, 4000) ? 1 : -1), 5.0, 5.4, {
    tol: 1e-5,
    maxIter: 90,
  });
  assert.ok(
    Math.abs(far - bCrit) < 2e-3,
    `r0 -> infinity should give 3 sqrt(3) = ${bCrit}, got ${far}`,
  );
});

test("rays aimed at the photon sphere linger near r = 3M", () => {
  const a = 0;
  // Just above the capture threshold, the photon orbits near r = 3M before
  // escaping. That winding is what produces the photon ring in the image.
  const r = shoot(a, 5.25, { step: 0.01, maxSteps: 600000 });
  assert.ok(r.rPeri < 3.6, `near-critical ray should approach the photon sphere, rPeri=${r.rPeri}`);
  assert.ok(r.dPhi > 2 * Math.PI, `near-critical ray should wind at least once, dPhi=${r.dPhi}`);
});

test("direct capture cone: tiny impact parameter is swallowed", () => {
  for (const a of [0, 0.5, 0.9]) {
    const r = shoot(a, 0.05, { step: 0.02 });
    assert.equal(r.status, "captured", `a=${a}: a nearly radial ray must be captured`);
    assert.ok(r.rPeri <= horizonRadius(a) + 1e-2, `must reach the horizon, rPeri=${r.rPeri}`);
  }
});

test("captured rays terminate at a physical radius, independent of step size", () => {
  // Boyer–Lindquist coordinates are singular at r_+: Delta changes sign and the
  // inverse metric diverges. An infalling ray is regular and dr/dlambda stays
  // finite, but p_r diverges, so a fixed RK4 step can evaluate its *later
  // stages* inside the horizon and feed a huge spurious dp_r/dlambda back into
  // the accepted step.
  //
  // Measured before the guard was added, a = 0, b = 4, step 0.02: the step from
  // r = 2.001125 evaluated its fourth stage at r = 1.9837 and landed at
  // r = 9.37, then at r = -1.93e9. The existing assertion
  // `rPeri <= horizonRadius(a) + 1e-2` passed on that value, because a
  // negative radius is indeed less than 2.01. This test closes that hole: it
  // bounds rPeri from *below* as well as above, and requires step-independence.
  const rMinSlack = 1e-2;
  for (const a of [0, 0.5, 0.9]) {
    const rp = horizonRadius(a);
    const rMin = rp + 1e-3;
    // Derive the impact parameters from the prograde critical value rather
    // than hard-coding them: b_crit falls from 5.196M at a=0 to 2.844M at
    // a=0.9, so a fixed b that is captured at a=0 escapes at a=0.9.
    const bCrit = equatorialPhotonOrbit(a, true).impactParameter;
    for (const frac of [0.5, 0.8]) {
      const b = frac * bCrit;
      const perimeters = [];
      for (const step of [0.02, 0.005]) {
        const r = shoot(a, b, { step, maxSteps: 900000 });
        assert.equal(r.status, "captured", `a=${a} b=${b.toFixed(4)} step=${step}: must be captured`);
        assert.ok(
          Number.isFinite(r.rPeri),
          `a=${a} b=${b.toFixed(4)} step=${step}: rPeri must be finite, got ${r.rPeri}`,
        );
        assert.ok(
          r.rPeri > rp,
          `a=${a} b=${b.toFixed(4)} step=${step}: rPeri=${r.rPeri} is at or below the horizon ` +
            `r_+=${rp}; the integrator crossed the coordinate singularity`,
        );
        assert.ok(
          r.rPeri <= rMin + rMinSlack,
          `a=${a} b=${b.toFixed(4)} step=${step}: rPeri=${r.rPeri} never reached the termination radius ${rMin}`,
        );
        perimeters.push(r.rPeri);
      }
      // The overshoot past rMin is a step-size artefact, so the two runs differ
      // by O(step). 5e-3 covers 0.02 comfortably without hiding a blow-up.
      assert.ok(
        Math.abs(perimeters[0] - perimeters[1]) < 5e-3,
        `a=${a} b=${b.toFixed(4)}: rPeri is step-dependent, ${perimeters[0]} vs ${perimeters[1]}`,
      );
    }
  }
});

/* ------------------------------------------------------------------ *
 * Cross-validation: reference vs separated Carter integrator
 * ------------------------------------------------------------------ */

test("Carter integrator reproduces the reference integrator's trajectory", () => {
  for (const a of [0, 0.4, 0.8]) {
    for (const b of [8, 14, 25]) {
      const { state } = launch(a, b);
      const ref = integrateHamilton({
        state,
        a,
        step: 0.02,
        maxSteps: 300000,
        escapeRadius: R0,
      });

      const cState = carterStateFromMomentum(
        { r: state[1], theta: state[2], t: state[0], phi: state[3] },
        a,
        [state[4], state[5], state[6], state[7]],
      );
      const cart = integrateCarter({
        state: cState,
        a,
        step: 0.02,
        maxSteps: 300000,
        escapeRadius: R0,
      });

      assert.equal(
        cart.status,
        ref.status,
        `status mismatch a=${a} b=${b}: carter=${cart.status} ref=${ref.status}`,
      );
      assert.ok(
        Math.abs(cart.rPeri - ref.rPeri) < 1e-3,
        `closest approach mismatch a=${a} b=${b}: carter=${cart.rPeri} ref=${ref.rPeri}`,
      );
    }
  }
});

test("Carter momentum recovery agrees with the reference momentum", () => {
  for (const a of [0, 0.5, 0.9]) {
    const { state } = launch(a, 12);
    const cState = carterStateFromMomentum(
      { r: state[1], theta: state[2] },
      a,
      [state[4], state[5], state[6], state[7]],
    );
    const rec = carterMomentum(cState, a);
    assert.ok(Math.abs(rec.pt - state[4]) < 1e-10, "p_t recovery");
    assert.ok(Math.abs(rec.pphi - state[7]) < 1e-10, "p_phi recovery");
    assert.ok(Math.abs(rec.pr - state[5]) < 1e-8, `p_r recovery: ${rec.pr} vs ${state[5]}`);
    assert.ok(Math.abs(rec.pth - state[6]) < 1e-8, `p_theta recovery: ${rec.pth} vs ${state[6]}`);
  }
});

test("equatorial disk crossings agree between integrators", () => {
  // The renderer reads the disk where a ray crosses theta = pi/2, so those
  // crossing events must match between the two integrators.
  //
  // The launch starts slightly *below* the equator moving upward, so the ray
  // genuinely crosses the plane. Launching exactly on it would never register
  // a crossing, and patching p_theta afterwards would break the null condition
  // and make the two integrators follow different physics.
  const r0 = 60;
  for (const a of [0, 0.6]) {
    const { state, nullRes } = launchTilted(a, r0, Math.PI / 2 - 0.1, 0.2, 0.03);
    assert.ok(Math.abs(nullRes) < 1e-12, `launch must be null, residual ${nullRes}`);

    const ref = integrateHamilton({
      state: Float64Array.from(state),
      a,
      step: 0.005,
      maxSteps: 2000000,
      escapeRadius: r0,
    });
    const cState = carterStateFromMomentum(
      { r: state[1], theta: state[2] },
      a,
      [state[4], state[5], state[6], state[7]],
    );
    const cart = integrateCarter({
      state: cState,
      a,
      step: 0.005,
      maxSteps: 2000000,
      escapeRadius: r0,
    });

    assert.equal(cart.status, ref.status, `status mismatch at a=${a}`);
    assert.ok(ref.crossings.length > 0, "expected at least one equatorial crossing");
    assert.equal(
      cart.crossings.length,
      ref.crossings.length,
      `crossing count mismatch at a=${a}: ${cart.crossings.length} vs ${ref.crossings.length}`,
    );

    for (let i = 0; i < ref.crossings.length; i++) {
      assert.ok(
        Math.abs(ref.crossings[i].r - cart.crossings[i].r) < 5e-3,
        `crossing ${i} radius mismatch a=${a}: ${ref.crossings[i].r} vs ${cart.crossings[i].r}`,
      );
      assert.ok(
        Math.abs(ref.crossings[i].phi - cart.crossings[i].phi) < 5e-3,
        `crossing ${i} phi mismatch a=${a}: ${ref.crossings[i].phi} vs ${cart.crossings[i].phi}`,
      );
      assert.equal(
        ref.crossings[i].dir,
        cart.crossings[i].dir,
        `crossing ${i} direction mismatch a=${a}`,
      );
    }
  }
});
