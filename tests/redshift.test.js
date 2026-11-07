/**
 * Relativistic disk emission: redshift factor and temperature profile.
 *
 * The redshift factor is the one piece of the renderer's shading model that can
 * be checked against closed-form limits, and getting it inverted is easy and
 * visually convincing — light would simply look blueshifted on the wrong side.
 * So every branch is pinned to a limit that can be derived independently.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  orbitalOmega,
  redshiftFactor,
  diskTemperature,
  observedTemperature,
} from "../src/core/redshift.js";
import { diskInnerRadius } from "../src/core/kerr.js";

test("static emitter: pure gravitational redshift sqrt(1 - 2/r)", () => {
  // The cleanest arbiter in the whole shading model. With no orbital motion
  // (direction = 0) the only effect is gravitational redshift, which must
  // REDSHIFT (g < 1) light climbing out. A formula that returns u^t instead of
  // 1/u^t gives > 1 here.
  for (const r of [3, 5, 10, 50, 200]) {
    const g = redshiftFactor(r, 0, 0, 0);
    const expected = Math.sqrt(1 - 2 / r);
    assert.ok(
      Math.abs(g - expected) < 1e-12,
      `r=${r}: g=${g} vs sqrt(1-2/r)=${expected}`,
    );
    assert.ok(g < 1, `static emitter must be redshifted, got g=${g}`);
  }
});

test("flat-space limit: g -> 1 far from the hole", () => {
  for (const a of [0, 0.5, 0.9]) {
    const g = redshiftFactor(1e5, a, 0, 1);
    assert.ok(Math.abs(g - 1) < 1e-3, `a=${a}: g=${g} should approach 1`);
  }
});

test("special-relativistic limit: longitudinal Doppler for forward emission", () => {
  // Far from the hole the orbit is Newtonian with v = r Omega = r^-1/2. A
  // photon emitted exactly along the direction of motion carries local
  // tangential momentum E_local, so Lz = sqrt(g_phiphi) E_local ~ r E and
  // xi ~ r. The classical Doppler factor is sqrt((1+v)/(1-v)).
  //
  // The residual is the gravitational redshift and transverse Doppler, both
  // O(1/r), so the tolerance is O(1/r) — not O(1/sqrt(r)), which would be
  // loose enough to pass a formula that was wrong at leading order.
  for (const r of [100, 400, 1600]) {
    const v = 1 / Math.sqrt(r);
    const doppler = Math.sqrt((1 + v) / (1 - v));
    const g = redshiftFactor(r, 0, r, 1);
    const rel = Math.abs(g - doppler) / doppler;
    assert.ok(rel < 2 / r, `r=${r}: g=${g} vs Doppler ${doppler} (rel ${rel})`);
    assert.ok(g > 1, `approaching emitter must be blueshifted, got ${g}`);
  }
});

test("xi = 0 in Schwarzschild is exactly sqrt(1 - 3/r)", () => {
  // xi = 0 means zero angular momentum about the hole, so the photon is
  // emitted perpendicular to the orbital motion and the only effects left are
  // gravitational redshift and time dilation. Those combine exactly:
  //
  //   norm = g_tt + g_phiphi Omega^2 = -(1 - 2/r) + 1/r = -1 + 3/r
  //   g    = 1/u^t = sqrt(1 - 3/r)
  //
  // Exact, not asymptotic — so this pins the formula to machine precision.
  for (const r of [3.01, 5, 10, 100, 1000]) {
    const g = redshiftFactor(r, 0, 0, 1);
    const expected = Math.sqrt(1 - 3 / r);
    assert.ok(
      Math.abs(g - expected) < 1e-12,
      `r=${r}: g=${g} vs sqrt(1-3/r)=${expected}`,
    );
  }
});

test("Doppler beaming is asymmetric at a = 0 too; spin deepens the disk", () => {
  // An earlier version of this test asserted that a = 0 must be left-right
  // symmetric. That was wrong: Doppler beaming comes from the *orbital motion*
  // of the emitter, which exists in Schwarzschild. The spacetime mirror
  // symmetry is (phi -> -phi), which flips the spin AND the sign of xi
  // together; flipping xi alone with the orbital direction held fixed is not a
  // symmetry of anything.
  for (const r of [6, 12, 30]) {
    const gp = redshiftFactor(r, 0, 2, 1);
    const gm = redshiftFactor(r, 0, -2, 1);
    assert.ok(gp > gm, `a=0 r=${r}: approaching ${gp} must exceed receding ${gm}`);
  }

  // Spin does NOT increase the beaming asymmetry at a fixed radius. The
  // prograde orbital frequency is Omega = 1/(r^(3/2) + a), which *decreases*
  // with a, so at a given r a spinning hole is a slightly gentler beamer.
  const asym = (a, r) => redshiftFactor(r, a, 2, 1) / redshiftFactor(r, a, -2, 1);
  assert.ok(
    asym(0.9, 8) < asym(0, 8),
    `at fixed r the asymmetry should shrink with a: ${asym(0, 8)} vs ${asym(0.9, 8)}`,
  );

  // What spin actually does is let the disk survive much closer to the hole:
  // the ISCO falls from 6M to 2.32M at a = 0.9. Measured relative to each
  // disk's own inner edge, the beaming asymmetry grows sharply, because the
  // material is orbiting in a far deeper part of the potential well. This is
  // the qualitative difference between a Schwarzschild and a Kerr disk.
  const asymAtDisk = (a) => {
    const rIn = diskInnerRadius(a, true);
    const r = rIn * 1.5;
    return asym(a, r);
  };
  assert.ok(
    asymAtDisk(0.9) > asymAtDisk(0.5) && asymAtDisk(0.5) > asymAtDisk(0),
    `disk-relative asymmetry must grow with a: ${asymAtDisk(0)}, ${asymAtDisk(0.5)}, ${asymAtDisk(0.9)}`,
  );
});

test("beaming strengthens inward, where the orbital speed rises", () => {
  // The Doppler spread between the two sides of the disk must grow as r
  // approaches the inner edge. This is the qualitative signature of a
  // relativistic disk and it depends on the sign of the formula being right.
  for (const a of [0, 0.5, 0.9]) {
    const spread = (r) => {
      const gp = redshiftFactor(r, a, 3, 1);
      const gm = redshiftFactor(r, a, -3, 1);
      return gp / gm;
    };
    assert.ok(
      spread(4) > spread(10) && spread(10) > spread(24),
      `a=${a}: Doppler spread must increase inward: ${spread(4)}, ${spread(10)}, ${spread(24)}`,
    );
  }
});

test("a retrograde disk around +a is a prograde disk around -a", () => {
  // The genuine mirror symmetry: reversing the orbital direction and flipping
  // the spin sign describes the same physical configuration seen from the
  // other side of the disk, with the photon's angular momentum reversed.
  //
  // An earlier version asserted g(r, a, -xi, -1) == g(r, a, xi, +1) with the
  // spin held fixed. That is not a symmetry — it compares two genuinely
  // different configurations, and it fails even at a = 0.
  for (const a of [0, 0.4, 0.8]) {
    for (const r of [5, 10, 20]) {
      for (const xi of [0, 1.5, 4]) {
        const pro = redshiftFactor(r, a, -xi, -1);
        const retro = redshiftFactor(r, -a, xi, 1);
        assert.ok(
          Math.abs(pro - retro) < 1e-12,
          `a=${a} r=${r} xi=${xi}: ${pro} vs ${retro}`,
        );
      }
    }
  }

  // Corollary that matters for the renderer: around a non-spinning hole,
  // prograde and retrograde disks are optically identical.
  for (const r of [5, 10, 20]) {
    for (const xi of [0, 1.5, 4]) {
      const pro = redshiftFactor(r, 0, xi, 1);
      const retro = redshiftFactor(r, 0, -xi, -1);
      assert.ok(Math.abs(pro - retro) < 1e-12, `a=0 r=${r} xi=${xi}: ${pro} vs ${retro}`);
    }
  }
});

test("no circular orbit at or inside the photon orbit", () => {
  // The light surface is where the circular orbit becomes null:
  //   norm = g_tt + 2 g_tphi Omega + g_phiphi Omega^2
  // For a = 0 this is -1 + 2/r + 1/r = -1 + 3/r, which vanishes at r = 3M —
  // the photon orbit. Inside it no timelike circular worldline exists, so the
  // emitter has no four-velocity and g is undefined.
  assert.equal(redshiftFactor(3, 0, 0, 1), null, "circular orbit at r=3M is null, not timelike");
  assert.equal(redshiftFactor(2, 0, 0, 1), null, "inside the photon orbit there is no circular emitter");
  assert.ok(redshiftFactor(3.01, 0, 0, 1) !== null, "just outside r=3M must still work");
});

test("orbitalOmega matches the closed form and the Keplerian limit", () => {
  for (const a of [0, 0.5, 0.9]) {
    for (const r of [3, 6, 15]) {
      assert.equal(orbitalOmega(r, a, 1), 1 / (r * Math.sqrt(r) + a));
      assert.equal(orbitalOmega(r, a, -1), -1 / (r * Math.sqrt(r) - a));
    }
  }
  // a = 0 must be exactly Kepler.
  for (const r of [4, 9, 25]) {
    assert.ok(Math.abs(orbitalOmega(r, 0, 1) - Math.pow(r, -1.5)) < 1e-15);
  }
});

test("disk temperature profile peaks inside the disk and vanishes at the inner edge", () => {
  const rIn = 2.32;
  assert.equal(diskTemperature(rIn, rIn), 0, "flux factor must vanish at the inner edge");
  assert.equal(diskTemperature(1.5, rIn), 0, "no emission inside the disk");

  // T ~ r^-3/4 away from the inner edge, so it must fall monotonically once
  // past the peak, and the peak must sit close to the inner edge.
  let peakR = rIn;
  let peakT = 0;
  for (let r = rIn; r <= 40; r += 0.01) {
    const t = diskTemperature(r, rIn);
    if (t > peakT) {
      peakT = t;
      peakR = r;
    }
  }
  assert.ok(peakR > rIn && peakR < rIn + 1.5, `peak should sit near the inner edge, got r=${peakR}`);
  assert.ok(diskTemperature(40, rIn) < diskTemperature(6, rIn), "temperature must fall outward");
});

test("observedTemperature combines the profile with the redshift", () => {
  const rIn = 2.32;
  const a = 0.9;
  const r = 6;
  const g = redshiftFactor(r, a, 2, 1);
  const tObs = observedTemperature(r, a, 2, rIn, 1);
  assert.ok(Math.abs(tObs - g * diskTemperature(r, rIn)) < 1e-12);

  // The blueshifted side must look hotter than the redshifted side.
  const tApproach = observedTemperature(r, a, 2, rIn, 1);
  const tRecede = observedTemperature(r, a, -2, rIn, 1);
  assert.ok(tApproach > tRecede, `${tApproach} should exceed ${tRecede}`);
});
