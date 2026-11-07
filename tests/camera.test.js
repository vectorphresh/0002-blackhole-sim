/**
 * Orbit camera.
 *
 * The camera lives in Boyer–Lindquist coordinates and expresses ray directions
 * in the ZAMO orthonormal basis, which is the basis `momentumFromTetradDirection`
 * consumes. If the basis is not orthonormal, or if the forward vector is not
 * pointing at the hole, the traced image is silently wrong in a way that looks
 * like a plausible perspective rather than an obvious bug.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  cameraBasis,
  createCamera,
  orbit,
  zoom,
  rayDirection,
  basisMatrix,
} from "../src/core/camera.js";
import { horizonRadius } from "../src/core/kerr.js";
import { momentumFromTetradDirection, nullResidual } from "../src/core/geodesics.js";

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

test("camera basis is orthonormal at every inclination", () => {
  for (let theta = 0.05; theta < Math.PI; theta += 0.05) {
    const { right, up, forward } = cameraBasis(theta);
    for (const [name, v] of [["right", right], ["up", up], ["forward", forward]]) {
      assert.ok(
        Math.abs(norm(v) - 1) < 1e-12,
        `${name} must be unit length at theta=${theta}, got ${norm(v)}`,
      );
    }
    assert.ok(Math.abs(dot(right, up)) < 1e-12, `right.up at theta=${theta}`);
    assert.ok(Math.abs(dot(right, forward)) < 1e-12, `right.forward at theta=${theta}`);
    assert.ok(Math.abs(dot(up, forward)) < 1e-12, `up.forward at theta=${theta}`);
  }
});

test("forward points at the hole, i.e. inward along r", () => {
  // In the ZAMO basis the radial direction is the first component, so looking
  // at the centre means forward = -r_hat exactly.
  for (let theta = 0.1; theta < Math.PI; theta += 0.1) {
    const { forward } = cameraBasis(theta);
    assert.ok(forward[0] < 0, `forward must point inward at theta=${theta}`);
    assert.ok(Math.abs(forward[0] + 1) < 1e-12, `forward must be exactly -r_hat at theta=${theta}`);
    assert.equal(forward[1], 0);
    assert.equal(forward[2], 0);
  }
});

test("up has no radial component, so the camera does not tilt toward the hole", () => {
  for (let theta = 0.1; theta < Math.PI; theta += 0.1) {
    const { up } = cameraBasis(theta);
    assert.equal(up[0], 0, `up must be tangent to the sphere at theta=${theta}`);
  }
});

test("rayDirection returns unit vectors across the whole field of view", () => {
  const cam = createCamera({ r: 12, theta: 1.2, fov: (60 * Math.PI) / 180 });
  for (let x = -1.5; x <= 1.5; x += 0.25) {
    for (let y = -1.5; y <= 1.5; y += 0.25) {
      const d = rayDirection(cam, x, y);
      assert.ok(Math.abs(norm(d) - 1) < 1e-12, `not unit at (${x},${y}): ${norm(d)}`);
    }
  }
});

test("the central ray points exactly at the hole", () => {
  const cam = createCamera({ r: 12, theta: 1.2 });
  const d = rayDirection(cam, 0, 0);
  assert.ok(Math.abs(d[0] + 1) < 1e-12, `central ray must be radial inward, got ${d}`);
  assert.ok(Math.abs(d[1]) < 1e-12);
  assert.ok(Math.abs(d[2]) < 1e-12);
});

test("off-axis rays fan out symmetrically about the optical axis", () => {
  const cam = createCamera({ r: 12, theta: 1.2, fov: (50 * Math.PI) / 180 });
  const centre = rayDirection(cam, 0, 0);
  for (const [x, y] of [[0.5, 0], [-0.5, 0], [0, 0.5], [0, -0.5]]) {
    const d = rayDirection(cam, x, y);
    const angle = Math.acos(Math.min(1, dot(centre, d)));
    const mirror = rayDirection(cam, -x, -y);
    const angleMirror = Math.acos(Math.min(1, dot(centre, mirror)));
    assert.ok(
      Math.abs(angle - angleMirror) < 1e-12,
      `(${x},${y}) gives ${angle}, mirror gives ${angleMirror}`,
    );
    assert.ok(angle > 0, "off-axis ray must actually diverge");
  }
});

test("field of view controls the cone half-angle", () => {
  // The corner of a unit square in NDC is at distance sqrt(2) from the axis,
  // so the extreme ray makes angle atan(sqrt(2) tan(fov/2)).
  for (const fovDeg of [20, 50, 90]) {
    const cam = createCamera({ r: 12, theta: 1.2, fov: (fovDeg * Math.PI) / 180 });
    const centre = rayDirection(cam, 0, 0);
    const edge = rayDirection(cam, 1, 0);
    const angle = Math.acos(Math.min(1, dot(centre, edge)));
    const expected = Math.atan(Math.tan((fovDeg * Math.PI) / 360));
    assert.ok(
      Math.abs(angle - expected) < 1e-9,
      `fov=${fovDeg}deg: half-angle ${angle} vs ${expected}`,
    );
  }
});

test("every camera ray yields a null photon momentum", () => {
  // The strongest check on the basis: the tetrad builder produces a null
  // momentum only when |n| = 1, so a non-orthonormal basis shows up here as a
  // nonzero null residual rather than a subtly skewed image.
  for (const a of [0, 0.5, 0.9]) {
    for (const r of [6, 12, 30]) {
      for (const theta of [0.4, 1.2, 2.4]) {
        const cam = createCamera({ r, theta, fov: (60 * Math.PI) / 180 });
        for (const [x, y] of [[0, 0], [1, 1], [-1, 0.5], [0.3, -1.2]]) {
          const n = rayDirection(cam, x, y);
          const p = momentumFromTetradDirection(r, theta, a, { n1: n[0], n2: n[1], n3: n[2] }, 1);
          const res = nullResidual(r, theta, a, [p.pt, p.pr, p.pth, p.pphi]);
          assert.ok(
            Math.abs(res) < 1e-9,
            `null residual ${res} at a=${a} r=${r} theta=${theta} ndc=(${x},${y})`,
          );
        }
      }
    }
  }
});

test("orbit clamps theta away from the poles where azimuth degenerates", () => {
  const cam = createCamera();
  for (let i = 0; i < 1000; i++) orbit(cam, -1, 0);
  assert.ok(cam.theta > 0 && cam.theta < Math.PI, `theta escaped the open interval: ${cam.theta}`);
  assert.ok(cam.theta <= Math.PI - 1e-3 + 1e-12, `theta too close to pi: ${cam.theta}`);

  for (let i = 0; i < 1000; i++) orbit(cam, 1, 0);
  assert.ok(cam.theta >= 1e-3 - 1e-12, `theta too close to 0: ${cam.theta}`);

  // Azimuth is unbounded — wrapping it would make the camera jump.
  const before = cam.phi;
  orbit(cam, 0, 100);
  assert.equal(cam.phi, before + 100);
});

test("zoom never places the camera inside the horizon", () => {
  for (const a of [0, 0.5, 0.9, 0.998]) {
    const cam = createCamera({ r: 50 });
    for (let i = 0; i < 200; i++) zoom(cam, 0.5, a);
    const rH = horizonRadius(a);
    assert.ok(cam.r > rH, `camera r=${cam.r} inside horizon rH=${rH} at a=${a}`);
    assert.ok(cam.r >= rH + 1.5 - 1e-9, `camera must stay outside the near-horizon region: ${cam.r}`);

    for (let i = 0; i < 200; i++) zoom(cam, 2, a);
    assert.ok(cam.r <= 200 + 1e-9, `camera escaped to r=${cam.r}`);
  }
});

test("basisMatrix has the basis vectors as columns", () => {
  const cam = createCamera({ theta: 1.1 });
  const m = basisMatrix(cam);
  assert.equal(m.length, 9);
  const { right, up, forward } = cameraBasis(cam.theta);
  // Column-major: columns are (right, up, forward).
  assert.deepEqual([m[0], m[1], m[2]], right);
  assert.deepEqual([m[3], m[4], m[5]], up);
  assert.deepEqual([m[6], m[7], m[8]], forward);
});
