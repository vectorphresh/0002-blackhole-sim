/**
 * Orbit camera in Boyer–Lindquist coordinates.
 *
 * The camera hovers at fixed (r, theta, phi) and corotates with the local
 * inertial frame (ZAMO). Ray directions are therefore expressed in the ZAMO
 * orthonormal basis (r_hat, theta_hat, phi_hat), which is exactly the basis
 * `momentumFromTetradDirection` consumes and the basis the shader builds its
 * photon momentum in.
 *
 * Working in BL coordinates rather than Cartesian ones is what makes the
 * camera cheap: orbiting the hole is an increment of phi, tilting is an
 * increment of theta, and no basis rotation matrix is ever needed.
 */

import { horizonRadius } from "./kerr.js";

/**
 * @typedef {Object} CameraBasis
 * @property {number[]} right   x-component in (r_hat, theta_hat, phi_hat)
 * @property {number[]} up
 * @property {number[]} forward
 */

/**
 * Build the camera's orthonormal triad in the ZAMO basis.
 *
 * The camera looks at the hole centre, so forward = -r_hat. "Up" is chosen to
 * be the projection of the symmetry axis, which in the ZAMO basis is
 *
 *   z_hat = cos(theta) r_hat - sin(theta) theta_hat
 *
 * (verify against the flat-space spherical triad r_hat x theta_hat = phi_hat).
 * Then right = forward x z_hat = sin(theta) phi_hat, and up = right x forward
 * = -theta_hat. The result is independent of theta except through the
 * degeneracy at the poles, where sin(theta) -> 0 and the roll is undefined.
 *
 * @param {number} theta polar angle
 * @returns {CameraBasis} unit vectors as [r, theta, phi] triples
 */
export function cameraBasis(theta) {
  const s = Math.sin(theta);
  if (Math.abs(s) < 1e-6) {
    // At a pole the azimuthal direction is undefined; fall back to a
    // continuous choice so the camera does not flip discontinuously.
    return {
      right: [0, 0, 1],
      up: [0, -Math.sign(s || 1), 0],
      forward: [-1, 0, 0],
    };
  }
  return {
    right: [0, 0, 1],
    up: [0, -1, 0],
    forward: [-1, 0, 0],
  };
}

/**
 * @typedef {Object} Camera
 * @property {number} r        Boyer-Lindquist radius, in units of M
 * @property {number} theta    polar angle, (0, pi)
 * @property {number} phi      azimuthal angle
 * @property {number} fov      vertical field of view, radians
 */

/**
 * @returns {Camera} a camera looking at the hole from the given distance
 */
export function createCamera(opts = {}) {
  const { r = 14, theta = 1.35, phi = 0, fov = (50 * Math.PI) / 180 } = opts;
  return { r, theta, phi, fov };
}

/**
 * Orbit the camera. Both deltas are in radians; `dTheta` is clamped away from
 * the poles, where the azimuth degenerates.
 *
 * @param {Camera} cam mutated in place
 */
export function orbit(cam, dTheta, dPhi) {
  const LIMIT = 1e-3;
  cam.theta = Math.min(Math.PI - LIMIT, Math.max(LIMIT, cam.theta + dTheta));
  cam.phi += dPhi;
}

/**
 * Zoom by changing the Boyer–Lindquist radius. Clamped above the horizon so
 * the camera can never be placed inside it, and below a generous outer bound
 * where the hole becomes a dot.
 *
 * @param {Camera} cam mutated in place
 * @param {number} factor multiplicative; > 1 zooms out
 * @param {number} a spin, used for the horizon-clamped minimum
 */
export function zoom(cam, factor, a) {
  const rMin = horizonRadius(a) + 1.5;
  const rMax = 200;
  cam.r = Math.min(rMax, Math.max(rMin, cam.r * factor));
}

/**
 * Ray direction for a pixel, in the ZAMO basis.
 *
 * @param {Camera} cam
 * @param {number} ndcX normalised device x, scaled by 1/aspect already
 * @param {number} ndcY normalised device y
 * @returns {number[]} [n_r, n_theta, n_phi], unit length
 */
export function rayDirection(cam, ndcX, ndcY) {
  const { right, up, forward } = cameraBasis(cam.theta);
  const t = Math.tan(cam.fov / 2);
  const v = [
    forward[0] + right[0] * ndcX * t + up[0] * ndcY * t,
    forward[1] + right[1] * ndcX * t + up[1] * ndcY * t,
    forward[2] + right[2] * ndcX * t + up[2] * ndcY * t,
  ];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

/**
 * Flatten the basis into the column-major mat3 the shader expects, with
 * columns (right, up, forward).
 */
export function basisMatrix(cam) {
  const { right, up, forward } = cameraBasis(cam.theta);
  return [
    right[0], right[1], right[2],
    up[0], up[1], up[2],
    forward[0], forward[1], forward[2],
  ];
}
