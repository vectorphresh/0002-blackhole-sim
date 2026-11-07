/**
 * Shader / physics-module parity.
 *
 * The renderer's correctness argument is that the GLSL is a *port* of the
 * tested JavaScript in `src/core/`. That argument is only worth anything if
 * something enforces it. Without these checks the two can drift silently, and
 * the drift shows up as a subtly wrong image rather than a failing test — which
 * is exactly how a real bug got through: the shader computed the retrograde
 * orbital velocity as `dir / (r^1.5 + a)` instead of `dir / (r^1.5 + dir*a)`,
 * giving a retrograde disk the prograde denominator. At `a = 0` the two agree,
 * so no visual check and no physics test noticed.
 *
 * These tests read the shader source as text and assert that each formula that
 * has a tested JS counterpart appears in the shader in the same algebraic form.
 * They are deliberately pattern-based rather than evaluative: they cannot tell
 * you the physics is right, only that the shader and the JS say the same thing.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { TRACE } from "../src/gl/shaders.js";
import { orbitalOmega, redshiftFactor, diskTemperature } from "../src/core/redshift.js";
import { metric, idx, T, PH } from "../src/core/kerr.js";

/** Strip comments so prose cannot satisfy a pattern. */
function code(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
}

const body = code(TRACE);

/** The statements of a named function, up to its closing brace. */
function fnBody(name) {
  const start = body.indexOf(`${name}(`);
  assert.ok(start >= 0, `shader function ${name} not found`);
  const open = body.indexOf("{", start);
  let depth = 0;
  for (let i = open; i < body.length; i++) {
    if (body[i] === "{") depth++;
    else if (body[i] === "}" && --depth === 0) return body.slice(open, i);
  }
  throw new Error(`unterminated function ${name}`);
}

const emission = fnBody("diskEmission");

test("shader orbital velocity carries the direction sign on the spin term", () => {
  // Omega = dir / (r^(3/2) + dir*a). The `dir *` before `a` is the whole test.
  const m = emission.match(/omega\s*=\s*([^;]+);/);
  assert.ok(m, "no omega assignment in diskEmission");
  const rhs = m[1].replace(/\s+/g, "");
  assert.ok(
    /dir\/\(r\*sqrt\(r\)\+dir\*a\)/.test(rhs),
    `omega must be dir/(r^1.5 + dir*a) to match orbitalOmega(); shader has "${m[1]}"`,
  );

  // And the JS side, so the pattern is pinned to a real formula rather than to
  // whatever the shader happens to say today.
  assert.equal(orbitalOmega(8, 0.9, -1), -1 / (8 * Math.sqrt(8) - 0.9));
  assert.equal(orbitalOmega(8, 0.9, 1), 1 / (8 * Math.sqrt(8) + 0.9));
});

test("shader redshift is the reciprocal of u^t(1 - Omega xi), not the product", () => {
  // g = 1 / [u^t (1 - Omega xi)]. The inverted form reports gravitational
  // redshift as blueshift and looks plausible, so it must be pinned textually.
  const m = emission.match(/float\s+g\s*=\s*([^;]+);/);
  assert.ok(m, "no g assignment in diskEmission");
  const rhs = m[1].replace(/\s+/g, "");
  assert.ok(
    rhs === "1.0/denom2" || rhs === "1.0/(ut*(1.0-omega*xi))",
    `g must be the reciprocal of u^t(1 - Omega xi); shader has "${m[1]}"`,
  );

  // Arbiter: a static Schwarzschild emitter must REDSHIFT.
  assert.ok(redshiftFactor(10, 0, 0, 0) < 1, "static emitter must be redshifted");
});

test("shader equatorial metric components match kerr.js", () => {
  // At theta = pi/2, Sigma = r^2, so the equatorial components are:
  //   g_tt = -1 + 2/r,  g_tphi = -2a/r,  g_phiphi = r^2 + a^2 + 2a^2/r
  const expected = [
    { name: "gtt", re: /gtt\s*=\s*-1\.0\s*\+\s*2\.0\s*\/\s*r/ },
    { name: "gtphi", re: /gtphi\s*=\s*-2\.0\s*\*\s*a\s*\/\s*r/ },
    { name: "gpp", re: /gpp\s*=\s*r\s*\*\s*r\s*\+\s*a\s*\*\s*a\s*\+\s*2\.0\s*\*\s*a\s*\*\s*a\s*\/\s*r/ },
  ];
  for (const { name, re } of expected) {
    assert.ok(re.test(emission), `shader ${name} does not match the Kerr equatorial metric`);
  }

  // Cross-check the same components against the general metric function.
  const r = 7.5;
  const a = 0.8;
  const g = metric(r, Math.PI / 2, a);
  assert.ok(Math.abs(g[idx(T, T)] - (-1 + 2 / r)) < 1e-12, "g_tt");
  assert.ok(Math.abs(g[idx(T, PH)] - (-2 * a / r)) < 1e-12, "g_tphi");
  assert.ok(
    Math.abs(g[idx(PH, PH)] - (r * r + a * a + (2 * a * a) / r)) < 1e-12,
    "g_phiphi",
  );
});

test("shader temperature profile matches diskTemperature()", () => {
  // T = x^-0.75 (1 - (rIn/r)^0.75)^0.25 with x = r/rIn.
  const xPow = emission.match(/temp\s*=\s*([^;]+);/);
  assert.ok(xPow, "no temp assignment in diskEmission");
  const rhs = xPow[1].replace(/\s+/g, "");
  assert.ok(
    /pow\(x,-0\.75\)\*pow\(f,0\.25\)/.test(rhs),
    `temp must be x^-0.75 * f^0.25 to match diskTemperature(); shader has "${xPow[1]}"`,
  );
  assert.ok(
    /pow\(uDiskInner\s*\/\s*r,\s*0\.75\)/.test(emission),
    "flux factor must be 1 - (rIn/r)^0.75",
  );

  // JS side: vanishes at the inner edge, positive outside it.
  assert.equal(diskTemperature(3, 3), 0);
  assert.ok(diskTemperature(6, 3) > 0);
});

test("shader emission applies the g^4 beaming factor", () => {
  // g^3 from specific intensity plus g from bandwidth. Without the exponent the
  // disk is still one-sided (from the colour ramp) but far too weakly so.
  const m = emission.match(/intensity\s*=\s*([^;]+);/);
  assert.ok(m, "no intensity assignment in diskEmission");
  assert.ok(
    /pow\(g,\s*4\.0\)/.test(m[1]),
    `intensity must scale as g^4; shader has "${m[1]}"`,
  );
});

test("shader terminates rays at the horizon, not at r = 0", () => {
  // A ray that reaches r = 0 has already passed through the horizon; testing
  // for r <= 0 instead would let the integrator run on into the ring
  // singularity and produce garbage where black belongs.
  assert.ok(
    /r\s*<=\s*rHorizon/.test(body),
    "trace loop must break on the horizon radius, not on r <= 0",
  );
  assert.ok(/rHorizon\s*=\s*1\.0\s*\+\s*sqrt/.test(body), "rHorizon must be 1 + sqrt(1 - a^2)");
});
