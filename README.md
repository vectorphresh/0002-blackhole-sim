# Kerr Black Hole — Null Geodesic Ray Tracer

A real-time renderer for the accretion disk of a rotating (Kerr) black hole.
Every pixel is an integrated photon path: rays are traced **backwards from the
camera** along null geodesics of the Kerr metric until they either cross the
event horizon or intersect the accretion disk, and disk pixels are shaded with
the general-relativistic redshift factor so that Doppler beaming and
gravitational redshift appear without any faking.

No build step, no dependencies, no framework. WebGL2 in the browser for the
renderer, plain ES modules for the physics, `node:test` for the tests.

```
npm install      # nothing to install, but sets up node_modules/.bin if you like
npm run serve    # http://127.0.0.1:8080
npm test         # 68 physics tests
npm run validate # numerical report: measured vs reference vs error
```

Requires a browser with WebGL2 and the `EXT_color_buffer_float` extension
(Chrome, Edge, Firefox, Safari 16.4+). Node 20+ for the tests and dev server.

---

## What you are looking at

Three relativistic effects dominate the image, and all three are computed rather
than stylised:

- **The shadow.** Rays that cross the event horizon return black. Its outline is
  not the horizon — it is the silhouette of the photon capture region, which is
  why it is larger than the horizon and why it is D-shaped rather than circular
  when the hole spins.
- **Gravitational lensing of the far side of the disk.** Light from the part of
  the disk behind the hole bends over the top and under the bottom, producing the
  thin arc above the shadow. That arc is the *same physical disk* seen twice.
- **Relativistic beaming.** The approaching side is several times brighter than
  the receding side. This is not a lighting trick: it is `g⁴`, where `g` is the
  ratio of observed to emitted photon energy, computed from the emitter's orbital
  velocity and the local lapse. Flip **Prograde** off and the asymmetry mirrors.

The default view sits ~25° above the disk plane because that inclination shows
both the lensed far-side arc and the full beaming asymmetry. Edge-on hides the
arc behind the hole; face-on removes the asymmetry that makes a relativistic disk
recognisable.

## Controls

| Input | Action |
| --- | --- |
| Drag | Orbit the camera (polar angle θ, azimuth φ) |
| Wheel | Zoom (clamped to stay outside the near-horizon region) |
| Arrow keys | Nudge; hold <kbd>Shift</kbd> for larger steps |
| <kbd>+</kbd> / <kbd>-</kbd> | Zoom in / out |
| <kbd>R</kbd> | Reset the view |
| <kbd>S</kbd> | Save a PNG snapshot |

The panel's **Record** button captures a WebM of whatever you do next; press it
again to stop and download.

### The panel groups

**Spacetime** — spin `a/M` (0 = Schwarzschild, 0.998 = near-extremal) and mass in
solar masses. Mass is *only* a unit conversion: the entire simulation runs in
geometrised units `G = c = M = 1`, so changing the mass changes the HUD's metres
and seconds and nothing about the traced geometry.

**Accretion disk** — inner edge as a multiple of the ISCO, outer edge, colour
temperature, and a display gain. **Prograde** reverses the disk's orbital
direction, which is the cleanest way to confirm the beaming is physical: the
bright side flips to the opposite limb.

**Camera** — radius, polar angle, azimuth, field of view.

**Integration** — step tolerance, max steps per ray, escape radius. Lower the
step tolerance to sharpen the photon ring; raise max steps if you push the camera
close and rays start terminating early.

**Image** — exposure, bloom, vignette. These affect only the post-processing pass,
so adjusting them does **not** discard accumulated history. Everything in
Spacetime/Disk/Camera/Integration does.

**Performance** — render scale and progressive accumulation.

### Progressive accumulation

Each frame adds one jittered sample to a running average held in a float
framebuffer. The image starts noisy and converges while you are not moving; any
change to the traced scene resets it. This is what makes the photon ring — a
feature a few pixels wide that a single frame cannot resolve — visible at all.
Turn **Progressive** off to trade image quality for frame rate.

---

## Reading the code

The design rule that shapes everything: **the physics lives in tested JavaScript,
and the shader is a port of it.** Every formula in the GLSL has a
counterpart in `src/core/` with a test asserting the same number. When the image
looks wrong, the fix is nearly always found by running the tests, not by staring
at the shader.

```
src/core/       pure physics, no DOM, no WebGL — all 62 tests live here
  constants.js    SI constants and geometrised-unit conversions
  kerr.js         metric, inverse metric, horizon, ergosphere, ISCO, circular orbits
  geodesics.js    TWO independent geodesic integrators + ZAMO tetrad
  mino.js         Mino-time adaptive Verlet integrator (the one the shader ports)
  photonOrbit.js  photon sphere and critical impact parameter, solved numerically
  redshift.js     redshift factor and disk temperature profile
  camera.js       orbit camera in Boyer–Lindquist coordinates
  state.js        user parameters, derived quantities, HUD formatting

src/gl/
  shaders.js      GLSL ES 3.0 as JS template literals: trace pass + post pass
  renderer.js     WebGL2 setup, RGBA16F ping-pong targets, accumulation
  capture.js      PNG snapshots and WebM recording

src/ui/panel.js   declarative control panel (layout is data, not markup)
src/app.js        wiring; exposes `window.kerr` for probing from the console
serve.js          dependency-free static file server
scripts/          diagnostics, e.g. measuring the beaming asymmetry
```

### Why two integrators

`geodesics.js` integrates null geodesics twice, by methods that share no algebra:

- `integrateHamilton` — RK4 on the Hamiltonian `½ g^{μν} p_μ p_ν`, needing only
  the inverse metric and finite-differenced metric derivatives. Slow, and about
  as independent of my derivation as it is possible to be.
- `integrateCarter` / `mino.js` — uses Carter's separation into constants of
  motion `E`, `Lz`, `Q`. Fast, and the basis of the shader.

Because they share no algebra, a sign error or a wrong Carter constant makes them
disagree immediately rather than silently producing a plausible-looking image.
`tests/geodesics.test.js` asserts they agree to 1e-3 on closest approach and 5e-3
on the disk-crossing events the renderer actually reads.

### Probing from the browser console

`window.kerr` exposes the live simulation:

```js
kerr.horizonRadius(0.998)                       // 1.001 — the horizon barely moves
kerr.horizonRadius(0)                           // 2 — Schwarzschild
kerr.rayDirection(kerr.camera, 0, 0)            // [-1, 0, 0] — straight at the hole
kerr.derive(kerr.params)                        // ISCO, ergosphere, horizon in metres
kerr.params.spin = 0.998;                       // mutate, then nudge a slider to apply
```

## Tests

```
tests/kerr.test.js         15   metric symmetries, a=0 limits, ISCO vs Bardeen–Press–Teukolsky
tests/geodesics.test.js    13   the two integrators agree; Carter's constant is conserved
tests/mino.test.js          6   closed-form accelerations, adaptive step, turning points
tests/photonOrbit.test.js     6   photon orbit and b_crit against the BPT table
tests/redshift.test.js     11   Schwarzschild limits, mirror symmetry, light surface
tests/camera.test.js       11   orthonormal ZAMO basis, FOV geometry, null momentum
tests/shaderParity.test.js   6   the GLSL says the same thing as the tested JS
```

The tests deliberately avoid asserting memorised numbers where possible. The
recurring arbiter is substitution: take a closed-form `(E, Lz)`, plug it into the
radial potential `R(r)`, and assert `R(r) = 0` **and** `R'(r) = 0`. That validates
a formula against the geodesic equation itself, with no solver that can misbehave
and no result to misremember. Several real bugs were caught this way — see
[docs/PHYSICS.md](docs/PHYSICS.md#validation).

`tests/shaderParity.test.js` is the one that keeps the renderer honest: it reads
the GLSL as text and asserts every formula with a tested JS counterpart appears
in the same algebraic form. It exists because a shader once computed the
retrograde orbital velocity with the prograde denominator — a bug invisible at
`a = 0`, so neither a physics test nor a visual check caught it.

## Numerical validation

`npm run validate` prints a report rather than a pass/fail count. The distinction
matters for a physics core: a test that passes at 1e-3 hides whether you are at
1e-9 or 9e-4, and that difference is the difference between *correct* and *correct
by luck with a loose tolerance*. Every line prints the measured value, an
independently known reference, and both the absolute and relative error.

```
104 numeric checks, 104 within tolerance, 0 failed
```

The references are all external to this codebase — the Bardeen–Press–Teukolsk
(1973) photon-orbit table, closed-form Schwarzschild limits, the post-Newtonian
deflection series — so nothing is checked against our own output. It exits
non-zero on failure, so it doubles as a CI gate.

Two sections are worth knowing about, because the obvious way to write them is
wrong:

- **Order of convergence.** `rPeri` is a `min` over discrete samples, so it is a
  non-smooth functional; `|dphi|` is summed up to a step-dependent termination
  event. Both give nonsense orders — RK4 measured an apparent 1.0 and even −17.7.
  The report instead tracks the ingoing Eddington–Finkelstein advanced time
  `v = t + r*(r)`, which is *exactly* constant along a radially ingoing photon.
  That gives Verlet 2.01, RK4 4.43, Carter 1.00.
- **Captured rays.** For a ray that falls in, `rPeri` is only how far the final
  step overshot the termination radius — pure step-size noise. The report prints
  it but asserts only the classification (captured vs escaped), because that is
  the part that is physics.

The output is deterministic — identical byte-for-byte across runs — so a diff
between two reports is always a real change in the physics, never noise. A
captured report from the current tree is at
[docs/VALIDATION-REPORT.txt](docs/VALIDATION-REPORT.txt).

## Known limits

- The disk is a thin equatorial sheet of prescribed temperature, not a radiative
  transfer solution. It is optically thin and emits `g⁴ × B(T)` with no
  self-occultation: a ray that has already passed through the disk can light up
  again on the far side.
- No back-reaction, no plasma, no magnetic field. The metric is vacuum Kerr.
- The HUD's shadow radius is the edge-on prograde critical impact parameter,
  labelled as an approximation. The true silhouette depends on inclination.
- Rays terminate at `maxSteps`. Near the photon sphere the deflection diverges
  logarithmically, so the photon ring is always truncated — longer step budgets
  make it thinner, not complete.

## Further reading

- [docs/PHYSICS.md](docs/PHYSICS.md) — the derivations, conventions, and the
  checks that caught the bugs.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — why the physics lives in tested
  JavaScript and the shader is a port of it.
- Bardeen, Press & Teukolsky, *Rotating black holes: Separable wave equations,
  null geodesics, and shadows*, ApJ 179 (1973).
- Luminet, *Image of a Spherical Accretion Disk around a Schwarzschild Black
  Hole*, A&A 128 (1979).
- Ruchlin, James & Heil, *GravLens: A library for ray-tracing in general
  relativity*, Computer Physics Communications (2023).

## License

MIT — see [LICENSE](LICENSE). The project has no third-party dependencies, so
nothing here carries an inherited obligation: the physics is derived from the
open literature and validated against it rather than ported from another
implementation. The papers above are cited for academic credit, not because any
of them is linked or bundled.

