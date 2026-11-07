/**
 * Simulation state: the parameters the user controls, and the physical
 * quantities derived from them.
 *
 * Everything internal is in geometrised units (G = c = M = 1). The only place
 * a mass enters is `solarMasses`, which converts to metres and seconds for the
 * HUD. Keeping the conversion in one module means the physics core never has to
 * care about kilograms.
 */

import {
  horizonRadius,
  innerHorizonRadius,
  ergosphereRadius,
  diskInnerRadius,
  circularOmega,
  frameDragging,
} from "./kerr.js";
import {
  geomLengthToMetres,
  geomTimeToSeconds,
  GRAV_RADIUS_PER_SOLAR_MASS,
  TIME_PER_SOLAR_MASS,
} from "./constants.js";
import { equatorialPhotonOrbit } from "./photonOrbit.js";

/** Default parameters. Chosen so the first frame shows a recognisable disk. */
export function defaultParams() {
  return {
    // Spacetime
    spin: 0.9,
    solarMasses: 10,

    // Disk
    diskEnabled: true,
    diskPrograde: 1, // +1 prograde, -1 retrograde
    diskInnerScale: 1.0, // multiplier on the ISCO
    diskOuter: 22.0, // in units of M
    diskTemperature: 1.0, // scales the colour ramp, not the g^4 law
    // Display gain on disk emission. Not a physics parameter: g^4 spans a
    // ~3500:1 dynamic range (measured g from 0.165 receding to 1.265
    // approaching at a = 0.9), so without a gain the receding half of the disk
    // disappears into black. Default puts the bright inner rim near 1.0.
    diskBrightness: 1.2,

    // Camera. An inclination of ~25 degrees above the disk plane shows both
    // the lensed far-side image arcing over the shadow and the full Doppler
    // asymmetry. Edge-on hides the far side behind the hole; face-on removes
    // the asymmetry that makes a relativistic disk recognisable.
    camR: 16,
    camTheta: 1.13,
    camPhi: 0,
    fovDeg: 50,

    // Integration
    step: 0.02, // Mino-time tolerance
    maxSteps: 1400,
    escapeRadius: 120,

    // Rendering
    exposure: 1.1,
    bloomStrength: 0.55,
    bloomThreshold: 0.75,
    vignette: 0.55,
    starBrightness: 1.0,

    // Performance
    renderScale: 0.75,
    accumulate: true,
    maxAccumFrames: 240,
  };
}

/**
 * @typedef {Object} Derived
 * @property {number} horizon      event horizon radius r_+
 * @property {number} innerHorizon Cauchy horizon r_-
 * @property {number} ergosphereEquator ergosphere radius in the equatorial plane
 * @property {number} isco         innermost stable circular orbit
 * @property {number} diskInner    inner disk edge actually used
 * @property {number} omegaDisk    orbital frequency at the inner edge
 * @property {number} frameDraggingAtDisk frame-dragging rate at the inner edge
 * @property {number} horizonMetres
 * @property {number} timeUnitSeconds
 * @property {number} shadowRadiusApprox
 */

/**
 * Derive every quantity the HUD and shader need from the raw parameters.
 *
 * @param {ReturnType<typeof defaultParams>} p
 * @returns {Derived}
 */
export function derive(p) {
  const a = p.spin;
  const prograde = p.diskPrograde > 0;
  const rHorizon = horizonRadius(a);
  const rIsco = diskInnerRadius(a, prograde);
  const rDiskInner = Math.max(rIsco, rHorizon + 0.25) * p.diskInnerScale;
  const rDiskOuter = Math.max(rDiskInner + 1, p.diskOuter);

  // The apparent shadow radius for a distant observer is well approximated by
  // the prograde critical impact parameter. It is a HUD readout, not an input
  // to the integrator, so an approximation is fine and stated as such.
  const bCrit = criticalImpactParameter(a);

  return {
    horizon: rHorizon,
    innerHorizon: innerHorizonRadius(a),
    ergosphereEquator: ergosphereRadius(a, Math.PI / 2),
    isco: rIsco,
    diskInner: rDiskInner,
    diskOuter: rDiskOuter,
    omegaDisk: circularOmega(rDiskInner, a, prograde),
    frameDraggingAtDisk: frameDragging(rDiskInner, Math.PI / 2, a),
    horizonMetres: geomLengthToMetres(rHorizon, p.solarMasses),
    timeUnitSeconds: geomTimeToSeconds(1, p.solarMasses),
    shadowRadiusApprox: bCrit,
  };
}

/**
 * Critical impact parameter b_crit for a Kerr hole, taken from the equatorial
 * photon orbit solved numerically in `photonOrbit.js`.
 *
 * An earlier version of this function used the closed form
 *
 *   b_crit = sqrt(27 + a^2 + 12 sqrt(3 (3 - a^2)))
 *
 * which is wrong: at a = 0 it gives sqrt(63) = 7.94 rather than 3 sqrt(3) =
 * 5.196. Reusing the numerically solved orbit, which is validated against the
 * Bardeen–Press–Teukolsky table in tests/, is both correct and cheaper than
 * trying to repair a half-remembered formula.
 *
 * The true silhouette depends on inclination; this is the edge-on prograde
 * value, labelled as an approximation in the HUD.
 *
 * @param {number} a
 */
export function criticalImpactParameter(a) {
  return Math.abs(equatorialPhotonOrbit(a, true).impactParameter);
}

/* ------------------------------------------------------------------ *
 * Unit formatting for the HUD
 * ------------------------------------------------------------------ */

const SUFFIXES = [
  [1e15, "P"],
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
  [1e-3, "m"],
  [1e-6, "µ"],
  [1e-9, "n"],
];

/** Format an SI quantity with a metric prefix. */
export function si(value, unit = "", digits = 3) {
  if (!Number.isFinite(value)) return "—";
  const mag = Math.abs(value);
  if (mag === 0) return `0 ${unit}`;
  for (const [scale, prefix] of SUFFIXES) {
    if (mag >= scale) {
      return `${(value / scale).toPrecision(digits)} ${prefix}${unit}`;
    }
  }
  return `${value.toExponential(digits)} ${unit}`;
}

/** Format a duration given in seconds into the most readable unit. */
export function duration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  const s = Math.abs(seconds);
  if (s < 1e-9) return `${(seconds * 1e12).toPrecision(3)} ps`;
  if (s < 1e-6) return `${(seconds * 1e9).toPrecision(3)} ns`;
  if (s < 1e-3) return `${(seconds * 1e6).toPrecision(3)} µs`;
  if (s < 1) return `${(seconds * 1e3).toPrecision(3)} ms`;
  if (s < 60) return `${seconds.toPrecision(3)} s`;
  if (s < 3600) return `${(seconds / 60).toPrecision(3)} min`;
  if (s < 86400) return `${(seconds / 3600).toPrecision(3)} h`;
  if (s < 31557600) return `${(seconds / 86400).toPrecision(3)} days`;
  return `${(seconds / 31557600).toPrecision(3)} yr`;
}

/**
 * Rows for the HUD: label, value. Everything here is derived, never typed in,
 * so the panel cannot drift out of sync with the simulation.
 *
 * @param {ReturnType<typeof defaultParams>} p
 * @param {Derived} d
 */
export function hudRows(p, d) {
  const mass = p.solarMasses;
  return [
    ["Black hole mass", `${mass.toPrecision(4)} M☉`],
    ["Gravitational radius", si(GRAV_RADIUS_PER_SOLAR_MASS * mass, "m")],
    ["Time unit", duration(TIME_PER_SOLAR_MASS * mass)],
    ["Spin a / J", `${p.spin.toFixed(3)}  (${(p.spin * mass * mass).toPrecision(3)} M²)`],
    ["Event horizon r₊", `${d.horizon.toFixed(4)} M = ${si(d.horizonMetres, "m")}`],
    ["Cauchy horizon r₋", `${d.innerHorizon.toFixed(4)} M`],
    ["Ergosphere (equator)", `${d.ergosphereEquator.toFixed(4)} M`],
    ["ISCO", `${d.isco.toFixed(4)} M`],
    ["Disk inner edge", `${d.diskInner.toFixed(4)} M`],
    ["Disk outer edge", `${d.diskOuter.toFixed(2)} M`],
    ["Orbit freq at rim", `${(d.omegaDisk * d.timeUnitSeconds > 0 ? 1 / (d.omegaDisk * d.timeUnitSeconds) : 0).toPrecision(3)} Hz`],
    ["Frame dragging at rim", `${(d.frameDraggingAtDisk / d.timeUnitSeconds).toPrecision(3)} rad/s`],
    ["Shadow radius ≈", `${d.shadowRadiusApprox.toFixed(3)} M`],
    ["Camera r", `${p.camR.toFixed(3)} M = ${si(geomLengthToMetres(p.camR, mass), "m")}`],
    // Report the polar angle itself, matching the control panel. Inclination
    // measured from the disk plane is the more intuitive phrasing, so it is
    // given alongside rather than replacing theta.
    ["Camera polar angle θ", `${((p.camTheta * 180) / Math.PI).toFixed(1)}°`],
    ["Above disk plane", `${(((Math.PI / 2 - p.camTheta) * 180) / Math.PI).toFixed(1)}°`],
  ];
}
