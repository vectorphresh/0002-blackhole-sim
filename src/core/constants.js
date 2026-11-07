/**
 * Physical constants and unit conventions.
 *
 * Throughout the project we use geometrised units:
 *   G = c = M = 1
 * so every length is expressed in units of the gravitational radius GM/c^2 and
 * every time in units of GM/c^3. A dimensionless spin is therefore a = a/M and
 * must satisfy 0 <= a <= 1.
 *
 * To convert a length L_geom (in units of M) to metres for a black hole of mass
 * mu * M_sun, use:  L_m = L_geom * GM/c^2 = L_geom * 1.4766251e-3 * mu
 */

export const G = 6.6743e-11; // m^3 kg^-1 s^-2
export const C = 299792458; // m s^-1
export const M_SUN = 1.98847e30; // kg
export const SIGMA_SB = 5.670374419e-8; // W m^-2 K^-4
export const H_PLANCK = 6.62607015e-34; // J s
export const K_BOLTZMANN = 1.380649e-23; // J K^-1

/** Gravitational radius GM/c^2 of a solar mass, in metres. */
export const GRAV_RADIUS_PER_SOLAR_MASS = (G * M_SUN) / (C * C);

/** Orbital timescale GM/c^3 of a solar mass, in seconds. */
export const TIME_PER_SOLAR_MASS = (G * M_SUN) / (C * C * C);

/**
 * Convert a geometrised length (units of M) to metres.
 * @param {number} lengthGeom
 * @param {number} massSolarMasses
 * @returns {number}
 */
export function geomLengthToMetres(lengthGeom, massSolarMasses = 1) {
  return lengthGeom * GRAV_RADIUS_PER_SOLAR_MASS * massSolarMasses;
}

/**
 * Convert a geometrised time (units of M) to seconds.
 * @param {number} timeGeom
 * @param {number} massSolarMasses
 * @returns {number}
 */
export function geomTimeToSeconds(timeGeom, massSolarMasses = 1) {
  return timeGeom * TIME_PER_SOLAR_MASS * massSolarMasses;
}

export const TWO_PI = Math.PI * 2;
export const HALF_PI = Math.PI / 2;
