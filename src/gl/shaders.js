/**
 * GLSL ES 3.0 shader sources.
 *
 * The trace shader is a direct port of `src/core/mino.js`. That is deliberate:
 * the JavaScript integrator is cross-validated against an independent
 * metric-only integrator in `tests/`, so the shader inherits that validation
 * for as long as the two stay in step. Keep the maths identical — same
 * potentials, same accelerations, same adaptive step controller.
 */

/** Constants and the Kerr/Carter primitives shared by every pass. */
const COMMON = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

#define PI 3.141592653589793
#define SIN2_FLOOR 1e-12
#define TINY 1e-30
#define H_MIN 1e-9
#define H_MAX 0.25

// uSpin         dimensionless Kerr spin a = Jc/(GM^2), 0 <= a < 1
// uEscapeRadius Boyer-Lindquist radius beyond which a ray has left the scene
// uDiskInner    inner disk edge (the ISCO)
// uDiskOuter    outer disk edge
// uDiskEnabled  0 disables disk emission entirely
// uDiskTemp     colour-temperature multiplier; scales the ramp, not the g^4 law
// uDiskGain     display gain on disk emission (see state.js: g^4 is HDR by nature)
// uStep         Mino-time step tolerance; smaller is more accurate
uniform float uSpin;
uniform float uEscapeRadius;
uniform float uDiskInner;
uniform float uDiskOuter;
uniform float uDiskEnabled;
uniform float uDiskTemp;
uniform float uDiskGain;
uniform float uStep;

/** Delta = r^2 - 2r + a^2; vanishes at the horizons. */
float deltaOf(float r) {
  return r * r - 2.0 * r + uSpin * uSpin;
}

/**
 * Radial potential R(r) of the separated null geodesic equation.
 *   P = E(r^2 + a^2) - a Lz,   K = (Lz - aE)^2 + Q
 *   R = P^2 - Delta K
 */
float radialPotential(float r, float e, float lz, float q) {
  float a = uSpin;
  float p = e * (r * r + a * a) - a * lz;
  float k = (lz - a * e) * (lz - a * e) + q;
  return p * p - deltaOf(r) * k;
}

/** Polar potential Theta = Q + a^2 E^2 cos^2 - Lz^2 cos^2 / sin^2. */
float polarPotential(float th, float e, float lz, float q) {
  float a = uSpin;
  float c = cos(th);
  float s2 = max(sin(th) * sin(th), SIN2_FLOOR);
  float c2 = c * c;
  return q + a * a * e * e * c2 - lz * lz * c2 / s2;
}

/**
 * (1/2) dR/dr — the radial acceleration in Mino time.
 *   dR/dr = 2 P P' - Delta' K,   P' = 2Er,   Delta' = 2r - 2
 * Closed form rather than finite-differenced: cheaper, and it avoids
 * cancelling two quantities of order E^2 r^4.
 */
float radialAccel(float r, float e, float lz, float q) {
  float a = uSpin;
  float p = e * (r * r + a * a) - a * lz;
  float k = (lz - a * e) * (lz - a * e) + q;
  return 2.0 * e * r * p - (r - 1.0) * k;
}

/**
 * (1/2) dTheta/dtheta — the polar acceleration in Mino time.
 *   dTheta/dtheta = -2 a^2 E^2 cos sin + 2 Lz^2 cos / sin^3
 */
float polarAccel(float th, float e, float lz) {
  float a = uSpin;
  float s = sin(th);
  float c = cos(th);
  float s2 = max(s * s, SIN2_FLOOR);
  return -a * a * e * e * c * s + lz * lz * c / (s2 * s);
}

/** dt/dlambda_M = (A E - 2 a r Lz) / Delta. */
float dtDlambdaM(float r, float th, float e, float lz) {
  float a = uSpin;
  float a2 = a * a;
  float r2 = r * r;
  float d = deltaOf(r);
  float s2 = max(sin(th) * sin(th), SIN2_FLOOR);
  float A = (r2 + a2) * (r2 + a2) - a2 * d * s2;
  return (A * e - 2.0 * a * r * lz) / d;
}

/** dphi/dlambda_M = 2arE/Delta + Lz (1/sin^2 - a^2/Delta). */
float dphiDlambdaM(float r, float th, float e, float lz) {
  float a = uSpin;
  float a2 = a * a;
  float d = deltaOf(r);
  float s2 = max(sin(th) * sin(th), SIN2_FLOOR);
  return (2.0 * a * r * e) / d + lz * (1.0 / s2 - a2 / d);
}

/**
 * Adaptive Mino-time step.
 *
 * A fixed step is unusable: dr/dlambda_M = sqrt(R) ~ E r^2 grows without
 * bound, so a step that resolves the near-horizon region leaps thousands of
 * units at the launch radius — the ray sails straight past the hole. Each
 * limit is linear in the tolerance, so halving it halves the step
 * everywhere and preserves Verlet's second-order convergence.
 */
float adaptiveStep(float r, float th, float e, float lz, float q, float tol) {
  float vr = sqrt(max(0.0, radialPotential(r, e, lz, q)));
  float vt = sqrt(max(0.0, polarPotential(th, e, lz, q)));
  float ar = max(abs(radialAccel(r, e, lz, q)), TINY);
  float at = max(abs(polarAccel(th, e, lz)), TINY);
  float h = min(r / max(vr, TINY), 1.0 / max(vt, TINY));
  h = min(h, sqrt(r / ar));
  h = min(h, sqrt(1.0 / at));
  return clamp(tol * h, H_MIN, H_MAX);
}

/* ------------------------------------------------------------------ *
 * Boyer-Lindquist metric components.
 *
 *   Sigma    = r^2 + a^2 cos^2
 *   A        = (r^2 + a^2)^2 - a^2 Delta sin^2
 *   g_tt     = -1 + 2r / Sigma
 *   g_tphi   = -2 a r sin^2 / Sigma
 *   g_phiphi = A sin^2 / Sigma
 *   g_rr     = Sigma / Delta
 *   g_thth   = Sigma
 * ------------------------------------------------------------------ */

/** A = (r^2 + a^2)^2 - a^2 Delta sin^2. */
float aOf(float r, float s2) {
  float a = uSpin;
  float a2 = a * a;
  float r2 = r * r;
  return (r2 + a2) * (r2 + a2) - a2 * deltaOf(r) * s2;
}

/**
 * Pack the metric components we need into a vec4:
 *   x = g_tt, y = g_tphi, z = g_phiphi, w = Sigma
 */
vec4 metricComponents(float r, float th) {
  float a = uSpin;
  float a2 = a * a;
  float c = cos(th);
  float s2 = max(sin(th) * sin(th), SIN2_FLOOR);
  float sigma = r * r + a2 * c * c;
  float gtphi = -2.0 * a * r * s2 / sigma;
  float gpp = aOf(r, s2) * s2 / sigma;
  return vec4(-1.0 + 2.0 * r / sigma, gtphi, gpp, sigma);
}
`;

/**
 * Fullscreen triangle. No vertex buffer: the two required vertices are
 * derived from gl_VertexID.
 */
const VERTEX = /* glsl */ `#version 300 es
out vec2 vUv;

void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * Ray tracing pass: one null geodesic per pixel, integrated with
 * velocity-Verlet in Mino time, progressively accumulated into an HDR target.
 */
const TRACE = COMMON + /* glsl */ `
out vec4 outColor;

uniform vec3 uCamPos;      // camera position, Boyer-Lindquist (r, theta, phi)
uniform mat3 uCamBasis;    // columns: right, up, forward in the ZAMO frame
uniform vec2 uResolution;  // render-target size in pixels
uniform float uTanHalfFov;
uniform float uAccumIndex; // frames accumulated so far, 0 on the first
uniform float uStarBrightness;
uniform int uMaxSteps;
uniform int uDiskPrograde; // +1 prograde, -1 retrograde
uniform sampler2D uPrevAccum;

/**
 * Procedural star field on the unit sphere. Hash-based rather than textured so
 * the field is resolution independent and needs no asset.
 */
float starField(vec3 dir) {
  if (uStarBrightness <= 0.0) return 0.0;
  vec3 p = dir * 160.0;
  vec3 ip = floor(p);
  vec3 fp = fract(p);
  float h = fract(sin(dot(ip, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
  if (h <= 0.962) return 0.0;

  vec3 c = vec3(
    fract(sin(dot(ip, vec3(39.347, 11.136, 82.613))) * 24634.6345),
    fract(sin(dot(ip, vec3(71.297, 53.211, 17.893))) * 13245.1235),
    fract(sin(dot(ip, vec3(23.117, 91.431, 62.729))) * 76512.3451)
  );
  vec3 jitter = 0.5 + 0.35 * (c * 2.0 - 1.0);
  float d = length(fp - jitter);
  float mag = fract(sin(dot(ip, vec3(5.321, 7.913, 3.147))) * 12345.6789);

  float core = exp(-d * d * 300.0);
  float halo = exp(-d * d * 24.0) * 0.16;
  vec3 tint = mix(vec3(1.0, 0.70, 0.42), vec3(0.70, 0.83, 1.0), mag);
  return (core + halo) * (0.30 + 1.7 * mag * mag) * dot(tint, vec3(0.333));
}

/**
 * Map a local orthonormal direction (r_hat, theta_hat, phi_hat) at spherical
 * point (theta, phi) onto a fixed Cartesian sky frame. This keeps the star
 * field anchored while the camera orbits, yet still lets it be lensed.
 */
vec3 skyDirection(vec3 n, float th, float phi) {
  float st = sin(th), ct = cos(th);
  float sp = sin(phi), cp = cos(phi);
  return vec3(
    n.x * st * cp + n.y * ct * cp - n.z * sp,
    n.x * st * sp + n.y * ct * sp + n.z * cp,
    n.x * ct - n.y * st
  );
}

/**
 * Map an observed colour temperature to RGB.
 *
 * The observed temperature is g * T_emitted, and for a near-edge-on view of a
 * fast orbit g runs from ~0.8 (receding, redshifted) to ~6 (approaching,
 * blueshifted) — measured, not assumed. The stops below span that range so the
 * Doppler colour asymmetry is visible across the disk rather than clamping to
 * white. This is a display mapping, not a Planck integral; the physical
 * brightness is carried entirely by the g^4 factor.
 */
vec3 temperatureColor(float temp) {
  vec3 deep  = vec3(0.85, 0.22, 0.05); // strongly redshifted outer/receding disk
  vec3 amber = vec3(1.00, 0.55, 0.18);
  vec3 warm  = vec3(1.00, 0.82, 0.55);
  vec3 white = vec3(0.92, 0.94, 1.00);
  vec3 blue  = vec3(0.62, 0.78, 1.00); // strongly blueshifted approaching side

  if (temp < 0.8) return mix(deep, amber, temp / 0.8);
  if (temp < 1.6) return mix(amber, warm, (temp - 0.8) / 0.8);
  if (temp < 3.0) return mix(warm, white, (temp - 1.6) / 1.4);
  return mix(white, blue, clamp((temp - 3.0) / 3.0, 0.0, 1.0));
}

/**
 * Thin-disk emission at one equatorial crossing.
 *
 * The emitter is a circular geodesic with angular velocity
 *
 *   Omega = +/- 1 / (r^(3/2) + a)
 *
 * (prograde/retrograde). The photon energy measured in the emitter frame is
 *
 *   E_em = -u^mu p_mu = u^t (E - Omega Lz)
 *
 * with u^t = (-g_tt - 2 g_tphi Omega - g_phiphi Omega^2)^(-1/2). The observer
 * at infinity has u^mu = (1,0,0,0), so measures E_obs = E. The total redshift
 * factor is therefore
 *
 *   g = E_obs / E_em = 1 / [ u^t (1 - Omega xi) ],   xi = Lz / E
 *
 * which folds gravitational redshift, transverse Doppler, and line-of-sight
 * Doppler into one factor. Getting this inverted is a subtle and very visible
 * mistake: u^t (1 - Omega xi) alone is > 1 for a static emitter, which would
 * report light climbing out of the potential well as *blueshifted*. The
 * Schwarzschild static limit is the check — at Omega = 0,
 *
 *   g = 1/u^t = sqrt(1 - 2/r) < 1,
 *
 * the expected redshift.
 *
 * A blackbody observed through g scales as g^4 — g^3 from specific intensity
 * and g from bandwidth — so relativistic beaming appears as a strong one-sided
 * brightening with no extra tuning.
 */
vec3 diskEmission(float r, float e, float lz, out float alphaOut) {
  alphaOut = 0.0;
  if (uDiskEnabled < 0.5) return vec3(0.0);
  if (r < uDiskInner || r > uDiskOuter) return vec3(0.0);

  float a = uSpin;
  // Circular geodesic angular velocity, matching orbitalOmega() in
  // core/redshift.js: Omega = dir / (r^(3/2) + dir*a). The spin term must carry
  // the direction sign. Writing dir / (r^(3/2) + a) instead — which is what this
  // line originally did — gives the retrograde disk the PROGRADE denominator,
  // understating the retrograde orbital speed and so the beaming asymmetry.
  // Invisible at a = 0, which is exactly why a visual check never catches it.
  float dir = float(uDiskPrograde);
  float omega = dir / (r * sqrt(r) + dir * a);

  // Equatorial metric: Sigma = r^2 at theta = pi/2.
  float gtt = -1.0 + 2.0 / r;
  float gtphi = -2.0 * a / r;
  float gpp = r * r + a * a + 2.0 * a * a / r;

  float denom = -(gtt + 2.0 * gtphi * omega + gpp * omega * omega);
  if (denom <= 0.0) return vec3(0.0); // inside the light surface: no circular orbit
  float ut = 1.0 / sqrt(denom);

  float xi = lz / e;
  float denom2 = ut * (1.0 - omega * xi);
  if (denom2 <= 1e-6) return vec3(0.0); // emitted against the flow: infinitely redshifted
  float g = 1.0 / denom2;

  // Shakura-Sunyaev effective temperature, T^4 ~ (1 - sqrt(r_in/r)) / r^3.
  // Peak is at r/r_in = (5/4)^(4/3) with value (4/5)*5^(-1/4) = 0.535, not 1;
  // uDiskTemp and the display gain absorb the scale. Truncated at the ISCO
  // rather than extrapolated.
  float x = r / uDiskInner;
  float f = max(0.0, 1.0 - pow(uDiskInner / r, 0.75));
  float temp = pow(x, -0.75) * pow(f, 0.25) * uDiskTemp;

  // Optical depth: dense inward, thinning outward. Combined with the g^4
  // weighting this gives the bright inner rim and dim outer disk.
  float tau = 3.5 * pow(uDiskInner / r, 1.2);
  float alpha = clamp(tau, 0.0, 1.0);

  // A blackbody's bolometric flux goes as T^4, so the temperature knob brightens
  // as well as recolouring. The relativistic g^4 factor is applied on top, then
  // scaled by the display gain: g^4 alone spans a wide dynamic range, so the raw
  // value would saturate the approaching side of the disk.
  float intensity = pow(g, 4.0) * temp * alpha * pow(uDiskTemp, 3.0) * uDiskGain;
  return temperatureColor(temp * g) * intensity;
}

void main() {
  vec2 res = uResolution;
  vec2 ndc = (gl_FragCoord.xy - 0.5 * res) / res.y * 2.0;

  vec3 dir = normalize(uCamBasis[2]
                     + uCamBasis[0] * ndc.x * uTanHalfFov
                     + uCamBasis[1] * ndc.y * uTanHalfFov);

  float a = uSpin;
  float a2 = a * a;
  float r0 = uCamPos.x;
  float th0 = uCamPos.y;

  // ZAMO (locally non-rotating) orthonormal frame at the camera. Corotating
  // with the local inertial frame is the natural choice for an observer
  // hovering at fixed (r, theta, phi).
  float s02 = max(sin(th0) * sin(th0), SIN2_FLOOR);
  vec4 mc = metricComponents(r0, th0);
  float gtt = mc.x, gtphi = mc.y, gpp = mc.z, sigma0 = mc.w;
  float d0 = deltaOf(r0);
  float alpha = sqrt(max(d0 * sigma0 / aOf(r0, s02), 1e-12));
  float betaPhi = gtphi / gpp;

  // Contravariant photon momentum p = E (e_hat_t + n^i e_hat_i) is null by
  // construction when |n| = 1; lowering the index settles the sign conventions.
  float e = 1.0;
  float pUpT = e / alpha;
  float pUpPh = -betaPhi * pUpT + dir.z / sqrt(gpp);
  float pUpR = dir.x / sqrt(sigma0 / max(d0, 1e-6));
  float pUpTh = dir.y / sqrt(sigma0);

  float pt = gtt * pUpT + gtphi * pUpPh;
  float pphi = gtphi * pUpT + gpp * pUpPh;
  float pr = (sigma0 / max(d0, 1e-6)) * pUpR;
  float pth = sigma0 * pUpTh;

  // Constants of motion: E = -p_t, Lz = p_phi, and the Carter constant
  //   Q = p_theta^2 + cos^2(theta) [ Lz^2/sin^2(theta) - a^2 E^2 ]
  e = max(-pt, 1e-8);
  float lz = pphi;
  float c0 = cos(th0);
  float q = pth * pth + c0 * c0 * ((lz * lz) / s02 - a2 * e * e);

  float r = r0;
  float th = th0;
  float phi = uCamPos.z;
  float signR = pr >= 0.0 ? 1.0 : -1.0;
  float signT = pth >= 0.0 ? 1.0 : -1.0;

  float rHorizon = 1.0 + sqrt(max(0.0, 1.0 - a2));
  vec3 acc = vec3(0.0);
  float transmittance = 1.0;
  int crossings = 0;
  bool escaped = false;

  for (int i = 0; i < 4096; i++) {
    if (i >= uMaxSteps) break;
    if (r <= rHorizon + 1e-3) break; // swallowed by the horizon
    if (r >= uEscapeRadius) { escaped = true; break; }

    float h = adaptiveStep(r, th, e, lz, q, uStep);

    float vr = signR * sqrt(max(0.0, radialPotential(r, e, lz, q)));
    float vt = signT * sqrt(max(0.0, polarPotential(th, e, lz, q)));
    float ar = radialAccel(r, e, lz, q);
    float at = polarAccel(th, e, lz);

    float rNew = r + vr * h + 0.5 * ar * h * h;
    float thNew = th + vt * h + 0.5 * at * h * h;

    float arNew = radialAccel(rNew, e, lz, q);
    float atNew = polarAccel(thNew, e, lz);
    float vrNew = vr + 0.5 * (ar + arNew) * h;
    float vtNew = vt + 0.5 * (at + atNew) * h;

    // Equatorial crossing: interpolate the event, then accumulate emission.
    if (crossings < 16) {
      float fPrev = th - PI * 0.5;
      float fNew = thNew - PI * 0.5;
      if (fPrev * fNew < 0.0) {
        float w = fPrev / (fPrev - fNew);
        float rc = r + w * (rNew - r);
        float alphaC;
        acc += transmittance * diskEmission(rc, e, lz, alphaC);
        transmittance *= 1.0 - alphaC;
        crossings++;
        if (transmittance < 0.01) break;
      }
    }

    phi += 0.5 * h * (dphiDlambdaM(r, th, e, lz) + dphiDlambdaM(rNew, thNew, e, lz));
    r = rNew;
    th = thNew;
    signR = vrNew >= 0.0 ? 1.0 : -1.0;
    signT = vtNew >= 0.0 ? 1.0 : -1.0;
  }

  // Background. The star field is sampled from the ray's local direction on
  // the escape sphere, so gravitational lensing bends it and produces the
  // shadow and photon ring — it is not a flat skybox.
  vec3 bg = vec3(0.0);
  if (escaped) {
    float s2 = max(sin(th) * sin(th), SIN2_FLOOR);
    vec4 me = metricComponents(r, th);
    float de = deltaOf(r);
    float alphaE = sqrt(max(de * me.w / aOf(r, s2), 1e-12));
    float betaE = me.y / me.z;

    // Tetrad components of the momentum from the covariant ones:
    //   p^(t)     = (E + beta^phi Lz) / alpha
    //   p^(r)     = signR sqrt(R) / sqrt(Delta Sigma)
    //   p^(th)    = signT sqrt(Theta) / sqrt(Sigma)
    //   p^(phi)   = Lz / sqrt(g_phiphi)
    //
    // The sign in p^(t) is easy to get backwards. The ZAMO 4-velocity is
    // n^mu = (1/alpha, 0, 0, -beta^phi/alpha), so the locally measured energy is
    //   E_local = -n^mu p_mu = (E + beta^phi Lz) / alpha,
    // and beta^phi = g_tphi/g_phiphi is *negative* for a > 0, so corotating
    // photons carry less local energy than alpha^-1 E.
    float ptHat = (e + betaE * lz) / alphaE;
    float nr = signR * sqrt(max(0.0, radialPotential(r, e, lz, q))) / sqrt(max(de * me.w, TINY));
    float nt = signT * sqrt(max(0.0, polarPotential(th, e, lz, q))) / sqrt(me.w);
    float np = lz / sqrt(me.z);
    vec3 n = vec3(nr, nt, np) / max(ptHat, TINY);
    bg = vec3(starField(normalize(skyDirection(n, th, phi)))) * uStarBrightness;
  }

  vec3 color = acc + transmittance * bg;

  // Progressive accumulation: blend this frame into the running average so
  // error from a coarse step budget resolves over time. Alpha carries the
  // frame count for the next iteration.
  vec4 prev = texture(uPrevAccum, gl_FragCoord.xy / res);
  float n = uAccumIndex;
  vec3 accum = n <= 0.0 ? color : mix(prev.rgb, color, 1.0 / (n + 1.0));
  outColor = vec4(accum, n + 1.0);
}
`;

/**
 * Post pass: bloom, exposure, ACES tonemap, vignette, dither. Reads the
 * accumulated HDR target.
 */
const POST = /* glsl */ `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uScene;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uBloomThreshold;
uniform float uVignette;

vec3 aces(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
  vec3 color = texture(uScene, vUv).rgb;

  // Bloom: bright-pass then a small rotated kernel. Cheap, and enough for the
  // disk glow to bleed the way a real long exposure does.
  vec2 texel = 1.0 / vec2(textureSize(uScene, 0));
  vec3 bright = max(color - uBloomThreshold, vec3(0.0));
  vec3 bloom = bright * 0.20;
  for (int i = 0; i < 4; i++) {
    float ang = float(i) * 0.7853981634;
    vec2 off = vec2(cos(ang), sin(ang)) * texel * 14.0;
    bloom += max(texture(uScene, vUv + off).rgb - uBloomThreshold, vec3(0.0)) * 0.20;
    bloom += max(texture(uScene, vUv - off).rgb - uBloomThreshold, vec3(0.0)) * 0.20;
  }
  color += bloom * uBloomStrength;

  color = aces(color * uExposure);

  vec2 p = vUv * 2.0 - 1.0;
  color *= clamp(1.0 - uVignette * dot(p, p) * 0.35, 0.0, 1.0);

  // Dither to break up banding in the smooth disk gradients.
  float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  color += (dither - 0.5) / 255.0;

  outColor = vec4(pow(max(color, 0.0), vec3(1.0 / 2.2)), 1.0);
}
`;

export { VERTEX, TRACE, POST };
