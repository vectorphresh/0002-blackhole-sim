# Physics

Conventions, derivations, and the checks that caught the bugs. Everything here
is implemented in `src/core/` and tested; the shader in `src/gl/shaders.js` is a
line-by-line port, enforced by `tests/shaderParity.test.js`.

## Conventions

Boyer–Lindquist coordinates $x^\mu = (t, r, \theta, \phi)$, metric signature
$(-,+,+,+)$, geometrised units $G = c = M = 1$. The mass $M$ is the only scale,
so every length is in units of $M$ and every time in units of $GM/c^3$; the HUD
converts to metres and seconds using the solar mass the user picks.

The dimensionless spin is $a = J/M^2$ with $0 \le a < 1$. The horizon is at

$$r_+ = 1 + \sqrt{1 - a^2}, \qquad r_- = 1 - \sqrt{1 - a^2}$$

and the ergosphere meets the equator at $r = 2$ regardless of spin.

## The metric

With $\Sigma = r^2 + a^2\cos^2\theta$, $\Delta = r^2 - 2r + a^2$, and
$A = (r^2 + a^2)^2 - a^2 \Delta \sin^2\theta$:

$$
\begin{aligned}
g_{tt} &= -1 + \frac{2r}{\Sigma} \\
g_{t\phi} &= -\frac{2 a r \sin^2\theta}{\Sigma} \\
g_{\phi\phi} &= \frac{A \sin^2\theta}{\Sigma} \\
g_{rr} &= \frac{\Sigma}{\Delta}, \qquad g_{\theta\theta} = \Sigma
\end{aligned}
$$

The off-diagonal $g_{t\phi}$ is what makes Kerr non-static: it is the frame
dragging, and it is the reason a disk around a spinning hole looks different on
its two limbs even before you account for orbital velocity.

## Null geodesics: two ways, deliberately

A photon follows a null geodesic. There are two ways to compute it here, chosen
so they share no algebra and can therefore check each other.

### 1. The Hamiltonian (reference)

$$\mathcal{H} = \tfrac12 g^{\mu\nu} p_\mu p_\nu = 0 \quad\text{(null constraint)}$$

$$\frac{dx^\mu}{d\lambda} = \frac{\partial \mathcal{H}}{\partial p_\mu} = g^{\mu\nu} p_\nu,
\qquad
\frac{dp_\mu}{d\lambda} = -\frac{\partial \mathcal{H}}{\partial x^\mu} = -\tfrac12 \partial_\mu g^{\alpha\beta} p_\alpha p_\beta$$

integrated with RK4. This needs only the inverse metric and finite-differenced
derivatives of it — no separation, no constants of motion, nothing I derived by
hand. It is slow and it is the arbiter.

### 2. Carter separation (fast, used by the shader)

Kerr is special: the geodesic equations separate, giving three constants of
motion along with the mass-shell condition. For a photon,

$$E = -p_t, \qquad L_z = p_\phi, \qquad
Q = p_\theta^2 + \cos^2\theta\left(\frac{L_z^2}{\sin^2\theta} - a^2 E^2\right)$$

and the motion factorises into two one-dimensional problems,

$$\Sigma^2 \frac{dr}{d\lambda}{}^2 = R(r) = P^2 - \Delta\left[(L_z - aE)^2 + Q\right], \quad P = E(r^2 + a^2) - a L_z$$

$$\Sigma^2 \frac{d\theta}{d\lambda}{}^2 = \Theta(\theta) = Q + a^2 E^2 \cos^2\theta - \frac{L_z^2 \cos^2\theta}{\sin^2\theta}$$

$Q$ is the Carter constant; at $a = 0$ it is the total angular momentum squared
minus $L_z^2$, i.e. the separation of spherical harmonics.

## Mino time: why the shader can use Verlet

Written in the affine parameter $\lambda$, $r$ and $\theta$ are coupled through
the common $\Sigma$. Mino's parameter removes it:

$$d\lambda = \Sigma \, d\lambda_M$$

giving equations that decouple **completely**:

$$\frac{dr}{d\lambda_M} = s_r \sqrt{R(r)}, \qquad
\frac{d\theta}{d\lambda_M} = s_t \sqrt{\Theta(\theta)}$$

$$\frac{dt}{d\lambda_M} = \frac{A E - 2 a r L_z}{\Delta}, \qquad
\frac{d\phi}{d\lambda_M} = \frac{2 a r E}{\Delta} + L_z\left(\frac{1}{\sin^2\theta} - \frac{a^2}{\Delta}\right)$$

Because $r$ and $\theta$ are now independent one-dimensional motions, each obeys

$$\frac{d^2 r}{d\lambda_M^2} = \tfrac12 \frac{dR}{dr}$$

(differentiate $v = s_r\sqrt{R}$: $dv/d\lambda_M = s_r R' v / (2 s_r \sqrt{R}) = R'/2$,
independent of the sign $s_r$). So velocity-Verlet applies directly, and it
handles turning points **for free**: at a turning point $v = 0$ but the
acceleration $\tfrac12 R'$ is not, so the integrator accelerates the ray back
into the allowed region with the correct sign. No sign bookkeeping, no
bisection, no stalling.

Mino time is a pure reparametrisation of affine parameter, so the *path* through
spacetime is identical to the Hamiltonian integrator's — only the parameter
differs. That is what makes the two cross-checkable.

### Closed-form accelerations

Finite-differencing $R$ means cancelling two quantities of order $E^2 r^4$. The
closed forms are cheaper in GLSL and exact:

$$\tfrac12 \frac{dR}{dr} = 2 E r P - (r - 1) K, \qquad K = (L_z - aE)^2 + Q$$

$$\tfrac12 \frac{d\Theta}{d\theta} = -a^2 E^2 \cos\theta \sin\theta + \frac{L_z^2 \cos\theta}{\sin^3\theta}$$

### Adaptive stepping is mandatory, not an optimisation

In Mino time the radial speed is $dr/d\lambda_M = \sqrt{R} \sim E r^2$, which
grows without bound. A step that resolves the near-horizon region leaps
thousands of units per iteration out at the launch radius, and the ray sails
straight past the hole. Worse, near a turning point $R'$ is large, a fixed step
overshoots, and because $R \sim r^4$ the overshoot compounds ($r$: 12 → 45 → 365
→ $10^{12}$ → NaN).

The step is the minimum of four limits, each scaled so that halving the
tolerance halves the step everywhere — which is what preserves Verlet's
second-order convergence:

$$h = \text{clamp}\left(\text{step} \cdot \min\left(\frac{r}{|v_r|}, \frac{1}{|v_\theta|}, \sqrt{\frac{r}{|a_r|}}, \sqrt{\frac{1}{|a_\theta|}}\right),\; 10^{-9},\; 0.25\right)$$

At a turning point $v \to 0$, so the velocity limits relax and the acceleration
limits take over — exactly the regime where the integrator must be careful.

## The observer: ZAMO tetrad

The camera is a hovering observer, and the natural orthonormal frame there is
the ZAMO (zero angular momentum observer) — locally non-rotating, i.e. corotating
with the local inertial frame. With lapse $\alpha = \sqrt{\Delta\Sigma/A}$ and
shift $\beta^\phi = g_{t\phi}/g_{\phi\phi}$ (negative for $a > 0$), a photon of
energy $E$ emitted in local direction $\hat{n}$ (a unit vector) has

$$p^{\hat{t}} = \frac{E}{\alpha}, \qquad
p^{\hat{\phi}} = -\beta^\phi p^{\hat{t}} + \frac{n^{\hat{\phi}}}{\sqrt{g_{\phi\phi}}}, \qquad
p^{\hat{r}} = \frac{n^{\hat{r}}}{\sqrt{g_{rr}}}, \qquad
p^{\hat{\theta}} = \frac{n^{\hat{\theta}}}{\sqrt{g_{\theta\theta}}}$$

Lowering the index then gives $E = -p_t$ and $L_z = p_\phi$ with the signs
settled. Because $p = E(e_{\hat{t}} + n^{\hat{i}} e_{\hat{i}})$ is null by
construction **iff** $|\hat{n}| = 1$, the null constraint is a direct test of the
camera basis — which is what `tests/camera.test.js` asserts.

## Redshift factor

The disk's brightness asymmetry is the whole point of the image, and it comes
from one factor. A photon crossing the disk at radius $r$ was emitted by
material on a circular orbit with angular velocity

$$\Omega = \frac{\pm 1}{r^{3/2} \pm a}$$

(the upper sign prograde; note the $\pm$ on $a$ too — getting that sign wrong is
a real bug, see below). The emitted energy measured in the comoving frame is
$E_{\text{em}} = -u^\mu p_\mu = u^t (E - \Omega L_z)$, and the observer at
infinity measures $E_{\text{obs}} = E$, so

$$\boxed{\;g = \frac{E_{\text{obs}}}{E_{\text{em}}} = \frac{1}{u^t (1 - \Omega \xi)}, \qquad \xi = \frac{L_z}{E}\;}$$

$$u^t = \left(-g_{tt} - 2 g_{t\phi}\Omega - g_{\phi\phi}\Omega^2\right)^{-1/2}$$

This single factor folds together gravitational redshift, transverse Doppler,
and line-of-sight Doppler. A blackbody observed through $g$ scales as $g^4$ —
$g^3$ from specific intensity and $g$ from bandwidth — so beaming appears as a
strong one-sided brightening with no extra tuning.

### The reciprocal is the whole point

Writing $g = u^t(1 - \Omega\xi)$ instead reports light climbing out of the
potential well as *blueshifted*. The arbiter is the Schwarzschild static limit:
at $\Omega = 0$ the correct formula gives $g = 1/u^t = \sqrt{1 - 2/r} < 1$, the
expected redshift; the inverted one gives $u^t = 1/\sqrt{1-2/r} > 1$.

## Disk model

A thin equatorial sheet on circular geodesics with the Shakura–Sunyaev effective
temperature:

$$T_{\text{eff}}(r) \propto \left(\frac{r}{r_{\text{in}}}\right)^{-3/4} \left[1 - \left(\frac{r_{\text{in}}}{r}\right)^{3/4}\right]^{1/4}$$

The flux factor vanishes at the inner edge, which is where the disk stops being
thin and the geodesic assumption breaks down. Writing $x = r/r_{\text{in}}$, the
profile is $T^4 = x^{-3} - x^{-15/4}$, so $d(T^4)/dx = 0$ gives $x^{3/4} = 5/4$,
hence the peak sits at

$$x_{\text{peak}} = (5/4)^{4/3} = 1.3465216\ldots, \qquad T_{\text{peak}} = \tfrac{4}{5}\,5^{-1/4} = 0.5349922\ldots$$

The profile is **not** normalised to a peak of 1 — the maximum is the value
above. The observed colour is $g \cdot T$ and the intensity is $g^4 T^4$ times a
display gain. The gain is **not** physics: $g$ diverges at the *light surface*
(where circular orbits become null, equatorially at $3M$ for Schwarzschild), so
the raw dynamic range is unbounded and meaningless. What matters for display is
the percentile span the renderer actually maps into a colour ramp: measured over
the disk, $g^4$ spans $p99/p01 \approx 15{:}1$ at $a = 0$, $62{:}1$ at $a = 0.5$
and $2400{:}1$ at $a = 0.9$. Without a gain the receding half disappears into
black and the approaching half saturates.

## Validation

The recurring arbiter is **substitution, not memorisation**: take a closed-form
$(E, L_z)$, plug it into the radial potential, and assert $R(r) = 0$ **and**
$R'(r) = 0$. That validates a formula against the geodesic equation itself, with
no solver that can misbehave and no number to misremember.

### Checks that caught real bugs

- **`xiAtRadius` boolean arithmetic.** `prograde * sq` coerces the boolean to 0,
  silently selecting the wrong branch. Both this and a sign error in the same
  expression were invisible at $a = 0$.
- **Inverted redshift.** Caught by the Schwarzschild static limit above.
- **Retrograde orbital velocity in the shader.** The shader had
  $\Omega = \text{dir}/(r^{3/2} + a)$ instead of $\text{dir}/(r^{3/2} + \text{dir}\cdot a)$,
  giving a retrograde disk the *prograde* denominator — understating retrograde
  orbital speed and beaming. Invisible at $a = 0$, so no visual check caught it;
  the JS module was right and the shader had drifted from it. Now pinned by
  `tests/shaderParity.test.js`, which was verified to fail on the buggy source.
- **`iscoRadius` spurious term**, **`circularOmega` denominator**
  ($r^{3/2} \pm a$, not $\pm\sqrt{a}$ — the $a=0$ Kepler test cannot catch it),
  and a wrong closed form for $b_{\text{crit}}$ (gave $\sqrt{63}$ instead of
  $3\sqrt{3}$ at $a=0$).

### Limits that are exact, not asymptotic

- Static Schwarzschild emitter: $g = \sqrt{1 - 2/r}$.
- Zero-angular-momentum photon from a circular orbit in Schwarzschild:
  $g = \sqrt{1 - 3/r}$ exactly.
- The light surface — where a circular orbit becomes null — is at $r = 3M$
  equatorially for $a = 0$ (not $2M$); inside it there is no timelike circular
  emitter and $g$ is undefined.
- The genuine mirror symmetry is $g(r, a, -\xi, -1) = g(r, -a, \xi, +1)$:
  reversing the orbital direction *and* flipping the spin. Schwarzschild is
  **not** left-right symmetric at fixed $\xi$ — beaming comes from orbital
  motion, which exists at $a = 0$.

### Finite launch radius

Rays are launched from a finite radius $r_0$, not infinity, and two HUD
quantities carry that:

- The shadow radius measured from $r_0$ is $3\sqrt{3}\,\sqrt{1 - 2/r_0}$, not
  $3\sqrt{3}$. The $\sqrt{1-2/r_0}$ is the gravitational redshift of the
  shadow's angular size. Asserting this relation validates the tetrad too.
- The flat-space reference for deflection is $2\arccos(b/r_0)$, not $\pi$.
