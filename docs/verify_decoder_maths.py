"""
Numerical verification of the decoder maths (V-JEPA 2.1 -> diffusion decoder -> Cosmos).

Everything here is checkable by hand or on paper. Each block prints:
  - what is claimed analytically
  - the measured value
  - PASS / FAIL

Run:  python3 verify_decoder_maths.py
"""

import numpy as np

rng = np.random.default_rng(0)
W = 78
def hdr(t): print("\n" + "=" * W + "\n" + t + "\n" + "=" * W)
def chk(name, got, want, tol=1e-9, unit=""):
    ok = abs(got - want) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"          claimed = {want:.6g}{unit}   measured = {got:.6g}{unit}")
    return ok

# ============================================================================
hdr("1. MSE optimum is the conditional mean  (E[f*] = E[x|z])")
# ============================================================================
# Claim: argmin_f E||x - f(z)||^2  is  f*(z) = E[x|z].
# Build a 2-component conditional distribution and fit a constant per z.
n = 200_000
z = rng.integers(0, 2, n)                     # two contexts
mu_ctx = np.array([-1.0, 3.0])                # E[x|z]
mu = mu_ctx[z]
x = mu + rng.normal(0, 1.0, n)                # x|z ~ N(mu_z, 1)

for zz in (0, 1):
    m = z == zz
    chk(f"E[x | z={zz}] (the MSE optimum)", x[m].mean(), mu_ctx[zz], tol=0.02)

# also: the MSE-optimal constant minimizes squared error -> check gradient ~ 0
for zz in (0, 1):
    m = z == zz
    c = x[m].mean()
    g = 2 * (c - x[m]).mean()                 # d/dc E[(x-c)^2]
    chk(f"gradient at optimum, z={zz}", g, 0.0, tol=0.02)

# ============================================================================
hdr("2. The conditional mean is a BLUR: variance shrinks")
# ============================================================================
# Claim: Var(E[x|z]) <= Var(x), with equality iff x is a function of z.
# The "blur" is exactly this lost variance.
for zz in (0, 1):
    m = z == zz
    print(f"  z={zz}: Var(x)={x[m].var():.4f}  Var(E[x|z])={mu[zz]:.4f} "
          f"-> residual (imagined) variance={x[m].var():.4f}")
tot_var = x.var()
between = np.var(mu)                          # Var(E[x|z])
within = np.mean([x[z == zz].var() for zz in (0, 1)])
print(f"  law of total variance: Var(x) = Var(E[x|z]) + E[Var(x|z)]")
chk("  total", tot_var, between + within, tol=0.05)   # MC estimate -> loose tol
print(f"  => fraction of variance that MUST be invented: {within/tot_var:.4f}")

# ============================================================================
hdr("3. Frame skip = 4 destroys 3/4 of the frames (the decimation)")
# ============================================================================
# Claim: a signal sampled at rate R, decimated by s, keeps only R/s of the samples.
# The information in the dropped samples is gone; you can only model p(dropped | kept).
R, s = 20.0, 4.0
chk("kept rate (Hz)", R / s, 5.0, unit=" Hz")
chk("fraction of frames kept", 1 / s, 0.25)
chk("frames to invent per feature", s - 1, 3.0)

# Fourier view: decimation by s folds frequencies above the new Nyquist.
# Build a tone ABOVE the decimated Nyquist and show it aliases.
N = 4096
t = np.arange(N) / R
f_true = 4.0            # Hz, above decimated Nyquist = (R/s)/2 = 2.5 Hz
sig = np.cos(2 * np.pi * f_true * t)
dec = sig[::int(s)]     # decimated signal
td = np.arange(len(dec)) / (R / s)
# least-squares fit a single tone to the decimated data
fr = np.fft.rfftfreq(len(dec), d=1 / (R / s))
sp = np.abs(np.fft.rfft(dec))
f_alias_meas = fr[np.argmax(sp)]
f_alias_pred = abs(f_true - round(f_true / (R / s)) * (R / s))
chk("aliased frequency after decimation", f_alias_meas, f_alias_pred, tol=0.05, unit=" Hz")
print(f"  true tone {f_true} Hz -> appears as {f_alias_meas:.3f} Hz after decimation.")
print("  => the decoder is NOT reconstructing; it is resolving an ambiguity (anti-aliasing).")

# ============================================================================
hdr("4. Reverse SDE / probability-flow ODE (Song et al. 2021)")
# ============================================================================
# Forward:  dx = f(x,t) dt + g(t) dw,  x(0)~data
# Reverse:  dx = [f - g^2 * grad_x log p_t(x)] dt + g dw_bar
# PF-ODE:   dx = [f - 0.5 g^2 * score] dt     (deterministic, same marginals)
#
# Check with the VE (variance-exploding) SDE: f=0, g(t)=sqrt(d sigma^2/dt), sigma(t).
# Then score(x,sigma) = -(x - x0)/sigma^2  for a single data point x0.
# Claim: integrating the PF-ODE from x(t=T)~N(0,sigma_T^2) to t=0 recovers x0.
print("  VE-SDE: f=0, g(t)^2 = d(sigma^2)/dt, PF-ODE dx/dt = -0.5 g^2 score = (x-x0)/... ")
def run_ve_ode(x0, sigma_max=10.0, steps=4000):
    # sigma grows linearly 0 -> sigma_max; use s as the time variable for stability
    s = np.linspace(sigma_max, 1e-3, steps)          # integrate from large noise down
    x = rng.normal(0, sigma_max)
    for i in range(steps - 1):
        ds = s[i + 1] - s[i]
        # score of N(x0, sigma^2) is -(x-x0)/sigma^2 ; PF-ODE dx/dsigma = -sigma*score*? 
        # For VE: dx/dsigma = -sigma * score  (since 0.5 g^2 = sigma dsigma and dt... )
        score = -(x - x0) / s[i] ** 2
        x = x + ds * (-s[i] * score)
    return x
errs = [abs(run_ve_ode(x0) - x0) for x0 in (0.5, -1.3, 2.0)]
print(f"  PF-ODE reconstruction error over 3 data points: {[f'{e:.2e}' for e in errs]}")
chk("PF-ODE recovers x0 (max err)", max(errs), 0.0, tol=1e-2)

# ============================================================================
hdr("5. Tweedie's formula: E[x0 | x_t] = (x_t + sigma^2 score) / alpha")
# ============================================================================
# VE case: alpha=1, so E[x0|x_sigma] = x + sigma^2 * score(x).
# For a Gaussian mixture prior we can compute the score exactly and check.
mus = np.array([-2.0, 0.0, 2.0]); wts = np.array([0.3, 0.4, 0.3]); sd0 = 1.0
def score_gmm(x, sigma):
    s2 = sigma ** 2 + sd0 ** 2
    p = wts * np.exp(-0.5 * (x - mus) ** 2 / s2)
    Z = p.sum()
    return np.sum(p * (-(x - mus) / s2)) / Z
def tweedie(x, sigma):
    return x + sigma ** 2 * score_gmm(x, sigma)
# brute-force E[x0 | x_sigma] by quadrature
def posterior_mean(x, sigma):
    s2 = sigma ** 2 + sd0 ** 2
    p = wts * np.exp(-0.5 * (x - mus) ** 2 / s2)
    return np.sum(p * ((mus / sd0 ** 2 + x / sigma ** 2) / (1 / sd0 ** 2 + 1 / sigma ** 2))) / p.sum()
for x in (-1.5, 0.3, 2.4):
    for sigma in (0.5, 2.0):
        chk(f"Tweedie @ x={x}, sigma={sigma}", tweedie(x, sigma),
            posterior_mean(x, sigma), tol=1e-8)

# ============================================================================
hdr("6. Noise schedules (VP-SDE, cosine) - the actual functions")
# ============================================================================
# VP: beta(t) = b_min + t(b_max-b_min);  alpha(t) = exp(-0.5 int beta)
# DDPM (linear beta): alpha_bar_t = prod (1 - beta_i)
T = 1000
beta = np.linspace(1e-4, 0.02, T)
abar = np.cumprod(1 - beta)
print(f"  linear DDPM:  alpha_bar[0]={abar[0]:.6f}  alpha_bar[-1]={abar[-1]:.3e}")
chk("alpha_bar monotonically decreasing", float(np.all(np.diff(abar) < 0)), 1.0)
# cosine schedule (Nichol & Dhariwal 2021)
def cosine_abar(t, T=1000, s=0.008):
    f = lambda u: np.cos((u / T + s) / (1 + s) * np.pi / 2) ** 2
    return f(t) / f(0)
tt = np.arange(T + 1)
cab = cosine_abar(tt)
print(f"  cosine:      alpha_bar[0]={cab[0]:.6f}  alpha_bar[-1]={cab[-1]:.3e}")
chk("cosine alpha_bar[0] == 1", float(cab[0]), 1.0, tol=1e-9)
print(f"  signal-to-noise at t=500:  linear={abar[500]/(1-abar[500]):.3f}"
      f"   cosine={cab[500]/(1-cab[500]):.3f}")

# ============================================================================
hdr("7. Sampler step counts vs error (why N is the budget line)")
# ============================================================================
# Claim: first-order (Euler) discretisation error ~ O(1/N).
# IMPORTANT: for a SINGLE Gaussian the PF-ODE is exactly integrable, so Euler is
# exact at any N (error = 0). The N-dependence appears only for a NONLINEAR score.
# Use a Gaussian MIXTURE prior -> nonlinear score -> real discretisation error.
mus = np.array([-2.0, 0.0, 2.0]); wts = np.array([0.3, 0.4, 0.3]); sd0 = 1.0
def score_gmm(x, sigma):
    s2 = sigma ** 2 + sd0 ** 2
    p = wts * np.exp(-0.5 * (x - mus) ** 2 / s2)
    return np.sum(p * (-(x - mus) / s2)) / p.sum()

def euler_gmm(N, x_start, sigma_max=10.0, sigma_min=1e-3):
    s = np.linspace(sigma_max, sigma_min, N)
    x = x_start
    for i in range(N - 1):
        ds = s[i + 1] - s[i]
        x = x + ds * (-s[i] * score_gmm(x, s[i]))   # VE PF-ODE: dx/dsigma = -sigma*score
    return x

x_start = rng.normal(0, 10.0)
ref = euler_gmm(40000, x_start)
print("   N      |x_N - x_ref|    ratio  (should ~ 2 when N doubles)")
prev = None
for N in (50, 100, 200, 400):
    e = abs(euler_gmm(N, x_start) - ref)
    r = "" if prev is None else f"   {prev/e:.2f}x"
    print(f"  {N:5d}   {e:.3e}{r}")
    prev = e
print("  => doubling N roughly halves the error: cost is linear in N.")
print("  (for a single Gaussian the ODE is exact, so the error would be 0 at every N)")

# show that the single-Gaussian case is indeed exact at any N
def euler_single(N, x_start, x0=1.7, sigma_max=10.0, sigma_min=1e-3):
    s = np.linspace(sigma_max, sigma_min, N)
    x = x_start
    for i in range(N - 1):
        ds = s[i + 1] - s[i]
        x = x + ds * (-s[i] * (-(x - x0) / s[i] ** 2))
    return x
xs = rng.normal(0, 10.0)
exact = euler_single(2, xs)
chk("single Gaussian: Euler exact at N=2", euler_single(50, xs), exact, tol=1e-9)


# ============================================================================
hdr("8. The compute budget, as functions of the design")
# ============================================================================
P, n_tok, N, d = 3.16e8, 1e3, 100, 1e3
dec = 10 * P * n_tok * N          # FLOPs per second of 20 fps video
lin = 2 * d ** 2                  # FLOPs per readout step
print(f"  decoder : 10*P*n*N = {dec:.3e} FLOPs per second of video")
print(f"  readout : 2*d^2    = {lin:.3e} FLOPs per step")
chk("ratio decoder/readout", dec / lin, 1.5807e8, tol=1e6)
print(f"  at 5e14 FLOP/s dense: {dec/5e14:.3f} GPU-seconds per second of video")
print(f"  weights fp16: decoder {2*P/1e6:.1f} MB vs readout {2*d*d/1e6:.1f} MB")
print("  NOTE: this is '10' because 20 fps / frame-skip-4 = 5 decoder calls per second.")

hdr("done")
