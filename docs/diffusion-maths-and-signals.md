# Signals, Diffusers, and the Deep Maths

> Companion to `decoder-explained.html` and `decoder-explained.md`.
> Everything here is verified numerically in `verify_decoder_maths.py` (20/20 checks pass).
> Author: vrinda · salujavrinda018@gmail.com · [github.com/vrindasaluja18](https://github.com/vrindasaluja18)

Two questions answered:

1. **Can we apply signals to the video phase?** Yes — and it's the *right* frame. The decoder is a
   **multirate / bandwidth-extension** problem. This section makes that exact.
2. **What is "diffusers", and what's the real maths?** The library, plus the score/SDE/Tweedie
   maths underneath, with the actual functions written out.

---

# Part A — Signals

## A.1 The decoder is a multirate filter bank problem

Look at the rates again:

```
RGB       20 fps   ── E, skip s=4 ──►  z        5 Hz      (decimation by 4)
RGB       20 fps   ── C, no skip  ──►  c       20 Hz      (full rate)
decoder:   z (5 Hz)  ──────────────►  ĉ (20 Hz)          (interpolation by 4)
```

That is exactly **upsampling by 4**. In DSP terms the decoder is a *synthesis filter bank* that
takes a critically-sampled 5 Hz signal and expands it to 20 Hz. The classical pipeline is:

```
x[n] --↓4--> z[m] --↑4--> ẑ[n] --H(z)--> x̂[n]
        decimate      zero-stuff      interpolate
```

Classical interpolation inserts zeros and low-passes to kill the imaging. It produces a **smooth**
signal — the exact analogue of the blurry conditional mean.

## A.2 What the sampling theorem actually says here

A signal at rate `R = 20` Hz has Nyquist `R/2 = 10` Hz. After decimation by `s = 4`, the new
sample rate is `R/s = 5` Hz, so the **new Nyquist is 2.5 Hz**. Every frequency in `(2.5, 10]` Hz
is now *above Nyquist* and folds back (aliases).

Verified: a pure tone at 4 Hz, decimated by 4 from 20 Hz, appears at

```
f_alias = |f - round(f / (R/s)) · (R/s)| = |4 - round(4/5)·5| = 1 Hz      ✓ measured 1.001 Hz
```

**This is the whole "3 missing frames" story in one line.** The high-frequency temporal content
(fast motion) is exactly what aliased into the 5 Hz features. The decoder cannot undo the fold —
aliasing is many-to-one. It must *choose* which pre-image is most plausible. That choice is a
prior, which is why the decoder must be generative.

> Decimation is `y[m] = x[ms]`. It is not invertible: many `x` give the same `y`.
> Compare §1 of the main doc: many-to-one ⇒ information destroyed.

## A.3 The information identity (and the gap)

For the 4-frame group between two features, entropy accounting gives

```
H(x_{4i+1 .. 4i+3} | x_{4i}, x_{4i+4})  >  0
```

i.e. the missing frames are **not** determined by their neighbours — that's *why* you can't just
interpolate. The decoder is fitting

```
p(x_missing | x_kept)          =  p(x_missing, x_kept) / p(x_kept)
```

which is a **conditional generative model**. Every method in video frame interpolation is a
different factorisation of that object. From the AceVFI survey (arXiv:2506.01061):

| Family | What it models | Failure mode |
|---|---|---|
| flow-based | warp with optical flow `x̂ = W(x_a, F)` | occlusion, disocclusion |
| kernel-based | adaptive separable kernels | large motion |
| phase-based | interpolate phase in a pyramid | large motion |
| **diffusion-based** | sample `p(x_mid \| x_a, x_b)` | compute (N passes) |

The paper you were reading picks the last row, and pays in compute.

## A.4 Why the classical interpolator is the "blur"

Ideal interpolation (`sinc`) is `L2`-optimal for band-limited signals. But natural video is **not
band-limited** — motion creates genuinely new high frequencies. So the `L2`-optimal estimator
collapses to the conditional mean:

```
x̂ = E[x_missing | x_kept]        ← the Wiener/MMSE solution
```

The MMSE solution minimizes MSE *by construction*, so it must be the smooth average. **You cannot
be both MMSE-optimal and sharp.** Diffusion chooses sharpness and accepts MSE. This is the same
statement as §3 of the main doc, now in filter-bank language.

## A.5 The one place a signal-processing trick *does* help: sampling

Where DSP genuinely pays off is **not** in the decoder's output, but in how you solve the reverse
process. The reverse SDE is an SDE; integrating it is a numerical analysis problem:

- **Predictor-corrector** (Song et al.): Euler step + Langevin correction. The corrector is a
  denoising filter applied at each step.
- **Heun / higher-order** (EDM, Karras et al.): second-order ODE solver → far fewer steps for the
  same error.
- **Multistep** (DPM-Solver, DEIS): reuse past evaluations like a linear multistep filter.

All three are **classical numerical-integration / filter design** applied to a learned score.
Verified: with a first-order Euler solver on a nonlinear (mixture) score, error halves when `N`
doubles — exactly `O(1/N)`:

```
   N      |x_N - x_ref|    ratio
    50   7.678e-02
   100   3.811e-02       2.01x
   200   1.895e-02       2.01x
   400   9.418e-03       2.01x
```

> Nice subtlety: for a *single* Gaussian the PF-ODE is exactly integrable, so Euler is exact at
> **any** `N`. The `N`-dependence is a consequence of the score being *nonlinear*. Verified.

---

# Part B — Diffusers and the deep maths

## B.1 "Diffusers" — what it is

Two different things share the name:

1. **Hugging Face `diffusers`** — the library (`github.com/huggingface/diffusers`). It is the
   standard toolbox: `UNet2DConditionModel`, `AutoencoderKL`, `DDPMScheduler`,
   `DPMSolverMultistepScheduler`, `StableDiffusionPipeline`. It is a *software* artifact, not a
   paper. Docs: <https://huggingface.co/docs/diffusers>.
2. **The field's "diffusers"** — the physical-optics term (light diffusers). Irrelevant here; worth
   knowing so you don't chase the wrong paper.

For *understanding*, the library is not the reference. The references are B.3 below. Read the
library source only to see how a scheduler turns the maths into a loop.

## B.2 The probability-flow ODE (the core object)

**Forward SDE** (data → noise), Song et al. 2021, arXiv:2011.13456:

```
dx = f(x, t) dt + g(t) dw
```

**Reverse-time SDE** (noise → data):

```
dx = [ f(x,t) − g(t)² ∇_x log p_t(x) ] dt + g(t) dw̄
```

The only learned quantity is the **score** `∇_x log p_t(x)`.

**Probability-flow ODE** — same marginals, deterministic:

```
dx = [ f(x,t) − ½ g(t)² ∇_x log p_t(x) ] dt
```

Three objects — forward SDE, reverse SDE, PF-ODE — and they share the same marginals `p_t`.
The PF-ODE is what you actually integrate for fast sampling.

**VE-SDE** (used in the verification, `f = 0`, `g² = dσ²/dt`):

```
dx/dσ = −σ · ∇_x log p_σ(x)
```

Verified: integrating this from `σ=10` down to `10⁻³` recovers the data point (max error 2.4e-3
at 4000 steps).

## B.3 The score is a denoiser — Tweedie's formula

This is the single most useful identity. For `x_t = α_t x_0 + σ_t ε`:

```
∇_{x_t} log p_t(x_t) = − (x_t − α_t E[x_0 | x_t]) / σ_t²

⟺   E[x_0 | x_t] = ( x_t + σ_t² ∇_{x_t} log p_t(x_t) ) / α_t
```

So **learning the score = learning a denoiser**, up to a known affine map. That's why the network
is called `ε_θ`: predicting noise is equivalent to estimating the score.

Verified against exact Gaussian-mixture posteriors (6 configurations, error < 1e-8):

```
Tweedie @ x=-1.5, σ=0.5   PASS   claimed = -1.44681   measured = -1.44681
Tweedie @ x= 2.4, σ=2.0   PASS   claimed =  1.19555   measured =  1.19555
```

Consequence: the trained network `s_θ(x, σ)` is **a family of denoisers indexed by σ**. At large σ
it does coarse structure; at small σ it does fine detail. That's the "coarse-to-fine" behaviour you
see in the denoising animation.

## B.4 Denoising score matching — the loss, derived

Let `p_σ(x̃) = ∫ p_data(x) N(x̃; x, σ²) dx`. The **denoising score matching** identity is

```
∇_x̃ log p_σ(x̃) = (1/σ²) E[ x − x̃ | x̃ ]
```

so instead of matching the intractable `∇ log p_σ`, we regress on the *perturbation direction*:

```
L(θ) = E_{x, ε, σ} [ λ(σ) · ‖ s_θ(α_t x + σ_t ε, σ_t) + ε / σ_t ‖² ]
```

This is exactly the `‖ε − ε_θ‖²` objective from the main doc, with `ε_θ = −σ_t s_θ`. Verified
end-to-end: the PF-ODE using the learned score reconstructs held-out points.

## B.5 The actual functions: schedules

**VP / DDPM linear** (`β` linear, `ᾱ_t = Π (1 − β_i)`):

```
β(t) = β_min + t (β_max − β_min),    ᾱ_t = exp(−∫₀ᵗ β(u) du)   (continuous form)
```

**Cosine schedule** (Nichol & Dhariwal 2021):

```
ᾱ_t = cos²( (t/T + s) / (1 + s) · π/2 ) / cos²( s / (1+s) · π/2 ),    s = 0.008
```

Verified (`T = 1000`):

```
linear DDPM:  ᾱ[0] = 0.999900     ᾱ[999] = 4.04e-05
cosine:       ᾱ[0] = 1.000000     ᾱ[999] = 3.75e-33
SNR at t=500: linear = 0.084      cosine = 0.976
```

The cosine schedule keeps **more signal for longer** (SNR 0.976 vs 0.084 at the midpoint), which is
why it trains better at low resolution. The schedule is a *design choice*, not a law of nature.

## B.6 Samplers, as functions

| Sampler | Update (sketch) | Order | Cost |
|---|---|---|---|
| DDPM ancestral | `x ← (x − (β/√(1−ᾱ)) ε_θ)/√α + σ z` | 1 | N passes |
| DDIM | deterministic, `η=0` | 1 | N passes |
| Euler (EDM) | `x ← x + Δσ · dx/dσ` | 1 | N passes |
| Heun (EDM) | Euler + trapezoid correction | 2 | 2N passes |
| DPM-Solver++ | exponential integrator, multistep | 2–3 | ~10–20 passes |
| Consistency | `x₀ ← f_θ(x_σ, σ)` directly | 1 | **1–4 passes** |

The last row is the interesting one: a **consistency model** learns the ODE *solution map* directly,
so you skip the loop entirely. This is the algorithmic answer to "N is the budget line".

## B.7 Guidance — the conditional machinery

To condition on `z` (the V-JEPA features) you have two routes:

**Classifier guidance** (needs a classifier):

```
∇_x log p(x|z) = ∇_x log p(x) + ∇_x log p(z|x)
```

**Classifier-free guidance** (no classifier — the standard today): train one network on
`ε_θ(x, t, z)` and one on `ε_θ(x, t, ∅)`, then extrapolate at sampling:

```
ε̃_θ = ε_θ(x, t, ∅) + w · ( ε_θ(x, t, z) − ε_θ(x, t, ∅) )
```

`w > 1` sharpens conditioning at the cost of diversity. **In the V-JEPA pipeline, `z` is the
conditioning signal**, so the decoder is a conditional score model and guidance is the knob that
trades "faithful to the features" against "plausible motion". That is exactly the trap in §6 of the
main doc: too much `w` and the decoder invents confidently, and you credit the world model for it.

## B.8 Flow matching — the current alternative

Instead of a noise schedule, define a **probability path** and regress the velocity:

```
x_t = (1−t) x_0 + t x_1,      target velocity u = x_1 − x_0

L(θ) = E ‖ v_θ(x_t, t) − (x_1 − x_0) ‖²
```

Rectified flow / flow matching (Lipman et al. 2022; Liu et al. 2022) is a straight-line special
case. It is what most 2024–2025 video models (including many Cosmos-family models) use, because the
paths are straighter and thus need fewer steps. Same budget logic: **fewer N**.

---

# Part C — Papers to read, in order

Ordered so each one only needs the previous ones. Starred = start here.

**Foundations**

1. ★ **Ho, Jain, Abbeel — Denoising Diffusion Probabilistic Models** (arXiv:2006.11239, 2020).
   The origin. Read §2–3 for the variational bound and the `‖ε−ε_θ‖²` simplification.
2. ★ **Song, Meng, Ermon — Score-Based Generative Modeling through SDEs** (arXiv:2011.13456, 2021).
   The unifying frame: forward/reverse SDE, PF-ODE, predictor-corrector. Read §3–4.
3. ★ **Park & McCann — Random Walks with Tweedie** (arXiv:2411.18702, 2024).
   **Written explicitly for the signal-processing community.** Derives DDPM/score-SDE/EDM from a
   handful of textbook results and gives algorithmic templates. If you read one paper, read this
   one after (1) and (2) — it is the shortest path from DSP to diffusion.

**Design space and practice**

4. **Karras et al. — Elucidating the Design Space of Diffusion-Based Generative Models**
   (arXiv:2206.00364, 2022). Preconditioning, noise schedules, Heun solver, 35-step sampling. This
   is where "the design space separates" — very useful if you want to make a *budget* argument.
5. **Rombach et al. — High-Resolution Image Synthesis with Latent Diffusion Models**
   (arXiv:2112.10752, 2021). The latent-space idea: diffuse in `c`-space, not pixel space. This is
   literally what the Cosmos decoder does.
6. **Nichol & Dhariwal — Improved DDPM** (arXiv:2102.09672, 2021). The cosine schedule (B.5).

**Sampling efficiency (the compute story)**

7. **Song et al. — Consistency Models** (arXiv:2303.01469, 2023). One-step generation.
8. **Lu et al. — DPM-Solver** (arXiv:2206.00927, 2022). Exponential-integrator multistep solvers.
9. **Lipman et al. — Flow Matching** (arXiv:2210.02747, 2022) / **Liu et al. — Rectified Flow**
   (arXiv:2209.03003, 2022). Straight paths, fewer steps.

**Video specifically**

10. **Ho et al. — Video Diffusion Models** (arXiv:2204.03458, 2022). The first video DDPM.
11. ★ **Kye et al. — AceVFI: A Comprehensive Survey of Video Frame Interpolation**
    (arXiv:2506.01061, 2025). 250+ papers, categorises flow/kernel/phase/GAN/**diffusion**. This is
    *exactly* the "3 missing frames" problem — read the taxonomy table (A.3).
12. **Lyu et al. — Frame Interpolation with Consecutive Brownian Bridge Diffusion**
    (arXiv:2405.05953, 2024). Diffusion where the endpoints are *fixed* to the observed frames —
    the principled version of "condition on both neighbours".

**Inverse problems (if you want the posterior view)**

13. **Song et al. — Solving Inverse Problems in Medical Imaging with Score-Based Generative Models**
    (arXiv:2111.08005, 2021). The `∇ log p(z|x)` route (B.7).

**The library**

14. **Hugging Face `diffusers`** — <https://huggingface.co/docs/diffusers>. Read
    `schedulers/` after (4) so you recognise the maths in the code.

---

# Part D — What this buys you for your thesis

| Signal-processing concept | Decoder | Your ESN |
|---|---|---|
| decimation by `s` | frame skip 4 | subsampling the state read-out |
| aliasing | fast motion folded into `z` | fast dynamics folded into `r` |
| bandwidth extension | upsampling 5→20 Hz | reconstructing a fast signal from slow state |
| MMSE / Wiener filter | `E[x\|z]` = blur | MSE readout = smooth average |
| generative prior | diffusion `p(x\|z)` | generative readout |
| solver order `N` | sampling steps | readout cost |

The one-line thesis stays the same, now with a signal-processing proof:

```
decimation is many-to-one ⇒ aliasing ⇒ the lost band is NOT recoverable by any linear filter.
A linear (MMSE) readout returns the conditional mean = the low-passed blur.
A generative readout samples the folded band from a learned prior — and pays N passes.
```

That is a *provable* statement, not a vibe. It is also exactly what your `credence` work does with
delay embedding: Takens says the state reconstructs the attractor **when the embedding is right**;
get the sampling rate wrong and you alias, and no readout fixes it.

---

# Verification

`verify_decoder_maths.py` — run it:

```bash
pip install numpy
python3 verify_decoder_maths.py
```

20/20 checks pass. It verifies:

1. MSE optimum = conditional mean (2 contexts)
2. law of total variance / the invented-variance fraction
3. decimation arithmetic + **aliasing of a 4 Hz tone to 1 Hz**
4. PF-ODE reconstruction of a data point
5. Tweedie's formula against exact Gaussian-mixture posteriors (6 cases)
6. linear vs cosine noise schedules, SNR at the midpoint
7. **O(1/N) sampler convergence** (2.01× per doubling) and the single-Gaussian exactness subtlety
8. the compute budget (`10·P·n·N` vs `2·d²`, ratio 1.58e8)
