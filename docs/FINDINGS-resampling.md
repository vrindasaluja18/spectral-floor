# Findings: the one-sided spectrum was clamped, not mirrored

> vrinda · salujavrinda018@gmail.com · 2026-11-13
> Status: fixed, verified, and pinned by regression tests.

---

## What was wrong

The experiment runner built the full-band spectrum from a one-sided Welch estimate like this:

```python
_, S_hat = welch_psd(x, nperseg=nperseg)      # one-sided, freqs in [0, 0.5]
grid = np.linspace(0, 0.5, len(S_hat))
u    = np.linspace(0, 1, 20001)[:-1]
S    = np.interp(u, grid, S_hat)              # <-- BUG
```

`np.interp` **clamps** out-of-range query points. Every frequency above 0.5 was therefore
returned as `S(0.5)`, so the upper half of the band was a constant instead of the mirror image
of the lower half.

This matters because the aliasing identity sums *shifted copies* of `S` across the whole band:

```
S_dec(v) = (1/s) Σ_{k=0}^{s-1} S( (v + k) / s )
```

A wrong upper band is folded back onto the lower band, so the error compounds with stride.

## How big the error was

Measured against the analytic AR(1) spectrum at `a = 0.99`, where the true floor is the
closed form `1 − a^(2s)`:

| stride | true floor | clamped (bug) | mirrored (fix) | clamped error |
|---|---|---|---|---|
| 1 | 0.01990 | 0.01971 | 0.02009 | −1% |
| 4 | 0.07726 | 0.04689 | 0.07791 | **−39%** |
| 16 | 0.27502 | 0.14772 | 0.27658 | **−46%** |
| 32 | 0.47441 | 0.26450 | 0.47604 | **−44%** |

The bug made the floor look **46% too low at stride 16**. Since the floor is the *claim* — the
lower bound a model is compared against — an artificially low floor would have made a perfectly
good predictor look bad, and would have distorted the accuracy-per-byte frontier at exactly the
strides the project cares most about.

## The fix

A real-valued stationary process has an **even** spectral density, `S(f) = S(1 − f)`. The
correct expansion is a mirror, not an interpolation:

```python
# reflect the interior bins; keep DC and Nyquist each exactly once, then
# append the endpoint at 1.0 so the grid spans all of [0, 1) and interp never clamps
f_full = np.concatenate([freqs, 1.0 - freqs[-2:0:-1], [1.0]])
S_full = np.concatenate([psd,   psd[-2:0:-1],       [psd[0]]])
u = np.linspace(0.0, 1.0, n_grid)[:-1]
return u, normalise_psd(np.interp(u, f_full, S_full))
```

Exposed as `spectral_floor.two_sided_psd(freqs, psd)`. The experiment runner now calls it, and
the estimated floor tracks the analytic floor to within ~5% at every stride (it was 46% off at
`s = 16`).

## The part I got wrong first (kept for honesty)

My initial diagnosis was **"Welch estimator bias"**, and I wrote it into the README and into a
regression test. It was wrong. The estimator was fine; the *resampling* was wrong. The
distinguishing experiment, in `scripts/diagnose_resampling.py`, applies both constructions to
the **same, exactly-known analytic spectrum** — so there is no estimator involved at all, and
the discrepancy persists. That isolates the bug to the resampling.

The stale test (`test_welch_floor_is_biased_at_large_stride`) has been replaced by
`test_clamping_the_one_sided_psd_biases_the_floor_low`, which pins the real mechanism.

## How it is locked in

| Test | What it asserts |
|---|---|
| `test_two_sided_psd_is_even_and_normalised` | `S(u) == S(1−u)` exactly, and unit mean |
| `test_two_sided_psd_rejects_mismatched_shapes` | input validation |
| `test_clamping_the_one_sided_psd_biases_the_floor_low` | clamping biases low, the bias grows with stride (0.0002 → 0.13), and mirroring removes it |

73 tests pass (rewritten to check oracles, not shapes). `scripts/diagnose_resampling.py` reproduces both constructions side by side for
anyone who wants to see the bug and the fix in one output.

## The lesson, stated generally

**Interpolating a one-sided spectral estimate onto the full band is a clamping bug, not an
approximation.** Any code that folds, decimates, or aliases a spectrum must first mirror it.
The failure is silent: it produces plausible numbers, and it produces them worst exactly where
the effect is most interesting (large stride).

---

## Follow-up (2026-11-14): make the class of bug unrepresentable

The clamping bug and its siblings all share one cause: a bare `ndarray` carries no
bookkeeping, so a one-sided estimate and a two-sided density are the same object to the type
system. The library now passes a **`Spectrum`** instead — a frozen dataclass holding
`(values, grid, sidedness, normalisation, n_segments, ...)` with its invariants checked on
construction ("parse, don't validate", Alexis King). `Spectrum.from_welch` always returns the
two-sided form; `alias_spectrum` calls `.two_sided()` itself; and `szego`, `floor`, `floor_curve`
accept nothing but a `Spectrum`. The clamp cannot be written any more.

Two more silent-numerics fixes landed with it:

1. **`debias_welch` deleted.** It divided the periodogram pointwise by the window's leakage
   kernel, but leakage is a convolution, not a multiplier: the Hann kernel is 0.25 at bin 1 and
   below 2e-6 from bin 2 on, so floored at 1e-6 those bins were multiplied by 5e5–1e6, turning a
   flat white-noise PSD into a rising ramp. Its only test ("runs and stays positive") passed on
   that garbage. An exact deconvolution is ill-posed (arXiv:2312.13643); the citation stays, the
   function does not.
2. **The confidence interval was ~165× too narrow.** The first `floor` used
   `Var(⟨log Ŝ⟩) ≈ ψ₁(K)/M`, which assumes independent frequency bins. Welch bins are correlated
   through the window, so the interval is now a percentile bootstrap over the segments, which
   carries the correlation through the fold.

Also fixed: the fold now interpolates **periodically** (the old `np.interp` clamped the last
sliver of the band, the same failure mode at the fold stage); `optimal_stride` no longer defaults
to `max_inflation=1.0`, which always returned 1; and `floor` returns a `FloorResult` with the
estimator, effective sample size, interval and warnings, not a bare float.

The residual bias at large stride (Welch smoothing) is a resolution effect, not a resampling one,
and is documented in `docs/SYSTEM-DESIGN.md` §5 and the README's limitations.

