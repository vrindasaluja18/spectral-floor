# System design

> What the pieces are, how data flows, and which invariants hold at each boundary.
> Companion to `docs/api.md`.

---

## 1. The shape of the problem

A world model that observes at rate `R/s` sees a *decimated* process. Decimation
is the covering map on the unit circle

```
z  ↦  z^s          (the "big circle" wraps s times onto the "small circle")
```

and the spectrum folds with it:

```
S_dec(v) = (1/s) Σ_{k=0}^{s-1} S( (v + k)/s )
```

This is the same *shape* at every scale: a circle maps onto a smaller circle, and
the object we reason about (a density on a circle) recurs at each stride. That
self-similarity is the whole architecture. It is also why the bug class is so
persistent: at every scale you hold "a spectrum", and it is only the *bookkeeping*
(grid, sidedness, normalisation, resolution) that changes. If that bookkeeping
lives in a docstring, one wrong call silently produces a plausible wrong number.

**Design decision:** the bookkeeping becomes the type. `Spectrum` is the single
currency. The floor routines accept nothing else, so the two failure modes that
actually bit this project — folding a one-sided estimate as if it were two-sided,
and clamping the last sliver of the band — cannot be expressed.

---

## 2. Layers

```
  raw signal x[n]
        │
        │  welch_psd()               segmented periodogram + averaging
        ▼
  (freqs, psd, n_segments, segs)     one-sided, [0, 0.5]
        │
        │  two_sided_psd()           MIRROR (S(f)=S(1-f)), never interpolate
        ▼
  Spectrum(values, grid, sidedness="two-sided", normalisation, n_segments, ...)
        │
        │  alias_spectrum(Spectrum, s)   periodic fold, z ↦ z^s
        ▼
  Spectrum (the decimated density, still even)
        │
        │  floor()                   Szegő + bias correction + bootstrap CI
        ▼
  FloorResult(eps2, stride, estimator, n_segments, ess, ci_low, ci_high, warnings)
        │
        │  design_law() / optimal_stride()      the decision layer
        ▼
  a stride, with a stated error budget
```

| Module | Responsibility | Must not |
|---|---|---|
| `psd.py` | estimate a density from data | decide sidedness or fold |
| `spectrum.py` | **parse** an estimate into a typed, self-describing object | hold hidden state |
| `floor.py` | fold, evaluate Szegő, quantify uncertainty, decide | accept a bare array |

The dependency arrow only points one way: `floor → spectrum → psd`. `spectrum`
imports `psd` lazily inside its constructors so the types stay importable without
pulling in the estimators.

---

## 3. Data flow, concretely

```python
from spectral_floor import Spectrum, floor, design_law, optimal_stride

S = Spectrum.from_welch(x, nperseg=512)   # always two-sided, unit-mean
r = floor(S, stride=4)                     # FloorResult
print(r.eps2, r.ci_low, r.ci_high, r.warnings)

law = design_law(S, [1, 2, 4, 8, 16])      # inflation vs stride 1, no AR(1) needed
s   = optimal_stride(S, max_inflation=2.0) # the largest affordable stride
```

`Spectrum.from_welch` is the only blessed entry from data. Everything else
(`from_psd`, `from_ar1`) builds a spectrum from values you already trust — fixtures
and analytic ground truth.

---

## 4. Invariants, and where they are enforced

| Invariant | Enforced in | Why it matters |
|---|---|---|
| `values` and `grid` same shape, finite, non-negative, positive mean | `Spectrum.__post_init__` | a density must be a density |
| grid uniform, strictly increasing, in `[0, 0.5]` (one-sided) or `[0, 1)` (two-sided) | `Spectrum.__post_init__` | the fold assumes a uniform grid on the circle |
| `normalisation="unit-mean"` ⇒ mean is 1 | `Spectrum.__post_init__` | keeps the label honest |
| frozen, arrays read-only | `Spectrum.__post_init__` | a parsed value must not drift |
| fold input is two-sided | `alias_spectrum` calls `.two_sided()` itself | removes the clamp-class bug |
| fold wraps periodically | `_periodic_interp` | no flat tail; the fold is exact |
| fold preserves mean power | the `1/s` factor | the result stands alone |
| `log S` finite | `_szego_core` warns on zero bins | the discrete floor is a grid artefact there |
| point estimate + interval + provenance | `FloorResult` | no bare floats leave the library |

---

## 5. Uncertainty: why a bootstrap, not a formula

The first version of `floor` put a formula interval on the result:

```
Var(<log S_hat>)  ≈  ψ₁(K) / M        K segments, M "independent" cells
```

Measured against the AR(1) oracle, that interval was **~165× too narrow**. The
reason is structural, not arithmetic: Welch frequency bins are *correlated* — the
window's leakage kernel couples neighbours — so `M = nperseg/2` independent cells
is simply wrong.

The honest fix is a percentile bootstrap over the segments: resample the `K`
periodograms, recompute the whole mirror-and-fold pipeline on each replicate, and
take the percentiles. Because every replicate goes through the same fold, the
correlation is carried through automatically. Coverage against the oracle is now
~10/10 at small stride.

**The residual bias is real and documented.** At large stride the Welch window
smoothes the spectrum and the *point estimate* drifts high (e.g. +15% at `s=16`
with `nperseg=256`, falling to +0.04% at `nperseg=4096`). A bootstrap widens the
interval but cannot move a biased centre, so at large stride the interval does not
cover the truth. The cure is resolution (`nperseg`), and the README says so.

---

## 6. What was deleted, and why

`debias_welch` divided the averaged periodogram pointwise by the window's leakage
kernel. Leakage is a **convolution**, not a multiplier, so dividing by the kernel
*multiplies* instead of undoing: for a Hann window the normalised kernel is 0.25
at bin 1 and below 2e-6 from bin 2 on, and floored at 1e-6 those bins were scaled
by 5e5–1e6, turning a flat white-noise PSD into a rising ramp.

It was exported, its only test asserted "runs and stays positive" (which passes on
garbage), and the experiment imported it but never called it. It is deleted rather
than repaired: an exact deconvolution is ill-posed and is an open problem
(arXiv:2312.13643, "Debiasing Welch's Method", Biometrika). The citation stays in
the module docstring as the pointer to the real method.

---

## 7. Extension points

- **New estimators** (multitaper, Lomb-Scargle): add a `Spectrum.from_*`
  constructor. Nothing downstream changes, because downstream only sees `Spectrum`.
- **New uncertainties**: `floor(..., n_boot=...)`; the bootstrap is over segments,
  so it generalises to any estimator that exposes per-segment replicates.
- **The state axis** (the open `latent-floor` half): the same shape applies — a
  typed `Attractor`/`State` object, a computable bound, and a measured gap. The
  temporal half is this repo; the state half reuses this discipline.
