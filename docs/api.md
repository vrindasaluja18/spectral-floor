# API

Everything is importable from the top-level package:

```python
from spectral_floor import (
    Spectrum, FloorResult,                      # types
    ONE_SIDED, TWO_SIDED, UNIT_MEAN, RAW,       # labels
    welch_psd, ar1_psd, normalise_psd, two_sided_psd,   # estimation
    szego, alias_spectrum, floor, floor_curve, ar1_floor,  # the floor
    ar1_design_law, design_law, optimal_stride,  # the decision layer
)
```

---

## `Spectrum`

A frozen, self-describing power spectral density. The floor routines accept only
this type, so a raw array cannot be mistaken for a spectrum.

```python
Spectrum(
    values, grid, sidedness, normalisation, n_segments,
    estimator="welch", sample_rate=1.0, nperseg=0, exact=False,
    segments=None, segment_freqs=None,
)
```

| Field | Meaning |
|---|---|
| `values` | the density, non-negative and finite |
| `grid` | uniform, increasing frequencies; `[0,0.5]` one-sided, `[0,1)` two-sided |
| `sidedness` | `ONE_SIDED` or `TWO_SIDED` |
| `normalisation` | `UNIT_MEAN` (mean == 1) or `RAW` |
| `n_segments` | averaged periodograms; sets the effective sample size |
| `estimator` | e.g. `"welch"`, `"analytic-ar1"` |
| `sample_rate` | Hz; `1.0` means the grid is in cycles/sample |
| `nperseg` | Welch segment length (`0` if not from Welch) |
| `exact` | `True` for an analytic density: no bias, degenerate interval |
| `segments`, `segment_freqs` | per-segment periodograms, for the bootstrap interval |

Invariants (shape, finiteness, non-negativity, positive mean, grid uniformity and
range, unit-mean consistency) are checked in `__post_init__`. Arrays are stored
read-only.

### Constructors

```python
Spectrum.from_welch(x, nperseg=256, noverlap=None, window="hann",
                    detrend=True, n_grid=20001, sample_rate=1.0)
```
Welch estimate, **always expanded to the two-sided density on `[0,1)`** by
mirroring (`S(f) = S(1-f)`), never by interpolation. This is the blessed entry
point from data.

```python
Spectrum.from_psd(values, grid, sidedness=TWO_SIDED, normalisation=UNIT_MEAN,
                  n_segments=1, estimator="unknown", sample_rate=1.0,
                  nperseg=0, exact=False)
```
Build from a known PSD (fixtures, analytic ground truth).

```python
Spectrum.from_ar1(a, n=200001, sample_rate=1.0)
```
The exact AR(1) density: the oracle. `exact=True`, `n_segments` effectively
infinite, degenerate interval.

### Conversions

| Method | Effect |
|---|---|
| `.two_sided()` | mirror a one-sided spectrum onto `[0,1)` (no-op if already) |
| `.one_sided()` | keep the half band `[0,0.5]` |
| `.resample(n)` | linear resample onto `n` uniform points, band preserved |

### Diagnostics

| Method | Returns |
|---|---|
| `.mean_power()` | mean of the density over its band |
| `.n_zero_bins()` | count of exactly-zero bins |
| `.is_log_integrable()` | `True` iff `log S` is finite everywhere |
| `.dynamic_range_db()` | peak-to-trough range over the non-zero bins |
| `.summary()` | one-line description |

---

## The floor

### `floor(S, stride=1, confidence=0.95, n_boot=200, seed=0) -> FloorResult`

The floor at one stride. Point estimate is bias-corrected for the log-chi-square
bias of a Welch estimate (`E[log Ŝ] = log S + ψ(K) − log K`); the interval is a
percentile bootstrap over the `K` segments. Pass `n_boot=0` to skip the bootstrap.

`FloorResult` fields: `eps2`, `stride`, `estimator`, `n_segments`, `ess`
(effective sample size, `2K`), `ci_low`, `ci_high`, `warnings`.

### `floor_curve(S, strides) -> dict[int, FloorResult]`

`floor` for each stride.

### `szego(S) -> float`

The bare ratio `exp(⟨log S⟩)/⟨S⟩` on the spectrum as given. Emits a `UserWarning`
on zero bins (log divergence) and on extreme dynamic range. Use `floor` for the
version with provenance and uncertainty.

### `alias_spectrum(S, s) -> Spectrum`

The stride-`s` fold. Expands a one-sided `S` via `.two_sided()` first, interpolates
periodically (no clamp), and normalises with the `1/s` factor so mean power is
preserved.

### `ar1_floor(a, s) -> float` / `ar1_design_law(a, strides) -> dict[int, float]`

Closed forms for AR(1): `1 − a^(2s)` and `(1 − a^(2s))/(1 − a²)`. The oracle.

---

## The decision layer

### `design_law(S, strides) -> dict[int, float]`

Relative error inflation vs stride 1, from a *measured* spectrum:
`floor(S,s).eps2 / floor(S,1).eps2`. The spectrum-general form of
`ar1_design_law`; it needs no AR(1) assumption.

### `optimal_stride(S, max_inflation=2.0, max_stride=1024) -> int`

The largest stride whose inflation stays within `max_inflation`. `max_inflation`
must exceed 1 (a budget of 1 leaves only stride 1 — which is why the old default of
1.0 was useless).

---

## Estimation primitives

### `welch_psd(x, nperseg=256, noverlap=None, window="hann", detrend=True, return_n_segments=False, return_segments=False)`

Segmented periodogram + averaging, no SciPy. Returns `(freqs, psd)`; optionally
appends the segment count, or the `(n_segments, n_freq)` stack of individual
periodograms.

### `two_sided_psd(freqs, psd, n_grid=20001) -> (u, S)`

Expand a one-sided PSD on `[0,0.5]` to the even density on `[0,1)` by mirroring.
Returns a unit-mean density. The correct construction: interpolating instead
clamps the upper band and biases the floor low (46% at stride 16).

### `ar1_psd(a, u=None) -> (u, S)` / `normalise_psd(S) -> S`

Analytic AR(1) density; and scaling a PSD to unit mean.

---

## Example

```python
import numpy as np
from spectral_floor import Spectrum, floor, design_law, optimal_stride

rng = np.random.default_rng(0)
x = np.zeros(200_000)
e = rng.normal(0, 1, 200_000)
for i in range(1, len(x)):
    x[i] = 0.99 * x[i - 1] + e[i]

S = Spectrum.from_welch(x, nperseg=1024)
r = floor(S, stride=4)
print(r)                       # floor(s=4) = 0.075995 [0.07108, 0.07995] (welch, ess=778)

print(design_law(S, [1, 2, 4, 8, 16]))   # {1: 1.0, 2: 1.97, 4: 3.87, 8: 7.45, 16: 13.89}
print(optimal_stride(S, max_inflation=2.0))   # 2
```
