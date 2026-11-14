"""Is the 'floor(est)' gap at large stride a Welch bias, or a resampling bug?

The buggy pattern was:

    S_hat = welch_psd(x)             # one-sided, freqs in [0, 0.5]
    grid  = linspace(0, 0.5, len(S_hat))
    u     = linspace(0, 1, 20001)[:-1]
    S     = interp(u, grid, S_hat)   # <-- np.interp CLAMPS u > 0.5

A real signal's PSD is symmetric: S(f) = S(1-f). Clamping instead of mirroring
makes S constant on (0.5, 1), which is wrong. Decimation folds the upper half onto
the lower half, so the error is amplified with stride.

With the typed Spectrum this bug cannot be written: ``Spectrum.from_welch`` always
returns the mirrored two-sided density, and the fold routines accept only a
Spectrum.  This script compares the wrong construction (clamped) with the right one
(mirrored) on a process whose TRUE spectrum we know exactly (AR(1)), so the
difference is isolated from any estimator.
"""

import numpy as np

from spectral_floor import (
    Spectrum, UNIT_MEAN, floor, ar1_floor, normalise_psd,
)

a = 0.99
S_true = Spectrum.from_ar1(a)
half = S_true.one_sided()

# The clamped construction, as a Spectrum so we can fold it: interpolate the
# one-sided density onto [0, 1) with np.interp, which clamps above 0.5.  (The
# original code then renormalised; we do the same so only the *shape* differs.)
u = S_true.grid
S_clamped = Spectrum.from_psd(
    normalise_psd(np.interp(u, half.grid, half.values)), u,
    normalisation=UNIT_MEAN, exact=True,
)

print(f"  {'stride':>6}  {'TRUE':>9}  {'clamped':>9}  {'closed 1-a^2s':>13}  {'clamp err':>10}")
print("  " + "-" * 60)
for s in (1, 2, 4, 8, 16, 32):
    f_true = floor(S_true, s).eps2
    f_clamp = floor(S_clamped, s).eps2
    closed = ar1_floor(a, s)
    err = (f_clamp / f_true - 1.0) * 100.0
    print(f"  {s:>6}  {f_true:>9.5f}  {f_clamp:>9.5f}  {closed:>13.5f}  {err:>9.1f}%")

print()
print("READ: 'clamped' is biased low, and the bias grows with stride.")
print("      'TRUE' tracks the closed form.  So the old gap was a resampling bug,")
print("      not Welch bias -- and the typed Spectrum removes the failure mode.")
