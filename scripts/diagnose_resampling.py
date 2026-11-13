"""Is the 'floor(est)' gap at large stride a Welch bias, or a resampling bug?

run() does:
    S_hat = welch_psd(x)          # one-sided, freqs in [0, 0.5]
    grid  = linspace(0, 0.5, len(S_hat))
    u     = linspace(0, 1, 20001)[:-1]
    S     = interp(u, grid, S_hat)   # <-- np.interp CLAMPS u > 0.5

A real signal's PSD is symmetric: S(f) = S(1-f). Clamping instead of
mirroring makes S constant on (0.5, 1), which is wrong. Decimation folds
the upper half onto the lower half, so the error is amplified with stride.

This script compares three ways of building the full [0,1) spectrum from a
one-sided estimate, on a process whose TRUE spectrum we know exactly (AR(1)).
"""

import numpy as np

from spectral_floor import normalise_psd, szego, alias_spectrum, ar1_psd

a = 0.99
u = np.linspace(0, 1, 20001)[:-1]
_, S_true = ar1_psd(a, u)
S_true = normalise_psd(S_true)                    # two-sided density on [0,1)

# one-sided half, as welch_psd would return it
half = S_true[: len(S_true) // 2 + 1]             # S on [0, 0.5]
grid = np.linspace(0, 0.5, len(half))


def build_clamped():
    """What run() currently does: np.interp clamps u > 0.5 to S(0.5)."""
    return normalise_psd(np.interp(u, grid, half))


def build_mirrored():
    """Correct: a real signal's PSD is even, S(f) = S(1-f)."""
    full = np.concatenate([half, half[-2:0:-1]])   # length 2*len(half)-1
    uf = np.linspace(0, 1, len(full))
    return normalise_psd(np.interp(u, uf, full))


def floor_of(S, s):
    return szego(alias_spectrum(normalise_psd(S), s))


print(f"  {'stride':>6}  {'TRUE':>9}  {'clamped':>9}  {'mirrored':>9}  {'closed 1-a^2s':>13}")
print("  " + "-" * 56)
for s in (1, 2, 4, 8, 16, 32):
    print(f"  {s:>6}  {floor_of(S_true, s):>9.5f}  {floor_of(build_clamped(), s):>9.5f}  "
          f"{floor_of(build_mirrored(), s):>9.5f}  {1 - a ** (2 * s):>13.5f}")

print()
print("READ: 'clamped' must reproduce the README's floor(est)=0.153 at s=16.")
print("      'mirrored' must track the TRUE / closed-form column.")
print("      If so, the README's 'Welch bias' explanation is wrong: it is a")
print("      resampling bug, and it is fixable exactly.")
