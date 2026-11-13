"""The Kolmogorov-Szego spectral floor, and the effect of temporal stride.

For a stationary process with spectral density S the optimal one-step-ahead
relative MSE of any linear predictor is

        eps2 = exp( <log S> ) / <S>

(Kolmogorov 1941; Szego 1920).  Decimating by s folds the spectrum:

        S_dec(v) = sum_{k=0}^{s-1} S( (v + k) / s )

so the floor of a world model observing at rate R/s is szego(alias_spectrum(S, s)).

For an AR(1) process this collapses to the closed form  eps2(s) = 1 - a^(2s).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-300


def szego(S: np.ndarray) -> float:
    """Kolmogorov-Szego relative prediction error floor.

    Returns eps2 = exp(<log S>) / <S>.  Lower values mean more predictable.
    The spectrum is compared on a uniform grid over the unit circle, so S should be
    (a discrete sampling of) a density on [0, 1).
    """
    S = np.asarray(S, dtype=float).ravel()
    if S.size == 0:
        raise ValueError("empty spectrum")
    if np.any(S < 0):
        raise ValueError("spectrum must be non-negative")
    if S.mean() <= 0:
        raise ValueError("spectrum must have positive mean")
    logmean = np.mean(np.log(np.maximum(S, _EPS)))
    return float(np.exp(logmean) / S.mean())


def alias_spectrum(S: np.ndarray, s: int) -> np.ndarray:
    """Spectrum of the stride-s decimated process.

    `S` is sampled uniformly on [0, 1).  Implements the decimation identity

        S_dec(v) = (1/s) sum_{k=0}^{s-1} S((v + k) / s)

    The 1/s factor matters: it makes the result a proper density, so the total
    power (the mean of the PSD) is preserved, matching Var(y) = Var(x) for the
    subsampled process y[m] = x[m s].  Note that `szego` is invariant to a uniform
    rescaling, so the floor itself is unaffected by this factor - but returning a
    correctly normalised spectrum keeps the function usable on its own.
    """
    if s < 1:
        raise ValueError("stride must be >= 1")
    S = np.asarray(S, dtype=float).ravel()
    if s == 1:
        return S.copy()
    u = np.linspace(0, 1, S.size + 1)[:-1]
    out = np.zeros_like(S)
    for k in range(s):
        out += np.interp((u + k) / s % 1.0, u, S)
    return out / s


def floor_curve(S: np.ndarray, strides) -> dict[int, float]:
    """Floor for each stride in `strides`."""
    return {int(s): szego(alias_spectrum(S, int(s))) for s in strides}


def ar1_floor(a: float, s: int) -> float:
    """Closed-form AR(1) floor after stride s:  1 - a^(2s)."""
    if not -1.0 < a < 1.0:
        raise ValueError("|a| must be < 1 for stationarity")
    if s < 1:
        raise ValueError("stride must be >= 1")
    return float(1.0 - a ** (2 * s))


def design_law(a: float, strides) -> dict[int, float]:
    """Relative error inflation vs stride 1:  (1 - a^(2s)) / (1 - a^2)."""
    if not -1.0 < a < 1.0:
        raise ValueError("|a| must be < 1")
    base = 1.0 - a * a
    return {int(s): ar1_floor(a, int(s)) / base for s in strides}


def optimal_stride(a: float, max_inflation: float = 1.0) -> int:
    """Largest stride whose error inflation stays within `max_inflation`.

    The design rule: pick the cheapest stride you can afford, i.e. the largest s
    such that (1 - a^(2s)) / (1 - a^2) <= max_inflation.
    """
    if max_inflation < 1.0:
        raise ValueError("max_inflation must be >= 1 (stride 1 is the baseline)")
    s = 1
    while design_law(a, [s + 1])[s + 1] <= max_inflation:
        s += 1
        if s > 4096:
            break
    return s
