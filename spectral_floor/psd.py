"""Power spectral density estimation.

We estimate S(f) from observed time series with Welch's method.  Bias matters:
the floor is a ratio of <log S> to <S>, and log makes it sensitive to the
low-power tail that a biased estimator smears.

A note on debiasing, kept because it is a real trap.  A finite window convolves
the true spectrum with the window's leakage kernel, so the periodogram is biased.
The obvious correction -- dividing the averaged periodogram pointwise by the
kernel -- is *wrong*: leakage is a convolution, not a multiplier, so dividing by
it multiplies instead of undoing.  For a Hann window the normalised kernel is
0.25 at bin 1 and below 2e-6 from bin 2 on; floored at 1e-6 those bins get
multiplied by 5e5 to 1e6, turning a flat white-noise PSD into a rising ramp.  An
earlier version of this module shipped exactly that function.  It is deleted, not
"fixed": a correct deconvolution is ill-posed and is a research problem in its own
right (see arXiv:2312.13643, "Debiasing Welch's Method", Biometrika).  What we do
instead is report the estimator's segment count so the floor can carry an honest
confidence interval.
"""

from __future__ import annotations

import numpy as np


def normalise_psd(S: np.ndarray) -> np.ndarray:
    """Scale a PSD so that its mean is 1 (so it is a density, not a scale)."""
    S = np.asarray(S, dtype=float)
    m = S.mean()
    if m <= 0:
        raise ValueError("PSD must have positive mean")
    return S / m


def welch_psd(
    x: np.ndarray,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: bool = True,
    return_n_segments: bool = False,
    return_segments: bool = False,
) -> tuple:
    """Welch periodogram.

    Returns (freqs, psd) with freqs in cycles/sample in [0, 0.5].  With
    ``return_n_segments`` it appends the segment count; with ``return_segments``
    it appends the (n_segments, n_freq) stack of individual periodograms, which
    ``floor`` bootstraps over to get an honest confidence interval.

    Implemented directly (segmented periodogram + averaging) so the package has no
    scipy dependency.  `window` is one of {"hann", "hamming", "boxcar"}.
    """
    x = np.asarray(x, dtype=float).ravel()
    if detrend:
        x = x - x.mean()
    if noverlap is None:
        noverlap = nperseg // 2
    if noverlap >= nperseg:
        raise ValueError("noverlap must be < nperseg")
    step = nperseg - noverlap
    if len(x) < nperseg:
        raise ValueError(f"need at least nperseg={nperseg} samples, got {len(x)}")

    if window == "hann":
        w = np.hanning(nperseg)
    elif window == "hamming":
        w = np.hamming(nperseg)
    elif window == "boxcar":
        w = np.ones(nperseg)
    else:
        raise ValueError(f"unknown window {window!r}")

    # normalise window so the periodogram is unbiased for white noise
    wpow = (w ** 2).sum()
    nseg = 1 + (len(x) - nperseg) // step
    n_freq = nperseg // 2 + 1
    segs = np.empty((nseg, n_freq))
    for i in range(nseg):
        seg = x[i * step: i * step + nperseg] * w
        spec = np.fft.rfft(seg)
        segs[i] = (np.abs(spec) ** 2) / wpow
    psd = segs.mean(axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0)
    if return_segments:
        return freqs, psd, int(nseg), segs
    if return_n_segments:
        return freqs, psd, int(nseg)
    return freqs, psd


def two_sided_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    n_grid: int = 20001,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a one-sided PSD on [0, 0.5] to an even density on [0, 1).

    A real-valued stationary process has an even spectral density, ``S(f) = S(1-f)``.
    ``welch_psd`` returns the one-sided spectrum on ``[0, 0.5]`` -- all the
    information, since the upper band is its mirror image -- but the aliasing
    identity folds the whole ``[0, 1)`` band and therefore needs the full density.

    The obvious shortcut, interpolating the one-sided spectrum onto ``[0, 1)``
    with ``numpy.interp``, is wrong: ``interp`` clamps every frequency above 0.5
    to ``S(0.5)``, making the upper band constant. Since the folding step sums the
    shifted copies of ``S``, that error compounds with stride -- for AR(1) it makes
    the floor ~46% too low at stride 16. Mirroring is the exact construction.

    Returns ``(u, S)`` on a dense uniform grid over ``[0, 1)``, with ``S``
    normalised to unit mean.
    """
    freqs = np.asarray(freqs, dtype=float).ravel()
    psd = np.asarray(psd, dtype=float).ravel()
    if freqs.shape != psd.shape:
        raise ValueError("freqs and psd must have the same shape")
    if freqs.size < 2:
        raise ValueError("need at least two frequency bins")
    # Reflect the interior bins; keep DC and Nyquist each exactly once. Append the
    # endpoint at 1.0 (== 0.0 on the circle) so the grid spans all of [0, 1) and
    # `interp` never clamps -- otherwise the last sliver of the band is flat and the
    # result is very slightly non-even.
    f_full = np.concatenate([freqs, 1.0 - freqs[-2:0:-1], [1.0]])
    S_full = np.concatenate([psd, psd[-2:0:-1], [psd[0]]])
    u = np.linspace(0.0, 1.0, n_grid)[:-1]
    return u, normalise_psd(np.interp(u, f_full, S_full))


def ar1_psd(a: float, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Analytic AR(1) PSD:  S(u) = 1 / |1 - a e^{-i 2 pi u}|^2.

    `u` is normalised frequency in [0, 1).  Returns (u, S).  This is the exact
    spectrum of x[n] = a x[n-1] + e[n], and it is what we use as ground truth when
    checking the estimator and the floor.
    """
    if u is None:
        u = np.linspace(0, 1, 200001)[:-1]
    u = np.asarray(u, dtype=float)
    S = 1.0 / np.abs(1.0 - a * np.exp(-2j * np.pi * u)) ** 2
    return u, S
