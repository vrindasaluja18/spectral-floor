"""Power spectral density estimation.

We estimate S(f) from observed time series with Welch's method, and optionally
debias it.  Bias matters: the floor is a ratio of <log S> to <S>, and log makes it
sensitive to the low-power tail that a biased estimator smears.
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
) -> tuple[np.ndarray, np.ndarray]:
    """Welch periodogram.

    Returns (freqs, psd) with freqs in cycles/sample in [0, 0.5].

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
    acc = np.zeros(nperseg // 2 + 1)
    for i in range(nseg):
        seg = x[i * step: i * step + nperseg] * w
        spec = np.fft.rfft(seg)
        acc += (np.abs(spec) ** 2) / wpow
    psd = acc / nseg
    freqs = np.fft.rfftfreq(nperseg, d=1.0)
    return freqs, psd


def debias_welch(
    x: np.ndarray,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Welch with a first-order bias correction.

    The plain periodogram of a finite window is a *biased* estimate of the true
    spectrum: it convolves S with the window's spectral leakage kernel.  A standard
    cheap correction is to divide by the squared window's transfer function, which
    we apply in the frequency domain on the averaged periodogram.

    This is a pragmatic correction, not an exact inverse (which is ill-posed).  For
    a principled treatment see arXiv:2312.13643 ("Debiasing Welch's Method").
    """
    freqs, psd = welch_psd(x, nperseg=nperseg, noverlap=noverlap, window=window)
    if window == "boxcar":
        return freqs, psd
    w = np.hanning(nperseg) if window == "hann" else np.hamming(nperseg)
    # |W(f)|^2 normalised to unit DC gain
    Wf = np.abs(np.fft.rfft(w, n=2 * nperseg)) ** 2
    Wf = Wf / Wf[0]
    # resample the kernel onto our freq grid and divide
    grid = np.linspace(0, 0.5, len(Wf))
    kernel = np.interp(freqs, grid, Wf)
    corrected = psd / np.maximum(kernel, 1e-6)
    return freqs, corrected


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
