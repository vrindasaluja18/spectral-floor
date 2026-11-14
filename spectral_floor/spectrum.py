"""The Spectrum: a parsed, self-describing spectral density.

"Parse, don't validate" (Alexis King, 2019).  Rather than passing bare arrays
around and hoping every callee guesses what they mean, we make the meaning part
of the type.  A ``Spectrum`` knows its grid, its sidedness, its normalisation and
how many segments it was averaged from, and the floor routines accept nothing
else.  A raw ``np.ndarray`` cannot be mistaken for a spectrum, so the class of
bug where a one-sided PSD is silently folded as if it were two-sided is removed
by construction rather than caught by a check.

The invariant that matters most is *sidedness*.  The aliasing identity folds the
whole unit circle,

    S_dec(v) = (1/s) sum_{k=0}^{s-1} S((v + k)/s),

so it needs the two-sided density on ``[0, 1)``.  ``Spectrum.from_welch`` always
returns that form; a one-sided spectrum must be expanded with
``.two_sided()`` (a mirror, never an interpolation) before it may be folded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ONE_SIDED = "one-sided"
TWO_SIDED = "two-sided"
UNIT_MEAN = "unit-mean"
RAW = "raw"

_SIDEDNESS = (ONE_SIDED, TWO_SIDED)
_NORMALISATION = (UNIT_MEAN, RAW)


@dataclass(frozen=True, eq=False)
class Spectrum:
    """A power spectral density together with everything needed to interpret it.

    Parameters
    ----------
    values:
        The density, sampled on ``grid``.  Non-negative and finite.
    grid:
        Frequencies.  Uniformly spaced, strictly increasing, within ``[0, 0.5]``
        for a one-sided spectrum and ``[0, 1)`` for a two-sided one.  Cycles per
        sample (a normalised frequency, so ``1.0`` is one cycle per sample).
    sidedness:
        ``"one-sided"`` (on ``[0, 0.5]``) or ``"two-sided"`` (on ``[0, 1)``).
    normalisation:
        ``"unit-mean"`` (a density, mean == 1) or ``"raw"`` (carries absolute
        power).  The floor is a ratio and so is invariant to this, but tracking
        it keeps the object honest.
    n_segments:
        Number of independent periodogram segments averaged to form the
        estimate.  Sets the effective sample size and the confidence interval.
    estimator:
        Human-readable name of the estimator, e.g. ``"welch"`` or ``"analytic"``.
    sample_rate:
        Sampling rate of the underlying signal, in Hz.  ``1.0`` means the grid
        is in cycles/sample.
    nperseg:
        Welch segment length, if the estimate came from Welch.  ``0`` for an
        analytic or hand-built spectrum.  Sets the number of independent
        frequency cells, and hence the confidence-interval width.
    exact:
        ``True`` if the values are the true density (analytic ground truth), so
        there is no estimator bias to correct and the interval is degenerate.
    segments:
        Optional ``(n_segments, n_freq)`` stack of individual one-sided
        periodograms.  When present, ``floor`` bootstraps over it for an honest
        confidence interval, because Welch bins are correlated and a naive
        analytic variance is far too narrow.
    segment_freqs:
        Frequencies of ``segments``, in ``[0, 0.5]``.
    """

    values: np.ndarray
    grid: np.ndarray
    sidedness: str
    normalisation: str
    n_segments: int
    estimator: str = "welch"
    sample_rate: float = 1.0
    nperseg: int = 0
    exact: bool = False
    segments: np.ndarray | None = None
    segment_freqs: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float).ravel()
        grid = np.array(self.grid, dtype=float).ravel()
        if values.shape != grid.shape:
            raise ValueError(
                f"values and grid must have the same shape, got {values.shape} "
                f"and {grid.shape}"
            )
        if values.size < 2:
            raise ValueError("a Spectrum needs at least two frequency bins")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectrum values must be finite")
        if np.any(values < 0):
            raise ValueError("spectrum values must be non-negative")
        if values.mean() <= 0:
            raise ValueError("spectrum must have positive mean (all-zero is degenerate)")
        if self.sidedness not in _SIDEDNESS:
            raise ValueError(
                f"sidedness must be one of {_SIDEDNESS}, got {self.sidedness!r}"
            )
        if self.normalisation not in _NORMALISATION:
            raise ValueError(
                f"normalisation must be one of {_NORMALISATION}, got "
                f"{self.normalisation!r}"
            )
        if int(self.n_segments) < 1:
            raise ValueError("n_segments must be >= 1")
        _validate_grid(grid, self.sidedness)
        if self.normalisation == UNIT_MEAN:
            m = values.mean()
            if not np.isclose(m, 1.0, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"normalisation='unit-mean' but the mean is {m:.6g}; either "
                    f"normalise the values or pass normalisation='raw'"
                )
        values.flags.writeable = False
        grid.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "n_segments", int(self.n_segments))
        if self.segments is not None:
            segs = np.array(self.segments, dtype=float)
            if segs.ndim != 2:
                raise ValueError("segments must be a 2-D (n_segments, n_freq) array")
            if segs.shape[0] != int(self.n_segments):
                raise ValueError(
                    f"segments has {segs.shape[0]} rows but n_segments is "
                    f"{self.n_segments}"
                )
            if self.segment_freqs is None:
                raise ValueError("segments requires segment_freqs")
            sfreqs = np.array(self.segment_freqs, dtype=float).ravel()
            if sfreqs.size != segs.shape[1]:
                raise ValueError("segment_freqs length must match segments columns")
            segs.flags.writeable = False
            sfreqs.flags.writeable = False
            object.__setattr__(self, "segments", segs)
            object.__setattr__(self, "segment_freqs", sfreqs)

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_welch(
        cls,
        x: np.ndarray,
        nperseg: int = 256,
        noverlap: int | None = None,
        window: str = "hann",
        detrend: bool = True,
        n_grid: int = 20001,
        sample_rate: float = 1.0,
    ) -> "Spectrum":
        """Welch estimate, expanded to the two-sided density on ``[0, 1)``.

        This is the blessed entry point: it always returns the two-sided form
        the aliasing identity needs, so a caller cannot accidentally fold a
        one-sided spectrum.  The expansion is a mirror (``S(f) = S(1-f)``), never
        an interpolation, so the upper band is not clamped.
        """
        from .psd import welch_psd, two_sided_psd

        freqs, psd, n_segments, segs = welch_psd(
            x, nperseg=nperseg, noverlap=noverlap, window=window,
            detrend=detrend, return_segments=True,
        )
        u, values = two_sided_psd(freqs, psd, n_grid=n_grid)
        return cls(
            values=values,
            grid=u,
            sidedness=TWO_SIDED,
            normalisation=UNIT_MEAN,
            n_segments=n_segments,
            estimator="welch",
            sample_rate=sample_rate,
            nperseg=int(nperseg),
            segments=segs,
            segment_freqs=freqs,
        )

    @classmethod
    def from_psd(
        cls,
        values: np.ndarray,
        grid: np.ndarray,
        sidedness: str = TWO_SIDED,
        normalisation: str = UNIT_MEAN,
        n_segments: int = 1,
        estimator: str = "unknown",
        sample_rate: float = 1.0,
        nperseg: int = 0,
        exact: bool = False,
    ) -> "Spectrum":
        """Build a Spectrum from a known PSD (analytic ground truth, fixtures)."""
        return cls(
            values=values,
            grid=grid,
            sidedness=sidedness,
            normalisation=normalisation,
            n_segments=n_segments,
            estimator=estimator,
            sample_rate=sample_rate,
            nperseg=nperseg,
            exact=exact,
        )

    @classmethod
    def from_ar1(cls, a: float, n: int = 200001, sample_rate: float = 1.0) -> "Spectrum":
        """Exact analytic AR(1) density on the unit circle: the oracle."""
        from .psd import ar1_psd

        u, S = ar1_psd(a, np.linspace(0.0, 1.0, n)[:-1])
        return cls(
            values=S / S.mean(),
            grid=u,
            sidedness=TWO_SIDED,
            normalisation=UNIT_MEAN,
            n_segments=10**9,  # analytic: effectively infinite sample
            estimator="analytic-ar1",
            sample_rate=sample_rate,
            exact=True,
        )

    # -- conversions --------------------------------------------------------
    def two_sided(self) -> "Spectrum":
        """The even density on ``[0, 1)``, mirroring a one-sided spectrum."""
        if self.sidedness == TWO_SIDED:
            return self
        from .psd import two_sided_psd

        u, values = two_sided_psd(self.grid, self.values)
        return Spectrum(
            values=values,
            grid=u,
            sidedness=TWO_SIDED,
            normalisation=UNIT_MEAN,
            n_segments=self.n_segments,
            estimator=self.estimator,
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            exact=self.exact,
        )

    def one_sided(self) -> "Spectrum":
        """The half-band density on ``[0, 0.5]``."""
        if self.sidedness == ONE_SIDED:
            return self
        half = len(self.grid) // 2 + 1
        grid = self.grid[:half]
        values = self.values[:half]
        return Spectrum(
            values=values / values.mean(),
            grid=grid,
            sidedness=ONE_SIDED,
            normalisation=UNIT_MEAN,
            n_segments=self.n_segments,
            estimator=self.estimator,
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            exact=self.exact,
        )

    def resample(self, n: int) -> "Spectrum":
        """Linearly resample onto a uniform grid of ``n`` points, preserving band."""
        if n < 2:
            raise ValueError("need at least two grid points")
        if self.sidedness == TWO_SIDED:
            grid = np.linspace(0.0, 1.0, n + 1)[:-1]
        else:
            grid = np.linspace(0.0, 0.5, n)
        values = np.interp(grid, self.grid, self.values)
        if self.normalisation == UNIT_MEAN:
            values = values / values.mean()
        return Spectrum(
            values=values,
            grid=grid,
            sidedness=self.sidedness,
            normalisation=self.normalisation,
            n_segments=self.n_segments,
            estimator=self.estimator,
            sample_rate=self.sample_rate,
            nperseg=self.nperseg,
            exact=self.exact,
        )

    # -- diagnostics --------------------------------------------------------
    def mean_power(self) -> float:
        """Mean of the density over its band."""
        return float(self.values.mean())

    def n_zero_bins(self) -> int:
        """Number of exactly-zero bins (a deterministic component)."""
        return int(np.count_nonzero(self.values == 0.0))

    def is_log_integrable(self) -> bool:
        """True iff ``log S`` is finite everywhere, so the floor is well posed."""
        return self.n_zero_bins() == 0

    def dynamic_range_db(self) -> float:
        """Peak-to-trough range in dB over the non-zero bins."""
        pos = self.values[self.values > 0]
        if pos.size == 0:
            return float("inf")
        return float(10.0 * np.log10(pos.max() / pos.min()))

    def summary(self) -> str:
        lo, hi = float(self.grid[0]), float(self.grid[-1])
        return (
            f"Spectrum({self.sidedness}, {self.estimator}, "
            f"{self.values.size} bins on [{lo:.4g}, {hi:.4g}], "
            f"{self.n_segments} segments, dynamic range "
            f"{self.dynamic_range_db():.1f} dB)"
        )

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _validate_grid(grid: np.ndarray, sidedness: str) -> None:
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid must be finite")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be strictly increasing")
    diffs = np.diff(grid)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "grid must be uniform; resample with Spectrum.resample(n) first"
        )
    if sidedness == ONE_SIDED:
        if grid[0] < -1e-9 or grid[-1] > 0.5 + 1e-9:
            raise ValueError("a one-sided grid must lie within [0, 0.5]")
    else:
        if grid[0] < -1e-9 or grid[-1] >= 1.0 - 1e-9:
            raise ValueError("a two-sided grid must lie within [0, 1)")
