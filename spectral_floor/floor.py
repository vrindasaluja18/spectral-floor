"""The Kolmogorov-Szego spectral floor, and the effect of temporal stride.

For a stationary process with spectral density S the optimal one-step-ahead
relative MSE of any linear predictor is

        eps2 = exp( <log S> ) / <S>

(Kolmogorov 1941; Szego 1920).  Decimating by s folds the spectrum:

        S_dec(v) = (1/s) sum_{k=0}^{s-1} S( (v + k) / s )

so the floor of a world model observing at rate R/s is szego(alias_spectrum(S, s)).

Every function here takes a :class:`~spectral_floor.spectrum.Spectrum`, never a
bare array.  The aliasing identity folds the whole unit circle, so it needs the
*two-sided* density; a ``Spectrum`` carries its sidedness, and ``alias_spectrum``
calls ``.two_sided()`` itself rather than trusting the caller.  That removes the
failure mode where a one-sided Welch estimate is folded as if it were two-sided --
it is the same class of silent bug as the clamp.

``floor`` returns a :class:`FloorResult` carrying the value *and* its provenance:
the estimator, the effective sample size, a bootstrap confidence interval, and any
warnings about the estimate.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .spectrum import Spectrum, TWO_SIDED, UNIT_MEAN

# Warn when the discrete floor is unsafe.  A zero bin makes ``log S`` diverge, so
# the discrete floor is 0 and is an artefact of the grid, not of the process.
_DYNAMIC_RANGE_WARN_DB = 100.0
_MIN_SEGMENTS_WARN = 8
_EULER_GAMMA = 0.5772156649015329
_TRIGAMMA_ONE = np.pi ** 2 / 6.0


@dataclass(frozen=True)
class FloorResult:
    """The floor at one stride, with its provenance.

    Attributes
    ----------
    eps2:
        The bias-corrected relative-MSE floor.
    stride:
        The decimation stride the floor is evaluated at.
    estimator:
        Name of the spectrum estimator the floor was computed from.
    n_segments:
        Number of averaged periodogram segments in the estimate.
    ess:
        Effective sample size: the equivalent degrees of freedom ``2 * n_segments``
        of a Welch estimate (each real periodogram ordinate contributes 2).
    ci_low, ci_high:
        Confidence interval on ``eps2``, from a percentile bootstrap over the
        Welch segments (degenerate when the spectrum is exact).
    warnings:
        Human-readable caveats: zero bins, extreme dynamic range, few segments.
    """

    eps2: float
    stride: int
    estimator: str
    n_segments: int
    ess: float
    ci_low: float
    ci_high: float
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        ci = f"[{self.ci_low:.5f}, {self.ci_high:.5f}]"
        out = (
            f"floor(s={self.stride}) = {self.eps2:.6f}  {ci}  "
            f"({self.estimator}, ess={self.ess:g})"
        )
        for w in self.warnings:
            out += f"\n  warning: {w}"
        return out


# ---------------------------------------------------------------------------
# digamma / trigamma at positive integers (exact via harmonic sums)
# ---------------------------------------------------------------------------
def _digamma_int(k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 10 ** 6:
        return float(np.log(k) - 0.5 / k - 1.0 / (12.0 * k * k))
    return float(-_EULER_GAMMA + sum(1.0 / i for i in range(1, k)))


def _trigamma_int(k: int) -> float:
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 10 ** 6:
        return float(1.0 / k + 0.5 / (k * k) + 1.0 / (6.0 * k ** 3))
    return float(_TRIGAMMA_ONE - sum(1.0 / (i * i) for i in range(1, k)))


def _independent_bins(S: Spectrum) -> int:
    """Number of statistically independent frequency cells in the estimate."""
    if S.nperseg and S.nperseg >= 2:
        return max(1, S.nperseg // 2)
    # Fallback for a hand-built spectrum: a two-sided grid is a mirror of the
    # half band, so it carries about a quarter of its bins of fresh information.
    return max(1, S.values.size // 4)


# ---------------------------------------------------------------------------
# the floor itself
# ---------------------------------------------------------------------------
def _szego_core(values: np.ndarray) -> tuple[float, list[str]]:
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("empty spectrum")
    if np.any(values < 0):
        raise ValueError("spectrum must be non-negative")
    mean = values.mean()
    if mean <= 0:
        raise ValueError("spectrum must have positive mean")

    messages: list[str] = []
    n_zero = int(np.count_nonzero(values == 0.0))
    if n_zero:
        logmean = -np.inf
        messages.append(
            f"{n_zero} zero-power bin(s): log S diverges, so the discrete floor "
            f"is 0 and is an artefact of the grid. The process may be "
            f"deterministic, or the estimate under-resolved."
        )
    else:
        logmean = float(np.log(values).mean())

    pos = values[values > 0]
    if pos.size:
        dr_db = float(10.0 * np.log10(pos.max() / pos.min()))
        if dr_db > _DYNAMIC_RANGE_WARN_DB:
            messages.append(
                f"spectral dynamic range {dr_db:.0f} dB exceeds "
                f"{_DYNAMIC_RANGE_WARN_DB:.0f} dB: the floor is set by the "
                f"low-power tail and is sensitive to estimator bias."
            )

    return float(np.exp(logmean) / mean), messages


def szego(S: Spectrum) -> float:
    """Kolmogorov-Szego relative prediction error floor, ``exp(<log S>)/<S>``.

    Takes a :class:`Spectrum`.  The value is a ratio of means over the same band,
    so it is invariant to sidedness and to a uniform rescaling; only the shape of
    the density matters.  Emits a warning when the estimate makes the value unsafe.
    """
    if not isinstance(S, Spectrum):
        raise TypeError(
            f"szego expects a Spectrum, got {type(S).__name__}; build one with "
            f"Spectrum.from_welch(x) or Spectrum.from_psd(values, grid, ...)"
        )
    eps2, messages = _szego_core(S.values)
    for m in messages:
        _warnings.warn(m, stacklevel=2)
    return eps2


def _periodic_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation treating ``(xp, fp)`` as periodic on [0, 1)."""
    xp_ext = np.concatenate([xp - 1.0, xp, xp + 1.0])
    fp_ext = np.concatenate([fp, fp, fp])
    return np.interp(x, xp_ext, fp_ext)


def alias_spectrum(S: Spectrum, s: int) -> Spectrum:
    """Spectrum of the stride-s decimated process.

    Implements the decimation identity

        S_dec(v) = (1/s) sum_{k=0}^{s-1} S( (v + k) / s )

    ``S`` must describe the full band; a one-sided spectrum is expanded with
    ``.two_sided()`` first (a mirror, not an interpolation).  The shifts wrap
    *periodically* around the unit circle, so the last sliver of the band is
    interpolated across the wrap and never clamped.  The ``1/s`` factor keeps the
    result a proper density (mean power is preserved), so it stands alone.
    """
    if s < 1:
        raise ValueError("stride must be >= 1")
    if not isinstance(S, Spectrum):
        raise TypeError(f"alias_spectrum expects a Spectrum, got {type(S).__name__}")
    S = S.two_sided()
    if s == 1:
        return S

    out = np.zeros_like(S.values)
    for k in range(s):
        out += _periodic_interp((S.grid + k) / s, S.grid, S.values)
    out /= s
    if S.normalisation == UNIT_MEAN:
        out = out / out.mean()
    return Spectrum(
        values=out,
        grid=S.grid,
        sidedness=TWO_SIDED,
        normalisation=S.normalisation,
        n_segments=S.n_segments,
        estimator=S.estimator,
        sample_rate=S.sample_rate,
        nperseg=S.nperseg,
    )


def _bootstrap_ci(
    S: Spectrum,
    stride: int,
    confidence: float,
    n_boot: int,
    bias_correction: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap interval over the Welch segments.

    Welch frequency bins are strongly correlated (the window's leakage kernel
    couples neighbouring bins), so the naive ``psi1(K)/M`` variance is far too
    narrow -- measured 165x too narrow on AR(1).  Resampling the *segments* and
    recomputing the whole mirror-and-fold pipeline captures that correlation
    directly, because the same folding is applied to every replicate.
    """
    from .psd import two_sided_psd

    segs = S.segments
    freqs = S.segment_freqs
    K = segs.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, K, K)
        psd_b = segs[idx].mean(axis=0)
        u, full = two_sided_psd(freqs, psd_b, n_grid=S.values.size)
        S_b = Spectrum.from_psd(full, u, n_segments=K, normalisation=UNIT_MEAN)
        eps_b, _ = _szego_core(alias_spectrum(S_b, stride).values)
        vals[b] = eps_b * bias_correction
    alpha = 1.0 - confidence
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return lo, hi


def floor(
    S: Spectrum,
    stride: int = 1,
    confidence: float = 0.95,
    n_boot: int = 200,
    seed: int = 0,
) -> FloorResult:
    """The floor at one stride, as a :class:`FloorResult`.

    The point estimate is bias-corrected for the log-chi-square bias of a Welch
    estimate (``E[log S_hat] = log S + psi(K) - log K`` with ``K`` segments).  The
    interval is a percentile bootstrap over the segments, which is honest about
    the correlation between neighbouring frequency bins; pass ``n_boot=0`` to skip
    it and get a degenerate interval.
    """
    if not isinstance(S, Spectrum):
        raise TypeError(f"floor expects a Spectrum, got {type(S).__name__}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    A = alias_spectrum(S, stride)
    eps2, messages = _szego_core(A.values)

    K = int(S.n_segments)
    if S.exact:
        # An analytic density has no sampling bias and no sampling variance.
        ci_low = ci_high = eps2
        ess = float("inf")
    else:
        bias = _digamma_int(K) - np.log(K)      # E[log S_hat] - log S, negative
        correction = float(np.exp(-bias))
        eps2 = eps2 * correction
        ess = 2.0 * K
        if S.segments is not None and n_boot > 0:
            rng = np.random.default_rng(seed)
            ci_low, ci_high = _bootstrap_ci(
                S, stride, confidence, n_boot, correction, rng
            )
        else:
            # Fall back to the (optimistic) analytic variance.
            M = _independent_bins(S)
            se = float(np.sqrt(_trigamma_int(K) / M))
            z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
            ci_low = float(np.exp(np.log(eps2) - z * se))
            ci_high = float(np.exp(np.log(eps2) + z * se))

    if not S.exact and K < _MIN_SEGMENTS_WARN:
        messages.append(
            f"only {K} segment(s): the floor is biased and the interval is wide. "
            f"Increase the signal length or reduce nperseg."
        )
    if not S.exact and eps2 > 1.0 + 1e-9:
        messages.append(
            f"floor {eps2:.4f} exceeds 1: estimator noise on a near-white "
            f"spectrum; the true floor is <= 1."
        )

    return FloorResult(
        eps2=eps2,
        stride=int(stride),
        estimator=S.estimator,
        n_segments=K,
        ess=ess,
        ci_low=ci_low,
        ci_high=ci_high,
        warnings=tuple(messages),
    )


def floor_curve(S: Spectrum, strides) -> dict[int, FloorResult]:
    """Floor for each stride in ``strides``, as :class:`FloorResult` objects."""
    return {int(s): floor(S, int(s)) for s in strides}


# ---------------------------------------------------------------------------
# analytic closed forms (the oracle, AR(1))
# ---------------------------------------------------------------------------
def ar1_floor(a: float, s: int) -> float:
    """Closed-form AR(1) floor after stride s:  1 - a^(2s)."""
    if not -1.0 < a < 1.0:
        raise ValueError("|a| must be < 1 for stationarity")
    if s < 1:
        raise ValueError("stride must be >= 1")
    return float(1.0 - a ** (2 * s))


def ar1_design_law(a: float, strides) -> dict[int, float]:
    """Closed-form relative inflation vs stride 1:  (1 - a^(2s)) / (1 - a^2)."""
    if not -1.0 < a < 1.0:
        raise ValueError("|a| must be < 1")
    base = 1.0 - a * a
    return {int(s): ar1_floor(a, int(s)) / base for s in strides}


# ---------------------------------------------------------------------------
# the decision layer, on a measured spectrum
# ---------------------------------------------------------------------------
def design_law(S: Spectrum, strides) -> dict[int, float]:
    """Relative error inflation vs stride 1, from a *measured* spectrum.

    ``inflation(s) = floor(S, s).eps2 / floor(S, 1).eps2``.  This is the
    spectrum-general form of ``ar1_design_law``: it needs no AR(1) assumption, so
    the decision layer exists for a measured spectrum.

    Only the point estimate is needed, so the bootstrap is skipped (``n_boot=0``);
    the decision is a ratio of point estimates and the interval plays no part.
    """
    if not isinstance(S, Spectrum):
        raise TypeError(f"design_law expects a Spectrum, got {type(S).__name__}")
    base = floor(S, 1, n_boot=0).eps2
    if base <= 0.0:
        raise ValueError(
            "floor at stride 1 is 0 (a deterministic or degenerate spectrum); "
            "relative inflation is undefined"
        )
    return {int(s): floor(S, int(s), n_boot=0).eps2 / base for s in strides}


def optimal_stride(
    S: Spectrum,
    max_inflation: float = 2.0,
    max_stride: int = 1024,
) -> int:
    """Largest stride whose error inflation stays within ``max_inflation``.

    The design rule: pick the cheapest (largest) stride you can afford such that
    ``inflation(s) <= max_inflation``.  ``max_inflation`` must exceed 1; at
    exactly 1 the answer is trivially stride 1, which is why the old default of
    1.0 was useless.

    The scan compares point estimates, so it skips the bootstrap (``n_boot=0``).
    At the default ``n_boot=200`` each ``floor`` call resamples the Welch segments
    200 times, so scanning to ``max_stride`` cost minutes rather than the ~1 s the
    point estimate needs -- and the point estimate is identical either way.
    """
    if not isinstance(S, Spectrum):
        raise TypeError(f"optimal_stride expects a Spectrum, got {type(S).__name__}")
    if max_inflation <= 1.0:
        raise ValueError("max_inflation must be > 1 (a budget of 1 leaves only stride 1)")
    if max_stride < 1:
        raise ValueError("max_stride must be >= 1")

    base = floor(S, 1, n_boot=0).eps2
    if base <= 0.0:
        raise ValueError("floor at stride 1 is 0; the budget is undefined")
    best = 1
    for s in range(2, max_stride + 1):
        if floor(S, s, n_boot=0).eps2 / base <= max_inflation:
            best = s
        else:
            break
    return best
