"""Tests for the spectral floor.  Run:  python3 -m pytest tests/ -q

The old suite checked shapes and positivity, which passes on garbage.  These tests
check *correctness against oracles*:

  * white noise:      floor == 1 exactly
  * AR(1):            floor == 1 - a^(2s) (closed form, to 1e-4)
  * a fitted predictor on simulated data: achieved error >= floor, and within a
    stated factor of it (the floor must bind, not just hold)
  * aliasing:         mean power preserved, and the fold is periodic (no clamp)
  * log-integrability: zero bins and extreme dynamic range raise warnings
  * the estimator:    an interval that covers the analytic value across seeds
  * parse, don't validate: a bare ndarray is rejected, not silently folded
"""

import warnings

import numpy as np
import pytest

from spectral_floor import (
    Spectrum, FloorResult,
    szego, alias_spectrum, floor, floor_curve,
    ar1_floor, ar1_design_law, design_law, optimal_stride,
    ar1_psd, welch_psd, normalise_psd, two_sided_psd,
)
from spectral_floor import ONE_SIDED, TWO_SIDED, UNIT_MEAN

from experiments.run_floor_experiment import (
    simulate_ar1, linear_predictor_rel_mse, ar1_true_spectrum,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def flat_spectrum(n=20001, value=1.0, **kw):
    u = np.linspace(0.0, 1.0, n)[:-1]
    return Spectrum.from_psd(np.full_like(u, value), u, exact=True, **kw)


def one_sided_spectrum(n=129):
    f = np.linspace(0.0, 0.5, n)
    S = 1.0 / (1.0 + (f - 0.1) ** 2)
    S = S / S.mean()
    return Spectrum.from_psd(S, f, sidedness=ONE_SIDED, normalisation=UNIT_MEAN)


# ==========================================================================
# The Spectrum type: parse, don't validate
# ==========================================================================
class TestSpectrumType:
    def test_rejects_bare_array(self):
        for fn in (szego, lambda S: alias_spectrum(S, 2), floor, design_law):
            with pytest.raises(TypeError):
                fn(np.ones(100))

    def test_rejects_negative_and_nonfinite(self):
        u = np.linspace(0, 1, 11)[:-1]
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.array([1.0] * 5 + [-1.0] + [1.0] * 4), u)
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.array([1.0] * 5 + [np.nan] + [1.0] * 4), u)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.ones(10), np.linspace(0, 1, 9))

    def test_rejects_non_uniform_grid(self):
        g = np.concatenate([np.linspace(0, 0.5, 5), np.linspace(0.6, 0.99, 5)])
        with pytest.raises(ValueError, match="uniform"):
            Spectrum.from_psd(np.ones(10), g)

    def test_rejects_bad_sidedness_and_normalisation(self):
        u = np.linspace(0, 1, 11)[:-1]
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.ones(10), u, sidedness="both")
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.ones(10), u, normalisation="bogus")

    def test_two_sided_grid_must_stay_below_one(self):
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.ones(3), np.array([0.0, 0.5, 1.0]))

    def test_is_frozen_and_arrays_are_read_only(self):
        S = flat_spectrum()
        with pytest.raises(Exception):
            S.values = np.zeros_like(S.values)
        with pytest.raises(ValueError):
            S.values[0] = 99.0

    def test_unit_mean_normalisation_is_enforced(self):
        u = np.linspace(0, 1, 11)[:-1]
        with pytest.raises(ValueError, match="unit-mean"):
            Spectrum.from_psd(np.full(10, 3.0), u, normalisation=UNIT_MEAN)

    def test_from_welch_always_returns_two_sided(self):
        x = simulate_ar1(0.9, 200_000, seed=0)
        S = Spectrum.from_welch(x, nperseg=512)
        assert S.sidedness == TWO_SIDED
        assert S.grid[0] == 0.0 and S.grid[-1] < 1.0
        assert S.n_segments > 0 and S.nperseg == 512


# ==========================================================================
# The floor formula, against exact oracles
# ==========================================================================
class TestFloorOracle:
    def test_white_noise_is_exactly_one(self):
        """A flat density is maximally unpredictable: floor == 1 exactly."""
        S = flat_spectrum()
        assert szego(S) == pytest.approx(1.0, rel=1e-12)
        assert floor(S, 1).eps2 == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("a", [0.7, 0.8, 0.9, 0.95, 0.99, 0.995])
    def test_ar1_floor_matches_closed_form(self, a):
        """The floor of the analytic AR(1) density equals 1 - a^2."""
        S = Spectrum.from_ar1(a)
        assert szego(S) == pytest.approx(1 - a * a, rel=1e-4)

    def test_szego_is_scale_invariant(self):
        S = Spectrum.from_ar1(0.9, n=10001)
        u = np.linspace(0, 1, S.values.size + 1)[:-1]
        scaled = Spectrum.from_psd(S.values * 137.0, u, normalisation="raw")
        assert szego(scaled) == pytest.approx(szego(S), rel=1e-12)

    def test_rejects_degenerate_spectra(self):
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.zeros(4), np.linspace(0, 1, 5)[:-1],
                              normalisation="raw")
        with pytest.raises(ValueError):
            Spectrum.from_psd(np.ones(1), np.array([0.0]))


# ==========================================================================
# Aliasing: the fold, verified structurally
# ==========================================================================
class TestAliasing:
    def test_stride_one_is_identity(self):
        S = Spectrum.from_ar1(0.9)
        A = alias_spectrum(S, 1)
        assert np.allclose(A.values, S.values)

    @pytest.mark.parametrize("s", [2, 4, 8, 16])
    def test_preserves_mean_power(self, s):
        """Decimation preserves total power: the density's mean is invariant."""
        S = Spectrum.from_ar1(0.9)
        A = alias_spectrum(S, s)
        assert A.values.mean() == pytest.approx(S.values.mean(), rel=1e-9)

    def test_fold_is_periodic_not_clamped(self):
        """The shifts wrap around the circle; a clamped fold would leave a seam.

        The exact oracle: for ``S(u) = 2 + cos(2 pi u)`` the sum over the s-th
        roots of unity is exactly 2, so the fold is exactly 2 for every s > 1.
        A non-periodic interpolation clamps the last sliver and breaks this.
        """
        n = 20001
        u = np.linspace(0, 1, n + 1)[:-1]
        S = 2.0 + np.cos(2 * np.pi * u)
        Sp = Spectrum.from_psd(S, u, normalisation="raw", exact=True)
        for s in (2, 3, 4, 16):
            A = alias_spectrum(Sp, s).values
            assert np.allclose(A, 2.0, rtol=1e-6), (
                f"fold of 2+cos(2 pi u) at s={s} is {A.min():.4f}..{A.max():.4f}, "
                f"expected 2 exactly (a clamped fold breaks this)"
            )

    def test_folding_flattens_spectrum(self):
        S = Spectrum.from_ar1(0.99)
        def flatness(x):
            return x.max() / x.mean()
        assert flatness(alias_spectrum(S, 8).values) < flatness(S.values)

    def test_alias_takes_one_sided_spectrum_and_expands_it(self):
        """alias_spectrum expands a one-sided Spectrum itself (never trusts caller)."""
        one = one_sided_spectrum()
        A = alias_spectrum(one, 4)
        assert A.sidedness == TWO_SIDED
        assert A.values.mean() == pytest.approx(1.0)
        # the fold is even on the unit circle, as a real process requires
        n = A.values.size
        assert np.allclose(A.values, np.roll(A.values[::-1], 1), rtol=1e-6, atol=1e-9)


# ==========================================================================
# The closed form after stride, and the decision layer
# ==========================================================================
class TestDesignLaw:
    @pytest.mark.parametrize("a", [0.7, 0.9, 0.99])
    @pytest.mark.parametrize("s", [1, 2, 3, 4, 8, 16, 32])
    def test_closed_form_holds_via_folded_spectrum(self, a, s):
        S = Spectrum.from_ar1(a)
        assert floor(S, s).eps2 == pytest.approx(1 - a ** (2 * s), abs=1e-6)

    def test_ar1_design_law_baseline_is_one(self):
        assert ar1_design_law(0.99, [1])[1] == pytest.approx(1.0)

    def test_ar1_design_law_known_value(self):
        assert ar1_design_law(0.99, [16])[16] == pytest.approx(13.82, abs=0.05)

    def test_design_law_on_measured_spectrum_needs_no_ar1_assumption(self):
        """The spectrum-general design law exists for a measured spectrum."""
        S = Spectrum.from_ar1(0.99)
        law = design_law(S, [1, 2, 4, 8, 16])
        assert law[1] == pytest.approx(1.0)
        # and it agrees with the AR(1) closed form here, because S *is* AR(1)
        for s in (2, 4, 8, 16):
            assert law[s] == pytest.approx(ar1_design_law(0.99, [s])[s], rel=1e-3)

    def test_optimal_stride_respects_budget(self):
        S = Spectrum.from_ar1(0.99)
        s = optimal_stride(S, max_inflation=2.0)
        law = design_law(S, [s, s + 1])
        assert law[s] <= 2.0
        assert law[s + 1] > 2.0

    def test_optimal_stride_default_is_not_degenerate(self):
        """Regression: the old default max_inflation=1.0 always returned 1."""
        S = Spectrum.from_ar1(0.99)
        assert optimal_stride(S) > 1

    def test_optimal_stride_rejects_bad_budget(self):
        S = Spectrum.from_ar1(0.99)
        with pytest.raises(ValueError):
            optimal_stride(S, max_inflation=1.0)
        with pytest.raises(ValueError):
            optimal_stride(S, max_inflation=0.5)

    def test_floor_is_monotone_in_stride(self):
        S = Spectrum.from_ar1(0.99)
        fc = floor_curve(S, [1, 2, 4, 8, 16])
        vals = [fc[s].eps2 for s in (1, 2, 4, 8, 16)]
        assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


# ==========================================================================
# The result object
# ==========================================================================
class TestFloorResult:
    def test_returns_floor_result_with_provenance(self):
        x = simulate_ar1(0.99, 200_000, seed=0)
        S = Spectrum.from_welch(x, nperseg=512)
        r = floor(S, 4)
        assert isinstance(r, FloorResult)
        assert r.stride == 4
        assert r.estimator == "welch"
        assert r.n_segments == S.n_segments > 1
        assert r.ess == pytest.approx(2 * r.n_segments)
        assert r.ci_low < r.eps2 < r.ci_high

    def test_interval_covers_the_analytic_value(self):
        """Across seeds the Welch estimate's interval must contain 1 - a^(2s)."""
        for s in (1, 4, 16):
            true = ar1_floor(0.99, s)
            covered = 0
            for seed in range(8):
                x = simulate_ar1(0.99, 400_000, seed=seed)
                S = Spectrum.from_welch(x, nperseg=1024)
                r = floor(S, s)
                covered += r.ci_low <= true <= r.ci_high
            assert covered >= 7, f"interval missed the true floor too often at s={s}"

    def test_exact_spectrum_has_degenerate_interval(self):
        S = Spectrum.from_ar1(0.99)
        r = floor(S, 4)
        assert r.ci_low == r.ci_high == r.eps2
        assert r.ess == float("inf")

    def test_floor_curve_maps_strides_to_results(self):
        S = Spectrum.from_ar1(0.9)
        fc = floor_curve(S, [1, 2])
        assert set(fc) == {1, 2}
        assert all(isinstance(v, FloorResult) for v in fc.values())

    def test_bad_confidence_rejected(self):
        S = Spectrum.from_ar1(0.9)
        with pytest.raises(ValueError):
            floor(S, 1, confidence=1.5)


# ==========================================================================
# Silent numerics: log-integrability and dynamic range
# ==========================================================================
class TestSilentNumerics:
    def test_zero_bin_warns_and_is_grid_dependent(self):
        """A zero-power bin makes the discrete floor a grid artefact. Warn."""
        for n in (50, 500, 5000):
            u = np.linspace(0, 1, n + 1)[:-1]
            S = np.ones(n)
            S[n // 2] = 0.0
            Sp = Spectrum.from_psd(S, u, normalisation="raw")
            with pytest.warns(UserWarning, match="zero-power"):
                val = szego(Sp)
            assert val == 0.0

    def test_is_log_integrable_flag(self):
        S = flat_spectrum()
        assert S.is_log_integrable()
        assert S.n_zero_bins() == 0
        u = np.linspace(0, 1, 11)[:-1]
        Sz = Spectrum.from_psd(np.array([1.0] * 5 + [0.0] + [1.0] * 4), u,
                               normalisation="raw")
        assert not Sz.is_log_integrable()
        assert Sz.n_zero_bins() == 1

    def test_extreme_dynamic_range_warns(self):
        n = 4000
        u = np.linspace(0, 1, n + 1)[:-1]
        S = np.ones(n)
        S[0] = 1e-14                       # 140 dB trough
        Sp = Spectrum.from_psd(S / S.mean(), u, normalisation=UNIT_MEAN)
        with pytest.warns(UserWarning, match="dynamic range"):
            szego(Sp)

    def test_dynamic_range_db_is_reported(self):
        S = flat_spectrum()
        assert S.dynamic_range_db() == pytest.approx(0.0, abs=1e-9)


# ==========================================================================
# PSD estimation
# ==========================================================================
class TestPsdEstimation:
    def test_welch_recovers_ar1_shape(self):
        x = simulate_ar1(0.9, 200_000, seed=0)
        f, S_hat = welch_psd(x, nperseg=512)
        _, S_true = ar1_psd(0.9, f)
        r = np.corrcoef(np.log(S_hat[1:]), np.log(S_true[1:]))[0, 1]
        assert r > 0.9, f"welch/analytic log-spectrum correlation too low: {r:.3f}"

    def test_welch_reports_segment_count(self):
        x = simulate_ar1(0.9, 100_000, seed=0)
        f, S, nseg = welch_psd(x, nperseg=512, return_n_segments=True)
        assert nseg == 1 + (len(x) - 512) // 256
        assert len(f) == len(S)

    def test_welch_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            welch_psd(np.zeros(10), nperseg=64)
        with pytest.raises(ValueError):
            welch_psd(np.zeros(500), nperseg=64, window="triangle")
        with pytest.raises(ValueError):
            welch_psd(np.zeros(500), nperseg=64, noverlap=64)

    def test_normalise_psd(self):
        assert normalise_psd(np.array([2.0, 4.0])).mean() == pytest.approx(1.0)
        with pytest.raises(ValueError):
            normalise_psd(np.zeros(4))

    def test_debias_welch_is_gone(self):
        """The pointwise-kernel 'debias' multiplied bins by ~1e6; it is deleted."""
        import spectral_floor
        assert not hasattr(spectral_floor, "debias_welch")
        assert not hasattr(spectral_floor.psd, "debias_welch")


# ==========================================================================
# The two-sided expansion (mirror, not interpolation)
# ==========================================================================
class TestTwoSidedExpansion:
    def test_is_even_and_normalised(self):
        f = np.linspace(0, 0.5, 257)
        S = 1.0 / (1.0 + (f - 0.1) ** 2)
        u, S_full = two_sided_psd(f, S)
        assert S_full.mean() == pytest.approx(1.0)
        n = len(u)
        assert np.allclose(S_full, np.roll(S_full[::-1], 1), rtol=1e-9, atol=1e-12)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            two_sided_psd(np.linspace(0, 0.5, 10), np.ones(9))

    def test_clamping_the_one_sided_psd_biases_the_floor_low(self):
        """Regression: interpolating a one-sided PSD onto [0,1) clamps, biasing low.

        The floor from a clamped expansion is well below the true floor, and the
        error grows with stride; the mirror expansion removes it.  This is the bug
        the experiment runner used to have, pinned in place.
        """
        S_true = ar1_true_spectrum(0.99)
        half = S_true.values[: len(S_true.values) // 2 + 1]
        grid = np.linspace(0, 0.5, len(half))
        u = np.linspace(0, 1, 20001)[:-1]
        S_clamped = normalise_psd(np.interp(u, grid, half))
        S_clamp = Spectrum.from_psd(S_clamped, u, exact=True)
        _, S_mirrored = two_sided_psd(grid, half)
        S_mirror = Spectrum.from_psd(S_mirrored, u, exact=True)

        for s in (4, 16):
            f_true = floor(S_true, s).eps2
            f_clamp = floor(S_clamp, s).eps2
            f_mirror = floor(S_mirror, s).eps2
            assert f_clamp < f_true - 0.02, f"expected low bias at s={s}"
            assert f_mirror == pytest.approx(f_true, rel=0.02)

        bias1 = floor(S_true, 1).eps2 - floor(S_clamp, 1).eps2
        bias16 = floor(S_true, 16).eps2 - floor(S_clamp, 16).eps2
        assert bias1 < 0.005 and bias16 > 0.1
        assert bias16 > 20 * bias1


# ==========================================================================
# The floor binds on real data (the headline claim)
# ==========================================================================
class TestFloorBinds:
    def test_analytic_floor_is_tight(self):
        """With the true spectrum the achieved error is within 20% of the floor."""
        x = simulate_ar1(0.99, 400_000, seed=0)
        S_true = ar1_true_spectrum(0.99)
        for s in (1, 2, 4, 8, 16):
            fl = floor(S_true, s).eps2
            achieved = linear_predictor_rel_mse(x, s, window=256)
            assert achieved / fl < 1.2, f"floor too loose at s={s}: {achieved/fl:.3f}x"

    def test_achieved_never_beats_the_floor_materially(self):
        """The floor is a lower bound: a predictor must not come in far below it."""
        x = simulate_ar1(0.99, 400_000, seed=1)
        S_true = ar1_true_spectrum(0.99)
        for s in (1, 4, 16):
            fl = floor(S_true, s).eps2
            achieved = linear_predictor_rel_mse(x, s, window=256)
            assert achieved > 0.8 * fl, f"beat the floor at s={s}: {achieved/fl:.3f}x"
