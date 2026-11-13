"""Tests for the spectral floor.  Run:  python3 -m pytest tests/ -q"""

import numpy as np
import pytest

from spectral_floor import (
    szego, alias_spectrum, floor_curve, ar1_floor, design_law, optimal_stride,
    ar1_psd, welch_psd, debias_welch, normalise_psd,
)


# --------------------------------------------------------------------------
# Kolmogorov-Szego
# --------------------------------------------------------------------------
def test_szego_white_noise_is_one():
    """A flat spectrum is maximally unpredictable: floor == 1."""
    S = np.ones(1000)
    assert szego(S) == pytest.approx(1.0)


def test_szego_ar1_matches_closed_form():
    """AR(1) floor must equal the textbook 1 - a^2."""
    u, _ = ar1_psd(0.9)
    for a in (0.7, 0.8, 0.9, 0.95, 0.99, 0.995):
        _, S = ar1_psd(a, u)
        assert szego(S) == pytest.approx(1 - a * a, rel=1e-9)


def test_szego_is_scale_invariant():
    """The floor is a ratio, so scaling S must not change it."""
    u, S = ar1_psd(0.9)
    assert szego(S) == pytest.approx(szego(S * 137.0), rel=1e-12)


def test_szego_rejects_negative_and_empty():
    with pytest.raises(ValueError):
        szego(np.array([1.0, -1.0]))
    with pytest.raises(ValueError):
        szego(np.array([]))


# --------------------------------------------------------------------------
# Aliasing
# --------------------------------------------------------------------------
def test_alias_stride_one_is_identity():
    _, S = ar1_psd(0.9)
    assert np.allclose(alias_spectrum(S, 1), S)


def test_alias_preserves_mean_power():
    """Decimation preserves total power, so the mean of the PSD is invariant."""
    _, S = ar1_psd(0.9)
    for s in (2, 4, 8):
        assert alias_spectrum(S, s).mean() == pytest.approx(S.mean(), rel=1e-9)


def test_alias_flattens_spectrum():
    """Aliasing mixes frequencies, so the spectrum must get flatter."""
    _, S = ar1_psd(0.99)
    def flatness(x): return x.max() / x.mean()
    assert flatness(alias_spectrum(S, 8)) < flatness(S)


# --------------------------------------------------------------------------
# The closed form: the headline result
# --------------------------------------------------------------------------
@pytest.mark.parametrize("a", [0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999])
@pytest.mark.parametrize("s", [1, 2, 3, 4, 8, 16, 32])
def test_closed_form_holds_exactly(a, s):
    """eps2(s) == 1 - a^(2s), verified via the folded spectrum."""
    u, S = ar1_psd(a)
    assert szego(alias_spectrum(S, s)) == pytest.approx(1 - a ** (2 * s), abs=1e-6)


def test_ar1_floor_matches_numeric():
    for a in (0.9, 0.99):
        for s in (1, 2, 4, 8):
            _, S = ar1_psd(a)
            assert ar1_floor(a, s) == pytest.approx(szego(alias_spectrum(S, s)), abs=1e-6)


def test_floor_is_monotone_in_stride():
    """More stride can never help: the floor must be non-decreasing."""
    _, S = ar1_psd(0.99)
    fc = floor_curve(S, [1, 2, 4, 8, 16])
    vals = [fc[s] for s in (1, 2, 4, 8, 16)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------
# Design law
# --------------------------------------------------------------------------
def test_design_law_baseline_is_one():
    assert design_law(0.99, [1])[1] == pytest.approx(1.0)


def test_design_law_known_value():
    """a=0.99, s=16 -> ~13.82x inflation (the number in the write-up)."""
    assert design_law(0.99, [16])[16] == pytest.approx(13.82, abs=0.05)


def test_optimal_stride_respects_budget():
    s = optimal_stride(0.99, max_inflation=2.0)
    assert design_law(0.99, [s])[s] <= 2.0
    assert design_law(0.99, [s + 1])[s + 1] > 2.0


def test_optimal_stride_rejects_bad_budget():
    with pytest.raises(ValueError):
        optimal_stride(0.99, max_inflation=0.5)


# --------------------------------------------------------------------------
# PSD estimation
# --------------------------------------------------------------------------
def test_welch_recovers_ar1_shape():
    """Welch on a simulated AR(1) must correlate with the analytic PSD."""
    rng = np.random.default_rng(0)
    n = 200_000
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + e[i]
    f, S_hat = welch_psd(x, nperseg=512)
    u = f / 1.0  # cycles/sample
    _, S_true = ar1_psd(0.9, u)
    # compare in log space, correlation
    a, b = np.log(S_hat[1:]), np.log(S_true[1:])
    r = np.corrcoef(a, b)[0, 1]
    assert r > 0.9, f"welch/analytic log-spectrum correlation too low: {r:.3f}"


def test_welch_rejects_short_input():
    with pytest.raises(ValueError):
        welch_psd(np.zeros(10), nperseg=64)


def test_welch_rejects_bad_window():
    with pytest.raises(ValueError):
        welch_psd(np.zeros(500), nperseg=64, window="triangle")


def test_normalise_psd():
    assert normalise_psd(np.array([2.0, 4.0])).mean() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalise_psd(np.zeros(4))


def test_debias_runs_and_stays_positive():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50_000)
    f, S = debias_welch(x, nperseg=256)
    assert np.all(S > 0)
    assert len(f) == len(S)


# --------------------------------------------------------------------------
# Estimator bias - a real methodological finding, locked in as a regression test
# --------------------------------------------------------------------------
def test_welch_floor_is_biased_at_large_stride():
    """The floor from a Welch estimate is biased, and the bias grows with stride.

    At s=16 the estimate reads ~0.153 against a true 0.275. Use the analytic
    spectrum when you have one; this pins the size of the effect.
    """
    from experiments.run_floor_experiment import simulate_ar1, ar1_true_spectrum

    x = simulate_ar1(0.99, 400_000, seed=0)
    _, S_hat = welch_psd(x, nperseg=512)
    S_hat = normalise_psd(S_hat)
    grid = np.linspace(0, 0.5, len(S_hat))
    u = np.linspace(0, 1, 20001)[:-1]
    S = normalise_psd(np.interp(u, grid, S_hat))

    S_true = ar1_true_spectrum(0.99)
    floor_est = szego(alias_spectrum(S, 16))
    floor_true = szego(alias_spectrum(S_true, 16))
    assert floor_est < floor_true - 0.1, "expected a large downward bias at s=16"


def test_analytic_floor_binds_on_ar1():
    """With the true spectrum the floor is tight: achieved error within ~15%."""
    from experiments.run_floor_experiment import simulate_ar1, ar1_true_spectrum, linear_predictor_rel_mse

    x = simulate_ar1(0.99, 400_000, seed=0)
    S_true = ar1_true_spectrum(0.99)
    for s in (1, 2, 4, 8, 16):
        floor = szego(alias_spectrum(S_true, s))
        achieved = linear_predictor_rel_mse(x, s, window=256)
        assert achieved / floor < 1.2, f"floor too loose at s={s}: {achieved/floor:.3f}x"
