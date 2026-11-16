"""Tests for the Atari experiment's pure logic (no ROM, no ale-py needed).

The rollout needs the emulator, but the pieces that were actually *wrong* on real
data are pure functions: the ball detector's handling of line-like pixels, the
motion signal, the hold-fill of serve gaps, and the degenerate-signal guard.  Those
are what these tests pin down.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.run_atari_floor import (
    BRIGHT,
    _guarded,
    detect_ball,
    detect_paddle,
    motion_signal,
    normalise,
    stationarity_report,
)


def blank_frame() -> np.ndarray:
    """A frame in the ALE palette: background 87, ball/border 236, paddles 147."""
    f = np.full((210, 160), 87, dtype=np.uint8)
    f[:SCORE_ROWS_GUARD, :] = 236          # score strip (a horizontal line)
    f[190:, :] = 236                       # bottom border (a horizontal line)
    return f


SCORE_ROWS_GUARD = 34


def put_ball(f: np.ndarray, y: int, x: int) -> np.ndarray:
    f[y:y + 4, x:x + 2] = 236              # a 4x2 ball, as ALE draws it
    return f


def test_detect_ball_finds_the_ball_not_the_lines():
    """The borders and score strip are horizontal lines; the ball must win."""
    f = blank_frame()
    put_ball(f, 100, 60)
    y = detect_ball(f)
    assert y is not None and 98 <= y <= 104


def test_detect_ball_ignores_the_centre_line():
    """A vertical centre line must not be mistaken for the ball."""
    f = blank_frame()
    f[34:190, 79:81] = 236                 # centre line
    put_ball(f, 70, 40)
    y = detect_ball(f)
    assert y is not None and 68 <= y <= 74


def test_detect_ball_returns_nan_without_a_ball():
    f = blank_frame()
    assert np.isnan(detect_ball(f))


def test_detect_paddle_uses_the_dimmer_level():
    """Paddles render at 147, dimmer than the ball at 236, in columns 16..19."""
    f = blank_frame()
    f[80:96, 16:20] = 147
    y = detect_paddle(f)
    assert y is not None and 80 <= y <= 96


def test_motion_signal_is_zero_on_a_static_screen():
    f = blank_frame()
    frames = np.stack([f, f, f])
    assert np.allclose(motion_signal(frames), 0.0)


def test_motion_signal_detects_change():
    a = blank_frame()
    b = put_ball(blank_frame(), 50, 50)
    frames = np.stack([a, b])
    m = motion_signal(frames)
    assert m[0] == 0.0 and m[1] > 0.0


def test_guarded_returns_nan_for_degenerate_signal():
    """A constant signal makes the relative MSE explode; report nan, not 1e30."""
    const = np.ones(500)
    fn = lambda x, s: float(np.mean(x ** 2) / (np.var(x) + 1e-30))
    assert np.isnan(_guarded(fn, const, 1))


def test_guarded_passes_through_a_finite_value():
    assert _guarded(lambda x, s: 0.25, np.arange(10.0), 1) == 0.25


def test_stationarity_report_is_small_for_a_centred_signal():
    """The old bug divided by the mean of a centred signal and returned ~1e11."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    r = stationarity_report(x)
    assert r["mean_spread_rel"] < 1.0, r


def test_normalise_handles_a_constant_signal():
    assert np.allclose(normalise(np.full(10, 3.0)), 0.0)
