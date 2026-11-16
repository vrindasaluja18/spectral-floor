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


def test_motion_signal_is_block_invariant():
    """The block size bounds the float32 temporary; it must not change the result."""
    rng = np.random.default_rng(0)
    frames = (rng.random((137, 210, 160)) * 255).astype(np.uint8)
    ref = motion_signal(frames)
    for block in (1, 7, 64, 10_000):
        assert np.allclose(motion_signal(frames, block=block), ref), block


def test_rollout_is_streaming_not_accumulating(monkeypatch):
    """rollout must not retain the frame stack: memory is O(1) in `steps`.

    A fake ALE records any array whose first axis equals `steps` (that is what an
    accumulating `np.empty((steps, 210, 160))` would be).  A streaming rollout only
    ever holds a single (210, 160) frame plus two length-`steps` 1-D vectors, so no
    *3-D* array of that size is ever created.  This fails on the pre-fix code.
    """
    import sys
    import types

    import numpy as np
    from experiments import run_atari_floor as m

    steps = 500
    stacks = []

    def make_spy(name):
        real = getattr(np, name)

        def spy(shape, *a, **k):
            if isinstance(shape, tuple) and len(shape) == 3 and shape[0] == steps:
                stacks.append((name, shape))
            return real(shape, *a, **k)

        return spy

    for name in ("empty", "zeros", "full", "ones"):
        monkeypatch.setattr(m.np, name, make_spy(name))

    class FakeALE:
        def setInt(self, *a): pass
        def setFloat(self, *a): pass
        def loadROM(self, *a): pass
        def getMinimalActionSet(self): return [0, 1, 2, 3]
        def reset_game(self): pass
        def act(self, *a): pass
        def game_over(self): return False
        def getScreenGrayscale(self):
            return np.full((210, 160), 87, dtype=np.uint8)

    fake_ale_py = types.SimpleNamespace(
        ALEInterface=FakeALE,
        roms=types.SimpleNamespace(get_rom_path=lambda g: g),
    )
    monkeypatch.setitem(sys.modules, "ale_py", fake_ale_py)

    motion, ball_y, episodes = m.rollout("pong", steps)

    assert stacks == [], f"rollout allocated a {steps}-frame stack: {stacks}"
    assert motion.shape == (steps,) and ball_y.shape == (steps,)


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
