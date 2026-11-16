"""Tests for the byte-rate arithmetic.  Run:  python3 -m pytest tests/test_budget.py -q

Oracles, not shapes:

  * exact value:      a hand-computed case with no rounding
  * ratio property:   stride s moves exactly s times fewer bytes (the README's 16x)
  * cancellation:     n_state and dtype_bytes drop out of the stride ratio
  * rejects garbage:  nan/inf/bool/non-integers raise instead of returning
                      a silent nan or a plausible-looking wrong number
"""

import math

import numpy as np
import pytest

from spectral_floor import state_bytes_per_second as f


class TestValue:
    def test_exact_hand_computed(self):
        # 1 float32 at 100 Hz, stride 4: 1 * 4 * 100 / 4 = 100 B/s
        assert f(1, 4, 100.0, 4) == 100.0

    def test_atari_like_frame_stream(self):
        # 84x84 uint8 frame at 60 Hz, stride 1: 7056 * 1 * 60
        assert f(84 * 84, 1, 60.0, 1) == 84 * 84 * 60.0

    def test_returns_float(self):
        assert isinstance(f(1, 4, 100.0, 4), float)

    @pytest.mark.parametrize("stride", [1, 2, 4, 8, 16])
    def test_ratio_is_exactly_stride(self, stride):
        """The README's '16x' is exactly this ratio, for any fixed representation."""
        base = f(7, 4, 250.0, 1)
        assert base / f(7, 4, 250.0, stride) == pytest.approx(stride, rel=1e-12)

    def test_width_and_dtype_cancel_in_stride_ratio(self):
        r_small = f(1, 1, 50.0, 1) / f(1, 1, 50.0, 16)
        r_big = f(4096, 8, 50.0, 1) / f(4096, 8, 50.0, 16)
        assert r_small == pytest.approx(r_big, rel=1e-12)

    def test_linear_in_each_factor(self):
        b = f(3, 4, 10.0, 2)
        assert f(6, 4, 10.0, 2) == pytest.approx(2 * b)
        assert f(3, 8, 10.0, 2) == pytest.approx(2 * b)
        assert f(3, 4, 20.0, 2) == pytest.approx(2 * b)


class TestAccepts:
    def test_numpy_integers(self):
        assert f(np.int64(1), np.int32(4), 100.0, np.int64(4)) == 100.0

    def test_integral_float_stride_is_accepted(self):
        # Pinned deliberately: configs/JSON often hand back 4.0 for 4.
        assert f(1, 4, 100.0, 4.0) == f(1, 4, 100.0, 4)

    def test_integer_sample_rate(self):
        assert f(1, 4, 100, 4) == 100.0

    def test_numpy_float_sample_rate(self):
        assert f(1, 4, np.float64(100.0), 4) == 100.0


class TestRejects:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0, -1.0, 0.0])
    def test_bad_sample_rate_value(self, bad):
        with pytest.raises(ValueError, match="sample_rate"):
            f(1, 4, bad, 4)

    @pytest.mark.parametrize("name,args", [
        ("stride", (1, 4, 100.0, 0)),
        ("stride", (1, 4, 100.0, -2)),
        ("stride", (1, 4, 100.0, 1.5)),
        ("stride", (1, 4, 100.0, float("nan"))),
        ("stride", (1, 4, 100.0, float("inf"))),
        ("n_state", (0, 4, 100.0, 4)),
        ("n_state", (2.5, 4, 100.0, 4)),
        ("n_state", (float("nan"), 4, 100.0, 4)),
        ("n_state", (float("inf"), 4, 100.0, 4)),
        ("dtype_bytes", (1, 0, 100.0, 4)),
        ("dtype_bytes", (1, 1.5, 100.0, 4)),
        ("dtype_bytes", (1, 0.5, 100.0, 4)),
    ])
    def test_bad_integer_argument_names_the_argument(self, name, args):
        # nan/inf must raise OUR ValueError (naming the argument), not a bare
        # "cannot convert float NaN to integer" or an OverflowError from int().
        with pytest.raises(ValueError, match=name):
            f(*args)

    @pytest.mark.parametrize("args", [
        (1, 4, 100.0, True),        # bool is not a stride of 1
        (True, 4, 100.0, 4),
        (1, True, 100.0, 4),
        (1, 4, True, 4),
        (1, 4, 100.0, "4"),
        (1, 4, "100", 4),
        (None, 4, 100.0, 4),
        (1, 4, 100.0, None),
    ])
    def test_wrong_type_raises_typeerror(self, args):
        with pytest.raises(TypeError):
            f(*args)

    def test_never_returns_nonfinite(self):
        """The failure mode this guards: a silent nan/inf flowing into a budget."""
        for args in [(1, 4, float("nan"), 4), (1, 4, float("inf"), 4),
                     (float("nan"), 4, 1.0, 1), (float("inf"), 4, 1.0, 1)]:
            with pytest.raises((ValueError, TypeError)):
                out = f(*args)
                assert math.isfinite(out)
