"""Byte-rate arithmetic for a strided observation stream.

The rest of the package works in sampling-rate ratios (``s``).  This module is the
one place that turns a ratio into *bytes per second*, which needs a state width, an
element size and a native sample rate that the floor itself does not model.

It is deliberately only arithmetic.  It does not know whether a stride is
admissible for a given signal; that is the floor's job.  Pair the two:
``optimal_stride`` picks ``s`` under an accuracy budget, then this function prices it.
"""

from __future__ import annotations

import math
import numbers


def _require_int(name: str, value, minimum: int = 1) -> int:
    """Return ``value`` as an ``int`` or raise.

    Rejects bool (``True`` is not a stride of 1), non-numbers, nan/inf, and
    non-integral floats.  An integral float such as ``4.0`` is accepted, so
    numpy scalars and values that came through JSON or a config file work.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{name} must be a finite integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return int(value)


def state_bytes_per_second(
    n_state: int,
    dtype_bytes: int,
    sample_rate: float,
    stride: int,
) -> float:
    """Bytes per second needed to carry ``n_state`` values at the strided rate.

        bytes/s = n_state * dtype_bytes * (sample_rate / stride)

    ``n_state`` and ``dtype_bytes`` cancel in the ratio between two strides, so
    stride ``s`` moves exactly ``s`` times fewer bytes per second *for a fixed
    state representation*.  That is the precise form of the README's "16x".

    Scope
    -----
    * Observation-stream cost only.  Model-internal bytes (latents, KV cache)
      are not addressed.
    * Assumes plain subsampling (keep every ``stride``-th sample, no anti-alias
      filter).  That is the regime the floor analyses, and it is also why the
      saving is not free: see ``optimal_stride`` for the accuracy side.
    * Byte-aligned dtypes only.  Sub-byte types (int4, bit-packed) are not
      expressible through ``dtype_bytes``.

    Parameters
    ----------
    n_state : int
        Elements in one state/observation vector.  Must be >= 1.
    dtype_bytes : int
        Bytes per element (4 for float32, 8 for float64).  Must be >= 1.
    sample_rate : float
        Native sampling rate in Hz, before subsampling.  Finite and > 0.
    stride : int
        Temporal subsampling stride.  Must be a positive integer.

    Returns
    -------
    float
        Bytes per second of the strided observation stream.

    Raises
    ------
    TypeError
        If an argument is not a real number (or is a bool).
    ValueError
        If an integer argument is non-integral, non-finite or below its
        minimum, or ``sample_rate`` is non-finite or <= 0.
    """
    n_state = _require_int("n_state", n_state)
    dtype_bytes = _require_int("dtype_bytes", dtype_bytes)
    stride = _require_int("stride", stride)

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, numbers.Real):
        raise TypeError(
            f"sample_rate must be a real number, got {type(sample_rate).__name__}"
        )
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be finite and > 0, got {sample_rate!r}")

    # Integer product is exact; one float division at the end.
    return (n_state * dtype_bytes * float(sample_rate)) / stride
