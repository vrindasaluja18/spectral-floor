"""Kolmogorov-Szego spectral floor for temporally strided prediction.

The central object is the one-step-ahead relative MSE floor

        eps2 = exp( <log S> ) / <S>

evaluated on the *aliased* spectrum of a stride-s decimated process.  This is the
best any linear predictor can do, so it lower-bounds the error of any world model
that observes at rate R/s.
"""

from .psd import welch_psd, debias_welch, ar1_psd, normalise_psd
from .floor import (
    szego,
    alias_spectrum,
    floor_curve,
    ar1_floor,
    design_law,
    optimal_stride,
)

__all__ = [
    "welch_psd",
    "debias_welch",
    "ar1_psd",
    "normalise_psd",
    "szego",
    "alias_spectrum",
    "floor_curve",
    "ar1_floor",
    "design_law",
    "optimal_stride",
]
__version__ = "0.1.0"
