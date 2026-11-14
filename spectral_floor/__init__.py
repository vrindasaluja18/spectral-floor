"""Kolmogorov-Szego spectral floor for temporally strided prediction.

The central object is the one-step-ahead relative MSE floor

        eps2 = exp( <log S> ) / <S>

evaluated on the *aliased* spectrum of a stride-s decimated process.  This is the
best any linear predictor can do, so it lower-bounds the error of any world model
that observes at rate R/s.

Spectra are passed as :class:`Spectrum` objects, never as bare arrays.  A
``Spectrum`` knows its grid, sidedness and normalisation, so a one-sided estimate
cannot be silently folded as if it were two-sided.  ``Spectrum.from_welch`` is the
blessed entry point and always returns the two-sided form the folding step needs.
"""

from .spectrum import (
    Spectrum,
    ONE_SIDED,
    TWO_SIDED,
    UNIT_MEAN,
    RAW,
)
from .psd import welch_psd, ar1_psd, normalise_psd, two_sided_psd
from .floor import (
    FloorResult,
    szego,
    alias_spectrum,
    floor,
    floor_curve,
    ar1_floor,
    ar1_design_law,
    design_law,
    optimal_stride,
)

__all__ = [
    # types
    "Spectrum",
    "FloorResult",
    "ONE_SIDED",
    "TWO_SIDED",
    "UNIT_MEAN",
    "RAW",
    # estimation
    "welch_psd",
    "ar1_psd",
    "normalise_psd",
    "two_sided_psd",
    # the floor
    "szego",
    "alias_spectrum",
    "floor",
    "floor_curve",
    "ar1_floor",
    "ar1_design_law",
    # the decision layer
    "design_law",
    "optimal_stride",
]
__version__ = "0.2.0"
