"""Frequency-grid construction utilities for SDOF analyses.

This module materializes the oscillator frequency grid used throughout
the library. It accepts either an explicit frequency vector or an
implicit linear-grid specification stored in
:class:`fdscore.types.SDOFParams` and returns a validated NumPy array in
Hz.
"""

from __future__ import annotations

import numpy as np
from .types import SDOFParams
from .validate import validate_sdof, validate_frequency_vector


def build_frequency_grid(sdof: SDOFParams) -> np.ndarray:
    """Build the validated oscillator frequency grid in Hz.

    Parameters
    ----------
    sdof : SDOFParams
        Oscillator-grid definition. The grid may be provided explicitly
        through ``sdof.f`` or implicitly through the tuple
        ``(sdof.fmin, sdof.fmax, sdof.df)``.

    Returns
    -------
    numpy.ndarray
        Strictly increasing positive frequency vector in Hz.

    Notes
    -----
    The function first validates ``sdof`` through
    :func:`fdscore.validate.validate_sdof`. When an implicit linear grid
    is requested, the number of bins is computed from
    ``floor((fmax - fmin) / df + 1)`` and tiny floating-point overshoot
    beyond ``fmax`` is clipped before final validation.
    """
    validate_sdof(sdof)
    if sdof.f is not None:
        f = np.asarray(sdof.f, dtype=float).copy()
        validate_frequency_vector(f)
        return f

    fmin = float(sdof.fmin)
    fmax = float(sdof.fmax)
    df = float(sdof.df)

    n = int(np.floor((fmax - fmin) / df + 1.0 + 1e-12))
    f = fmin + df * np.arange(n, dtype=float)
    # clip tiny overshoot
    f = f[f <= fmax * (1.0 + 1e-12)]
    validate_frequency_vector(f)
    return f
