from __future__ import annotations

import numpy as np
from .types import SDOFParams
from .validate import validate_sdof, validate_frequency_vector


def build_frequency_grid(sdof: SDOFParams) -> np.ndarray:
    """Build or validate the oscillator frequency grid (Hz)."""
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
