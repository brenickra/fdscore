"""Algebraic post-processing for compatible extreme-response spectra.

This module provides safe combination utilities for
:class:`fdscore.types.ERSResult` objects that already share the same
engineering interpretation, oscillator grid, and compatibility
signature.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import numpy as np

from .types import ERSResult
from .validate import ValidationError, assert_ers_compatible


def _copy_meta(ers: ERSResult) -> dict[str, Any]:
    return deepcopy(ers.meta or {})


def _copy_provenance(ers: ERSResult) -> dict[str, Any]:
    return deepcopy((ers.meta or {}).get("provenance", {}))


def envelope_ers(results: Sequence[ERSResult]) -> ERSResult:
    """Compute a pointwise envelope across compatible ERS results.

    Parameters
    ----------
    results : sequence of ERSResult
        Response spectra to combine. All inputs must share the same
        frequency grid and ERS compatibility signature.

    Returns
    -------
    ERSResult
        Envelope spectrum whose response at each oscillator frequency is
        the maximum of the corresponding input values.

    Notes
    -----
    Compatibility is enforced internally, so the function will reject
    inputs that mix different response metrics, peak conventions,
    oscillator assumptions, or incompatible frequency grids.

    The returned result preserves the reference grid and stores
    provenance metadata recording the number and origin of contributing
    spectra.
    """
    if len(results) == 0:
        raise ValidationError("results must not be empty.")

    ref = results[0]
    for other in results[1:]:
        assert_ers_compatible(ref, other)

    response = np.asarray(ref.response, dtype=float).copy()
    for ers in results[1:]:
        response_i = np.asarray(ers.response, dtype=float)
        if response_i.shape != response.shape:
            raise ValidationError("All ERS response arrays must match the reference shape.")
        response = np.maximum(response, response_i)

    meta = _copy_meta(ref)
    meta["provenance"] = {
        "source": "envelope_ers",
        "n_inputs": int(len(results)),
        "inputs": [
            {
                "index": int(i),
                "provenance": _copy_provenance(ers),
            }
            for i, ers in enumerate(results)
        ],
    }
    return ERSResult(f=np.asarray(ref.f, dtype=float), response=response, meta=meta)
