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
    """Compute a pointwise envelope across compatible ERS results."""
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
