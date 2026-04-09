from __future__ import annotations

from copy import deepcopy
from typing import Optional, Sequence, Any
import numpy as np

from .types import FDSResult
from .validate import ValidationError, assert_fds_compatible


def _copy_meta(fds: FDSResult) -> dict[str, Any]:
    return deepcopy(fds.meta or {})


def _copy_provenance(fds: FDSResult) -> dict[str, Any]:
    return deepcopy((fds.meta or {}).get("provenance", {}))


def scale_fds(fds: FDSResult, factor: float) -> FDSResult:
    if not np.isfinite(factor) or factor <= 0:
        raise ValidationError("scale factor must be finite and > 0.")
    dmg = np.asarray(fds.damage, dtype=float) * float(factor)
    meta = _copy_meta(fds)
    meta["provenance"] = {
        "source": "scale_fds",
        "factor": float(factor),
        "input": _copy_provenance(fds),
    }
    return FDSResult(f=np.asarray(fds.f, dtype=float), damage=dmg, meta=meta)


def sum_fds(fds_list: Sequence[FDSResult], weights: Optional[Sequence[float]] = None) -> FDSResult:
    if len(fds_list) == 0:
        raise ValidationError("fds_list must not be empty.")
    if weights is None:
        w = np.ones(len(fds_list), dtype=float)
    else:
        if len(weights) != len(fds_list):
            raise ValidationError("weights length must match fds_list length.")
        w = np.asarray(weights, dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValidationError("weights must be finite and >= 0.")
        if np.all(w == 0):
            raise ValidationError("at least one weight must be > 0.")

    ref = fds_list[0]
    for other in fds_list[1:]:
        assert_fds_compatible(ref, other)

    damage = np.zeros_like(np.asarray(ref.damage, dtype=float))
    for wi, fds in zip(w, fds_list):
        if wi == 0:
            continue
        dmg = np.asarray(fds.damage, dtype=float)
        if dmg.shape != damage.shape:
            raise ValidationError("All FDS damage arrays must match the reference shape.")
        damage += float(wi) * dmg

    meta = _copy_meta(ref)
    meta["provenance"] = {
        "source": "sum_fds",
        "n_inputs": int(len(fds_list)),
        "n_nonzero": int(np.count_nonzero(w)),
        "weights": [float(x) for x in w],
        "inputs": [
            {
                "index": int(i),
                "weight": float(wi),
                "provenance": _copy_provenance(fds),
            }
            for i, (wi, fds) in enumerate(zip(w, fds_list))
        ],
    }
    return FDSResult(f=np.asarray(ref.f, dtype=float), damage=damage, meta=meta)
