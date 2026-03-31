from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .types import FDSResult
from .validate import ValidationError, assert_fds_compatible


def scale_fds(fds: FDSResult, factor: float) -> FDSResult:
    if not np.isfinite(factor) or factor <= 0:
        raise ValidationError("scale factor must be finite and > 0.")
    dmg = np.asarray(fds.damage, dtype=float) * float(factor)
    meta = dict(fds.meta or {})
    prov = dict(meta.get("provenance", {}))
    prov.setdefault("ops", []).append({"op": "scale", "factor": float(factor)})
    meta["provenance"] = prov
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
        damage += float(wi) * np.asarray(fds.damage, dtype=float)

    meta = dict(ref.meta or {})
    prov = dict(meta.get("provenance", {}))
    prov.setdefault("ops", []).append({"op": "sum", "n": len(fds_list)})
    meta["provenance"] = prov
    return FDSResult(f=np.asarray(ref.f, dtype=float), damage=damage, meta=meta)
