"""Algebraic operations on compatible fatigue damage spectra.

This module provides lightweight combinators for post-processing
instances of :class:`fdscore.types.FDSResult` after they have been
computed by a solver such as :func:`fdscore.fds_time.compute_fds_time`
or :func:`fdscore.fds_spectral.compute_fds_spectral_psd`.

The operations implemented here do not recompute oscillator responses or
damage from an underlying excitation. Instead, they apply scalar
transformations or weighted superposition directly to already assembled
damage spectra, preserving a provenance trail in the returned metadata.
"""

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
    """Scale all damage ordinates of a fatigue damage spectrum.

    Parameters
    ----------
    fds : FDSResult
        Input fatigue damage spectrum to be scaled.
    factor : float
        Positive finite multiplicative factor applied uniformly to all
        damage values.

    Returns
    -------
    FDSResult
        New fatigue damage spectrum with the same frequency grid as the
        input and damage values multiplied by ``factor``.

    Notes
    -----
    This operation is purely algebraic. It is appropriate when a known
    post hoc scaling of damage is required, for example to account for an
    external calibration factor or to compare normalized spectra on a
    common basis. The function does not revisit the underlying stress
    cycles, SDOF response model, or S-N assumptions used to generate the
    original result.

    The returned object stores provenance metadata indicating that the
    spectrum was produced by :func:`scale_fds`, together with the applied
    factor and the upstream provenance chain of the input spectrum.
    """
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
    """Form a weighted sum of mutually compatible fatigue damage spectra.

    Parameters
    ----------
    fds_list : sequence of FDSResult
        Sequence of fatigue damage spectra defined on compatible
        frequency grids and generated under the same compatibility
        contract.
    weights : sequence of float or None, optional
        Non-negative weights applied to each spectrum before summation.
        When omitted, unit weights are used for all inputs.

    Returns
    -------
    FDSResult
        Weighted sum of the input fatigue damage spectra on the reference
        frequency grid.

    Notes
    -----
    Compatibility is enforced internally before summation so that the
    spectra can be meaningfully combined without mixing damage metrics,
    oscillator assumptions, or incompatible S-N definitions.
    This function therefore acts as a safe superposition utility for
    spectra that already satisfy a shared engineering interpretation.

    The summation is performed directly on the damage ordinates. No
    normalization is applied to the weights, so the caller controls
    whether the result represents a simple sum, a convex combination, or
    another weighted aggregate.

    Provenance metadata is preserved and extended to record the number of
    inputs, the applied weights, and the provenance chain of each
    contributing spectrum.
    """
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
