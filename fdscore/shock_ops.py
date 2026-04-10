"""Post-processing helpers for shock-response and PVSS results.

This module provides envelope operations for one-sided and two-sided
shock spectra. It builds on the generic ERS envelope logic while adding
checks for shock-specific spectrum kinds such as SRS and PVSS.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from .ers_ops import envelope_ers
from .types import ERSResult, ShockSpectrumPair
from .validate import ValidationError, parse_ers_compat


def _copy_pair_meta(pair: ShockSpectrumPair) -> dict:
    return deepcopy(pair.meta or {})


def _ensure_expected_kind(ers: ERSResult, *, expected_kind: str) -> None:
    compat = parse_ers_compat((ers.meta or {}).get("compat", {}))
    if compat.ers_kind != expected_kind:
        raise ValidationError(
            f"Expected ERS kind '{expected_kind}', got '{compat.ers_kind}'."
        )


def _envelope_shock_results(
    results: Sequence[ERSResult],
    *,
    expected_kind: str,
    source: str,
) -> ERSResult:
    if len(results) == 0:
        raise ValidationError("results must not be empty.")
    for ers in results:
        if not isinstance(ers, ERSResult):
            raise ValidationError("results must contain only ERSResult values.")
        _ensure_expected_kind(ers, expected_kind=expected_kind)

    out = envelope_ers(results)
    meta = deepcopy(out.meta or {})
    meta["provenance"] = {
        "source": source,
        "n_inputs": int(len(results)),
        "inputs": [
            {
                "index": int(i),
                "provenance": deepcopy((ers.meta or {}).get("provenance", {})),
            }
            for i, ers in enumerate(results)
        ],
    }
    return ERSResult(f=out.f, response=out.response, meta=meta)


def _envelope_shock_pairs(
    results: Sequence[ShockSpectrumPair],
    *,
    expected_kind: str,
    source: str,
) -> ShockSpectrumPair:
    if len(results) == 0:
        raise ValidationError("results must not be empty.")
    for pair in results:
        if not isinstance(pair, ShockSpectrumPair):
            raise ValidationError("results must contain only ShockSpectrumPair values.")
        _ensure_expected_kind(pair.neg, expected_kind=expected_kind)
        _ensure_expected_kind(pair.pos, expected_kind=expected_kind)

    neg = _envelope_shock_results(
        [pair.neg for pair in results],
        expected_kind=expected_kind,
        source=source,
    )
    pos = _envelope_shock_results(
        [pair.pos for pair in results],
        expected_kind=expected_kind,
        source=source,
    )
    meta = _copy_pair_meta(results[0])
    meta["provenance"] = {
        "source": source,
        "n_inputs": int(len(results)),
        "inputs": [
            {
                "index": int(i),
                "provenance": deepcopy((pair.meta or {}).get("provenance", {})),
            }
            for i, pair in enumerate(results)
        ],
    }
    meta["peak_mode"] = "both"
    meta["ers_kind"] = expected_kind
    return ShockSpectrumPair(neg=neg, pos=pos, meta=meta)


def envelope_srs(results: Sequence[ERSResult | ShockSpectrumPair]) -> ERSResult | ShockSpectrumPair:
    """Compute an envelope across compatible shock-response spectra.

    Parameters
    ----------
    results : sequence
        Sequence containing either one-sided ``ERSResult`` values or
        two-sided ``ShockSpectrumPair`` values. Mixing the two forms in a
        single call is not allowed.

    Returns
    -------
    object
        Enveloped one-sided or two-sided shock-response spectrum,
        matching the representation of the input sequence.

    Notes
    -----
    This function accepts only spectra whose ERS compatibility metadata
    identifies them as ``"shock_response_spectrum"``. For pair-valued
    inputs, the negative and positive sides are enveloped independently.
    """
    if len(results) == 0:
        raise ValidationError("results must not be empty.")
    first = results[0]
    if isinstance(first, ShockSpectrumPair):
        return _envelope_shock_pairs(results, expected_kind="shock_response_spectrum", source="envelope_srs")  # type: ignore[arg-type]
    if isinstance(first, ERSResult):
        return _envelope_shock_results(results, expected_kind="shock_response_spectrum", source="envelope_srs")  # type: ignore[arg-type]
    raise ValidationError("results must contain ERSResult or ShockSpectrumPair values.")


def envelope_pvss(results: Sequence[ERSResult | ShockSpectrumPair]) -> ERSResult | ShockSpectrumPair:
    """Compute an envelope across compatible pseudo-velocity shock spectra.

    Parameters
    ----------
    results : sequence
        Sequence containing either one-sided ``ERSResult`` values or
        two-sided ``ShockSpectrumPair`` values. Mixing the two forms in a
        single call is not allowed.

    Returns
    -------
    object
        Enveloped one-sided or two-sided PVSS result, matching the
        representation of the input sequence.

    Notes
    -----
    This function accepts only spectra whose ERS compatibility metadata
    identifies them as ``"pseudo_velocity_shock_spectrum"``. For
    pair-valued inputs, the negative and positive sides are enveloped
    independently.
    """
    if len(results) == 0:
        raise ValidationError("results must not be empty.")
    first = results[0]
    if isinstance(first, ShockSpectrumPair):
        return _envelope_shock_pairs(results, expected_kind="pseudo_velocity_shock_spectrum", source="envelope_pvss")  # type: ignore[arg-type]
    if isinstance(first, ERSResult):
        return _envelope_shock_results(results, expected_kind="pseudo_velocity_shock_spectrum", source="envelope_pvss")  # type: ignore[arg-type]
    raise ValidationError("results must contain ERSResult or ShockSpectrumPair values.")
