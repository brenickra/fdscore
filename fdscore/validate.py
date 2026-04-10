"""Validation and compatibility contracts used across the public API.

This module centralizes input validation, metadata normalization, and
cross-result compatibility checks for FDS and ERS workflows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .types import SNParams, SDOFParams, FDSResult, ERSResult


class ValidationError(ValueError):
    """Raised when inputs are invalid or mutually incompatible."""


def _bool_flag_or_raise(value, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValidationError(f"{field} must be a boolean.")


def _finite_float_or_raise(value, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be finite.") from None
    if not np.isfinite(out):
        raise ValidationError(f"{field} must be finite.")
    return out


def _finite_positive_float_or_raise(value, *, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be finite and > 0.") from None
    if not np.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{field} must be finite and > 0.")
    return out


@dataclass(frozen=True, slots=True)
class SNCompatSignature:
    """Normalized S-N metadata used in compatibility checks.

    Parameters
    ----------
    slope_k : float
        S-N slope exponent.
    ref_stress : float
        Reference stress used to define the S-N intercept.
    ref_cycles : float
        Reference cycle count associated with ``ref_stress``.
    amplitude_from_range : bool
        Rainflow convention indicating whether the damage-driving amplitude is
        obtained from ``range / 2``.
    """
    slope_k: float
    ref_stress: float
    ref_cycles: float
    amplitude_from_range: bool

    @classmethod
    def from_sn(cls, sn: SNParams) -> "SNCompatSignature":
        return cls(
            slope_k=float(sn.slope_k),
            ref_stress=float(sn.ref_stress),
            ref_cycles=float(sn.ref_cycles),
            amplitude_from_range=_bool_flag_or_raise(
                sn.amplitude_from_range,
                field="SNParams.amplitude_from_range",
            ),
        )

    @classmethod
    def from_payload(cls, sn_sig) -> "SNCompatSignature":
        if not isinstance(sn_sig, dict):
            raise ValidationError("FDS compat metadata field 'sn' must be a dictionary.")

        required = ("slope_k", "ref_stress", "ref_cycles", "amplitude_from_range")
        missing = [name for name in required if name not in sn_sig]
        if missing:
            raise ValidationError(
                "Invalid S-N metadata in FDS compat. Missing required fields: "
                + ", ".join(missing)
                + "."
            )

        return cls(
            slope_k=float(sn_sig["slope_k"]),
            ref_stress=float(sn_sig["ref_stress"]),
            ref_cycles=float(sn_sig["ref_cycles"]),
            amplitude_from_range=_bool_flag_or_raise(
                sn_sig["amplitude_from_range"],
                field="FDS compat metadata field 'sn.amplitude_from_range'",
            ),
        )

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "slope_k": float(self.slope_k),
            "ref_stress": float(self.ref_stress),
            "ref_cycles": float(self.ref_cycles),
            "amplitude_from_range": bool(self.amplitude_from_range),
        }


@dataclass(frozen=True, slots=True)
class FDSCompatSignature:
    """Compatibility signature carried by fatigue-damage spectra.

    Parameters
    ----------
    engine : str
        Name of the computational route that produced the FDS.
    metric : str
        Response metric represented by the spectrum.
    q : float
        Oscillator quality factor used in the response model.
    p_scale : float
        Response-to-fatigue scaling factor used before damage evaluation.
    sn : SNCompatSignature
        Canonical S-N definition associated with the damage spectrum.
    fds_kind : str, optional
        Logical kind of spectrum. The default value identifies Miner-damage
        spectra.
    """
    engine: str
    metric: str
    q: float
    p_scale: float
    sn: SNCompatSignature
    fds_kind: str = "damage_spectrum"

    @classmethod
    def from_inputs(
        cls,
        *,
        sn: SNParams,
        metric: str,
        q: float,
        p_scale: float,
        engine: str,
    ) -> "FDSCompatSignature":
        return cls(
            engine=str(engine),
            metric=str(metric),
            q=float(q),
            p_scale=float(p_scale),
            sn=SNCompatSignature.from_sn(sn),
            fds_kind="damage_spectrum",
        )

    @classmethod
    def from_payload(cls, compat) -> "FDSCompatSignature":
        if not isinstance(compat, dict):
            raise ValidationError("FDS compat metadata must be a dictionary.")

        required = ("engine", "metric", "q", "p_scale", "sn", "fds_kind")
        missing = [name for name in required if name not in compat]
        if missing:
            raise ValidationError(f"FDS compat metadata is missing required fields: {', '.join(missing)}.")

        return cls(
            engine=str(compat["engine"]),
            metric=str(compat["metric"]),
            q=float(compat["q"]),
            p_scale=float(compat["p_scale"]),
            sn=SNCompatSignature.from_payload(compat["sn"]),
            fds_kind=str(compat["fds_kind"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "metric": self.metric,
            "q": float(self.q),
            "p_scale": float(self.p_scale),
            "sn": self.sn.as_dict(),
            "fds_kind": self.fds_kind,
        }


@dataclass(frozen=True, slots=True)
class ERSCompatSignature:
    """Compatibility signature carried by extreme-response spectra.

    Parameters
    ----------
    engine : str
        Name of the computational route that produced the ERS.
    metric : str
        Response metric represented by the spectrum.
    q : float
        Oscillator quality factor used in the response model.
    peak_mode : str
        Peak convention associated with the reported response values.
    ers_kind : str, optional
        Logical kind of response spectrum.
    """
    engine: str
    metric: str
    q: float
    peak_mode: str
    ers_kind: str = "response_spectrum"

    @classmethod
    def from_inputs(
        cls,
        *,
        metric: str,
        q: float,
        peak_mode: str,
        engine: str,
    ) -> "ERSCompatSignature":
        return cls(
            engine=str(engine),
            metric=str(metric),
            q=float(q),
            peak_mode=str(peak_mode),
            ers_kind="response_spectrum",
        )

    @classmethod
    def from_payload(cls, compat) -> "ERSCompatSignature":
        if not isinstance(compat, dict):
            raise ValidationError("ERS compat metadata must be a dictionary.")

        required = ("engine", "metric", "q", "peak_mode", "ers_kind")
        missing = [name for name in required if name not in compat]
        if missing:
            raise ValidationError(f"ERS compat metadata is missing required fields: {', '.join(missing)}.")

        return cls(
            engine=str(compat["engine"]),
            metric=str(compat["metric"]),
            q=float(compat["q"]),
            peak_mode=str(compat["peak_mode"]),
            ers_kind=str(compat["ers_kind"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "metric": self.metric,
            "q": float(self.q),
            "peak_mode": self.peak_mode,
            "ers_kind": self.ers_kind,
        }


def validate_sn(sn: SNParams) -> None:
    """Validate an ``SNParams`` instance.

    Parameters
    ----------
    sn : SNParams
        S-N definition to validate.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` if any
        field violates the library contract.
    """
    _finite_positive_float_or_raise(sn.slope_k, field="SNParams.slope_k")
    _finite_positive_float_or_raise(sn.ref_stress, field="SNParams.ref_stress")
    _finite_positive_float_or_raise(sn.ref_cycles, field="SNParams.ref_cycles")
    _bool_flag_or_raise(sn.amplitude_from_range, field="SNParams.amplitude_from_range")


def resolve_p_scale(*, p_scale: float | None, sn: SNParams) -> float:
    """Resolve the effective response scale for fatigue calculations.

    Parameters
    ----------
    p_scale : float or None
        Explicit response scale factor. When provided, it must be finite and
        strictly positive.
    sn : SNParams
        S-N definition used to determine whether the default normalized
        convention is allowed.

    Returns
    -------
    float
        Resolved positive scale factor.

    Notes
    -----
    The implicit default ``p_scale = 1.0`` is accepted only when the S-N
    definition is normalized, that is, when ``ref_stress = 1`` and
    ``ref_cycles = 1``. Physical workflows must pass ``p_scale`` explicitly.
    """
    if p_scale is not None:
        val = float(p_scale)
        if not np.isfinite(val) or val <= 0.0:
            raise ValidationError("p_scale must be finite and > 0.")
        return val

    if float(sn.ref_stress) == 1.0 and float(sn.ref_cycles) == 1.0:
        return 1.0

    raise ValidationError(
        "p_scale must be provided explicitly when using non-unit SN reference values. "
        "Use normalized SN parameters (ref_stress=1, ref_cycles=1) to rely on the default p_scale=1."
    )


def validate_frequency_vector(f: np.ndarray) -> None:
    """Validate a frequency grid used by spectral or oscillator routines.

    Parameters
    ----------
    f : numpy.ndarray
        One-dimensional frequency vector in Hz.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` if the
        frequency grid is not finite, strictly increasing, and strictly
        positive.
    """
    if f.ndim != 1:
        raise ValidationError("Frequency vector f must be 1D.")
    if f.size < 2:
        raise ValidationError("Frequency vector f must have length >= 2.")
    if not np.all(np.isfinite(f)):
        raise ValidationError("Frequency vector f must contain only finite values.")
    if np.any(f <= 0):
        raise ValidationError("Frequency vector f must be strictly positive (Hz).")
    df = np.diff(f)
    if not np.all(df > 0):
        raise ValidationError("Frequency vector f must be strictly increasing.")


def validate_sdof(sdof: SDOFParams) -> None:
    """Validate an ``SDOFParams`` definition.

    Parameters
    ----------
    sdof : SDOFParams
        Oscillator-grid definition to validate.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` when the
        SDOF configuration is incomplete or inconsistent.

    Notes
    -----
    The library accepts exactly one of two grid conventions:

    1. An explicit frequency vector ``sdof.f``.
    2. An implicit linear grid defined by ``(fmin, fmax, df)``.
    """
    _finite_positive_float_or_raise(sdof.q, field="SDOFParams.q")
    if sdof.f is not None and (sdof.fmin is not None or sdof.fmax is not None or sdof.df is not None):
        raise ValidationError("Provide either sdof.f OR (fmin, fmax, df), not both.")
    if sdof.f is None:
        if sdof.fmin is None or sdof.fmax is None or sdof.df is None:
            raise ValidationError("Provide either sdof.f OR (fmin, fmax, df).")
        try:
            fmin = float(sdof.fmin)
            fmax = float(sdof.fmax)
            df = float(sdof.df)
        except (TypeError, ValueError):
            raise ValidationError("SDOFParams (fmin,fmax,df) must be finite.") from None
        if not (np.isfinite(fmin) and np.isfinite(fmax) and np.isfinite(df)):
            raise ValidationError("SDOFParams (fmin,fmax,df) must be finite.")
        if fmin <= 0 or fmax <= 0 or df <= 0:
            raise ValidationError("SDOFParams fmin,fmax,df must be > 0.")
        if fmax <= fmin:
            raise ValidationError("SDOFParams.fmax must be > fmin.")
    else:
        try:
            f = np.asarray(sdof.f, dtype=float)
        except (TypeError, ValueError):
            raise ValidationError("Frequency vector f must contain only finite values.") from None
        validate_frequency_vector(f)


def _validate_nyquist_with_info(
    f: np.ndarray,
    fs: float,
    *,
    strict: bool = True,
    tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, object]]:
    fs_val = _finite_positive_float_or_raise(fs, field="Sampling rate fs")
    f_arr = np.asarray(f, dtype=float)
    nyq = fs_val / 2.0
    requested_count = int(f_arr.size)
    requested_fmax = float(np.max(f_arr))

    if strict and requested_fmax >= (nyq - tol):
        raise ValidationError(f"Frequency grid exceeds Nyquist: max(f)={requested_fmax} >= fs/2={nyq}.")

    if strict:
        out = f_arr
    else:
        mask = f_arr < (nyq - tol)
        if not np.any(mask):
            raise ValidationError("All oscillator frequencies are above Nyquist after clipping.")
        out = f_arr[mask]

    info = {
        "strict_nyquist": bool(strict),
        "nyquist_hz": float(nyq),
        "nyquist_clipped": bool(out.size != requested_count),
        "requested_frequency_count": int(requested_count),
        "returned_frequency_count": int(out.size),
        "requested_fmax_hz": float(requested_fmax),
        "returned_fmax_hz": float(np.max(out)),
    }
    return out, info


def validate_nyquist(f: np.ndarray, fs: float, strict: bool = True, tol: float = 1e-12) -> np.ndarray:
    """Validate an oscillator grid against the Nyquist limit.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency grid in Hz.
    fs : float
        Sampling rate in Hz.
    strict : bool, optional
        If ``True``, frequencies at or above Nyquist raise
        ``ValidationError``. If ``False``, out-of-range frequencies are
        removed.
    tol : float, optional
        Numerical tolerance applied to the Nyquist comparison.

    Returns
    -------
    numpy.ndarray
        Validated frequency grid. In non-strict mode, this may be a clipped
        subset of the original array.
    """
    out, _ = _validate_nyquist_with_info(f, fs, strict=strict, tol=tol)
    return out


def compat_dict(sn: SNParams, metric: str, q: float, p_scale: float, engine: str) -> dict[str, object]:
    """Build canonical compatibility metadata for an FDS result.

    Parameters
    ----------
    sn : SNParams
        S-N definition associated with the FDS.
    metric : str
        Response metric represented by the spectrum.
    q : float
        Oscillator quality factor.
    p_scale : float
        Response-to-fatigue scaling factor.
    engine : str
        Name of the computational route that produced the result.

    Returns
    -------
    dict
        Dictionary payload suitable for storage in ``FDSResult.meta["compat"]``.
    """
    return FDSCompatSignature.from_inputs(
        sn=sn,
        metric=metric,
        q=q,
        p_scale=p_scale,
        engine=engine,
    ).as_dict()


def ers_compat_dict(*, metric: str, q: float, peak_mode: str, engine: str, ers_kind: str = "response_spectrum") -> dict[str, object]:
    """Build canonical compatibility metadata for an ERS result.

    Parameters
    ----------
    metric : str
        Response metric represented by the spectrum.
    q : float
        Oscillator quality factor.
    peak_mode : str
        Peak convention used by the ERS.
    engine : str
        Name of the computational route that produced the result.
    ers_kind : str, optional
        Logical kind of response spectrum.

    Returns
    -------
    dict
        Dictionary payload suitable for storage in ``ERSResult.meta["compat"]``.
    """
    return ERSCompatSignature(
        engine=str(engine),
        metric=str(metric),
        q=float(q),
        peak_mode=str(peak_mode),
        ers_kind=str(ers_kind),
    ).as_dict()


def parse_fds_compat(compat) -> FDSCompatSignature:
    """Parse stored FDS compatibility metadata.

    Parameters
    ----------
    compat : object
        Raw payload stored under ``meta["compat"]``.

    Returns
    -------
    FDSCompatSignature
        Parsed and normalized compatibility signature.
    """
    return FDSCompatSignature.from_payload(compat)


def parse_ers_compat(compat) -> ERSCompatSignature:
    """Parse stored ERS compatibility metadata.

    Parameters
    ----------
    compat : object
        Raw payload stored under ``meta["compat"]``.

    Returns
    -------
    ERSCompatSignature
        Parsed and normalized compatibility signature.
    """
    return ERSCompatSignature.from_payload(compat)


def _ensure_compat_float_match(*, actual, expected: float, field: str, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    actual_f = float(actual)
    expected_f = float(expected)
    if not np.isfinite(actual_f):
        raise ValidationError(f"Incompatible {field}: target={actual} is not finite")
    if not np.isclose(actual_f, expected_f, rtol=rtol, atol=atol):
        raise ValidationError(f"Incompatible {field}: target={actual_f} vs inversion={expected_f}")


def assert_fds_compatible(a: FDSResult, b: FDSResult, f_rtol: float = 0.0, f_atol: float = 1e-9) -> None:
    """Assert that two FDS results can be combined without regridding.

    Parameters
    ----------
    a : FDSResult
        Reference fatigue-damage spectrum.
    b : FDSResult
        Candidate spectrum to compare against ``a``.
    f_rtol : float, optional
        Relative tolerance used for frequency-grid comparison.
    f_atol : float, optional
        Absolute tolerance used for frequency-grid comparison.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` when the
        compatibility metadata or frequency grids differ.
    """
    ca = parse_fds_compat((a.meta or {}).get("compat", {}))
    cb = parse_fds_compat((b.meta or {}).get("compat", {}))

    if ca.metric != cb.metric:
        raise ValidationError(f"Incompatible FDS metadata field 'metric': {ca.metric} != {cb.metric}")
    if ca.q != cb.q:
        raise ValidationError(f"Incompatible FDS metadata field 'q': {ca.q} != {cb.q}")
    if ca.p_scale != cb.p_scale:
        raise ValidationError(f"Incompatible FDS metadata field 'p_scale': {ca.p_scale} != {cb.p_scale}")
    if ca.fds_kind != cb.fds_kind:
        raise ValidationError(f"Incompatible FDS metadata field 'fds_kind': {ca.fds_kind} != {cb.fds_kind}")
    if ca.sn != cb.sn:
        raise ValidationError("Incompatible S-N parameters in FDS metadata.")

    fa = np.asarray(a.f, dtype=float)
    fb = np.asarray(b.f, dtype=float)
    if fa.shape != fb.shape or not np.allclose(fa, fb, rtol=f_rtol, atol=f_atol):
        raise ValidationError("Incompatible frequency grids. Use explicit regridding outside core.")


def assert_ers_compatible(a: ERSResult, b: ERSResult, f_rtol: float = 0.0, f_atol: float = 1e-9) -> None:
    """Assert that two ERS results can be combined without regridding.

    Parameters
    ----------
    a : ERSResult
        Reference response spectrum.
    b : ERSResult
        Candidate spectrum to compare against ``a``.
    f_rtol : float, optional
        Relative tolerance used for frequency-grid comparison.
    f_atol : float, optional
        Absolute tolerance used for frequency-grid comparison.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` when the
        compatibility metadata or frequency grids differ.
    """
    ca = parse_ers_compat((a.meta or {}).get("compat", {}))
    cb = parse_ers_compat((b.meta or {}).get("compat", {}))

    if ca.metric != cb.metric:
        raise ValidationError(f"Incompatible ERS metadata field 'metric': {ca.metric} != {cb.metric}")
    if ca.q != cb.q:
        raise ValidationError(f"Incompatible ERS metadata field 'q': {ca.q} != {cb.q}")
    if ca.peak_mode != cb.peak_mode:
        raise ValidationError(f"Incompatible ERS metadata field 'peak_mode': {ca.peak_mode} != {cb.peak_mode}")
    if ca.ers_kind != cb.ers_kind:
        raise ValidationError(f"Incompatible ERS metadata field 'ers_kind': {ca.ers_kind} != {cb.ers_kind}")

    fa = np.asarray(a.f, dtype=float)
    fb = np.asarray(b.f, dtype=float)
    if fa.shape != fb.shape or not np.allclose(fa, fb, rtol=f_rtol, atol=f_atol):
        raise ValidationError("Incompatible ERS frequency grids. Use explicit regridding outside core.")


def ensure_compat_inversion(*, target, metric: str, q: float, p_scale: float, sn, sdof: SDOFParams | None = None) -> None:
    """Validate that a target FDS matches a proposed inversion setup.

    Parameters
    ----------
    target : FDSResult
        Target fatigue-damage spectrum to be inverted.
    metric : str
        Response metric expected by the inversion route.
    q : float
        Oscillator quality factor expected by the inversion route.
    p_scale : float
        Response-to-fatigue scaling factor expected by the inversion route.
    sn : SNParams
        S-N definition expected by the inversion route.
    sdof : SDOFParams or None, optional
        Optional oscillator-grid definition. When provided, the target
        frequency grid must match the grid implied by ``sdof``.

    Returns
    -------
    None
        The function returns ``None`` and raises ``ValidationError`` when the
        target metadata is missing or incompatible with the proposed
        inversion.
    """
    if target.meta is None or "compat" not in target.meta:
        raise ValidationError("Target FDS is missing meta['compat']; cannot guarantee compatible inversion.")

    c = parse_fds_compat(target.meta["compat"])
    if c.metric != str(metric):
        raise ValidationError(f"Incompatible metric: target={c.metric} vs inversion={metric}")
    _ensure_compat_float_match(actual=c.q, expected=q, field="Q")
    _ensure_compat_float_match(actual=c.p_scale, expected=p_scale, field="p_scale")

    cur_sig = SNCompatSignature.from_sn(sn)
    if c.sn != cur_sig:
        raise ValidationError("Incompatible SN parameters between target and inversion.")

    if sdof is not None:
        from .grid import build_frequency_grid

        f_target = np.asarray(target.f, dtype=float)
        f_expected = np.asarray(build_frequency_grid(sdof), dtype=float)
        if f_target.shape != f_expected.shape or not np.allclose(f_target, f_expected, rtol=0.0, atol=1e-9):
            raise ValidationError("Target FDS frequency grid does not match the grid implied by sdof.")

