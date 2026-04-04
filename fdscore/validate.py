from __future__ import annotations

import numpy as np
from .types import SNParams, SDOFParams, FDSResult


class ValidationError(ValueError):
    """Raised when inputs are invalid or incompatible."""


def validate_sn(sn: SNParams) -> None:
    if not np.isfinite(sn.slope_k) or sn.slope_k <= 0:
        raise ValidationError("SNParams.slope_k must be finite and > 0.")
    if not np.isfinite(sn.ref_stress) or sn.ref_stress <= 0:
        raise ValidationError("SNParams.ref_stress must be finite and > 0.")
    if not np.isfinite(sn.ref_cycles) or sn.ref_cycles <= 0:
        raise ValidationError("SNParams.ref_cycles must be finite and > 0.")


def resolve_p_scale(*, p_scale: float | None, sn: SNParams) -> float:
    """Resolve p_scale under the normalized-vs-physical workflow contract."""
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
    if not np.isfinite(sdof.q) or sdof.q <= 0:
        raise ValidationError("SDOFParams.q must be finite and > 0.")
    if sdof.f is None:
        if sdof.fmin is None or sdof.fmax is None or sdof.df is None:
            raise ValidationError("Provide either sdof.f OR (fmin, fmax, df).")
        if not (np.isfinite(sdof.fmin) and np.isfinite(sdof.fmax) and np.isfinite(sdof.df)):
            raise ValidationError("SDOFParams (fmin,fmax,df) must be finite.")
        if sdof.fmin <= 0 or sdof.fmax <= 0 or sdof.df <= 0:
            raise ValidationError("SDOFParams fmin,fmax,df must be > 0.")
        if sdof.fmax <= sdof.fmin:
            raise ValidationError("SDOFParams.fmax must be > fmin.")
    else:
        f = np.asarray(sdof.f, dtype=float)
        validate_frequency_vector(f)


def validate_nyquist(f: np.ndarray, fs: float, strict: bool = True, tol: float = 1e-12) -> np.ndarray:
    if not np.isfinite(fs) or fs <= 0:
        raise ValidationError("Sampling rate fs must be finite and > 0.")
    nyq = fs / 2.0
    if strict and np.max(f) >= (nyq - tol):
        raise ValidationError(f"Frequency grid exceeds Nyquist: max(f)={np.max(f)} >= fs/2={nyq}.")
    if strict:
        return f
    mask = f < (nyq - tol)
    if not np.any(mask):
        raise ValidationError("All oscillator frequencies are above Nyquist after clipping.")
    return f[mask]


def compat_dict(sn: SNParams, metric: str, q: float, p_scale: float, f: np.ndarray, engine: str) -> dict:
    return {
        "engine": engine,
        "metric": metric,
        "q": float(q),
        "p_scale": float(p_scale),
        "sn": {
            "slope_k": float(sn.slope_k),
            "ref_stress": float(sn.ref_stress),
            "ref_cycles": float(sn.ref_cycles),
            "amplitude_from_range": bool(sn.amplitude_from_range),
        },
        "fds_kind": "damage_spectrum",
    }


def normalize_sn_compat(sn_sig) -> dict[str, float | bool]:
    if not isinstance(sn_sig, dict):
        raise ValidationError("FDS compat metadata field 'sn' must be a dictionary.")

    if {"k", "Sref", "Nref", "range2amp"}.issubset(sn_sig.keys()):
        return {
            "slope_k": float(sn_sig["k"]),
            "ref_stress": float(sn_sig["Sref"]),
            "ref_cycles": float(sn_sig["Nref"]),
            "amplitude_from_range": bool(sn_sig["range2amp"]),
        }

    if {"slope_k", "ref_stress", "ref_cycles", "amplitude_from_range"}.issubset(sn_sig.keys()):
        return {
            "slope_k": float(sn_sig["slope_k"]),
            "ref_stress": float(sn_sig["ref_stress"]),
            "ref_cycles": float(sn_sig["ref_cycles"]),
            "amplitude_from_range": bool(sn_sig["amplitude_from_range"]),
        }

    raise ValidationError(
        "Invalid S-N metadata in FDS compat. Expected either legacy keys "
        "{'k','Sref','Nref','range2amp'} or current keys "
        "{'slope_k','ref_stress','ref_cycles','amplitude_from_range'}."
    )


def assert_fds_compatible(a: FDSResult, b: FDSResult, f_rtol: float = 0.0, f_atol: float = 1e-9) -> None:
    ca = (a.meta or {}).get("compat", {})
    cb = (b.meta or {}).get("compat", {})
    for k in ("metric", "q", "p_scale", "fds_kind"):
        if ca.get(k) != cb.get(k):
            raise ValidationError(f"Incompatible FDS metadata field '{k}': {ca.get(k)} != {cb.get(k)}")

    if ca.get("sn", {}) != cb.get("sn", {}):
        raise ValidationError("Incompatible S-N parameters in FDS metadata.")

    fa = np.asarray(a.f, dtype=float)
    fb = np.asarray(b.f, dtype=float)
    if fa.shape != fb.shape or not np.allclose(fa, fb, rtol=f_rtol, atol=f_atol):
        raise ValidationError("Incompatible frequency grids. Use explicit regridding outside core.")


def ensure_compat_inversion(*, target, metric: str, q: float, p_scale: float, sn) -> None:
    """Validate that a target FDS is compatible with a proposed inversion configuration."""
    if target.meta is None or "compat" not in target.meta:
        raise ValidationError("Target FDS is missing meta['compat']; cannot guarantee compatible inversion.")
    c = target.meta["compat"]
    if str(c.get("metric")) != str(metric):
        raise ValidationError(f"Incompatible metric: target={c.get('metric')} vs inversion={metric}")
    if float(c.get("q")) != float(q):
        raise ValidationError(f"Incompatible Q: target={c.get('q')} vs inversion={q}")
    if float(c.get("p_scale")) != float(p_scale):
        raise ValidationError(f"Incompatible p_scale: target={c.get('p_scale')} vs inversion={p_scale}")
    # SN signature match
    sn_sig = c.get("sn")
    cur_sig = {
        "slope_k": float(sn.slope_k),
        "ref_stress": float(sn.ref_stress),
        "ref_cycles": float(sn.ref_cycles),
        "amplitude_from_range": bool(sn.amplitude_from_range),
    }

    sn_sig = normalize_sn_compat(sn_sig)
    if sn_sig != cur_sig:
        raise ValidationError(f"Incompatible SN parameters between target and inversion.")

