from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .types import SNParams, SDOFParams, FDSResult


class ValidationError(ValueError):
    """Raised when inputs are invalid or incompatible."""


@dataclass(frozen=True, slots=True)
class SNCompatSignature:
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
            amplitude_from_range=bool(sn.amplitude_from_range),
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
            amplitude_from_range=bool(sn_sig["amplitude_from_range"]),
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


def compat_dict(sn: SNParams, metric: str, q: float, p_scale: float, engine: str) -> dict[str, object]:
    return FDSCompatSignature.from_inputs(
        sn=sn,
        metric=metric,
        q=q,
        p_scale=p_scale,
        engine=engine,
    ).as_dict()


def parse_fds_compat(compat) -> FDSCompatSignature:
    return FDSCompatSignature.from_payload(compat)


def _ensure_compat_float_match(*, actual, expected: float, field: str, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    actual_f = float(actual)
    expected_f = float(expected)
    if not np.isfinite(actual_f):
        raise ValidationError(f"Incompatible {field}: target={actual} is not finite")
    if not np.isclose(actual_f, expected_f, rtol=rtol, atol=atol):
        raise ValidationError(f"Incompatible {field}: target={actual_f} vs inversion={expected_f}")


def assert_fds_compatible(a: FDSResult, b: FDSResult, f_rtol: float = 0.0, f_atol: float = 1e-9) -> None:
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


def ensure_compat_inversion(*, target, metric: str, q: float, p_scale: float, sn) -> None:
    """Validate that a target FDS is compatible with a proposed inversion configuration."""
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
