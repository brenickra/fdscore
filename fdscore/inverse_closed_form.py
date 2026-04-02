from __future__ import annotations

from math import gamma
import numpy as np

from .types import FDSResult, PSDResult
from .validate import ValidationError

EPS = 1e-30


def compute_damage_to_dp_factor(*, p_scale: float, b: float, c_sn: float) -> float:
    """Compute conversion factor K between Miner damage and DP index.

        Damage = K * DP   =>   DP = Damage / K

    K = (p_scale^b)/C_SN * gamma(1 + b/2) / ((8*pi)^(b/2))
    """
    p_scale = float(p_scale)
    b = float(b)
    c_sn = float(c_sn)
    if p_scale <= 0 or b <= 0 or c_sn <= 0:
        raise ValidationError("compute_damage_to_dp_factor requires positive p_scale, b, and c_sn.")
    return (p_scale ** b) / c_sn * (gamma(1.0 + b / 2.0) / ((8.0 * np.pi) ** (b / 2.0)))


def compute_psd_from_fds_closed_form(
    *,
    f0_hz: np.ndarray,
    dp_fds: np.ndarray,
    zeta: float,
    b: float,
    test_duration_s: float,
) -> np.ndarray:
    """Closed-form inverse equation DP -> PSD (Henderson-Piersol / MIL-STD form).

    Equation:
        G(f) = f * zeta * (DP(f) / (f * T)) ** (2 / b)

    Returns acceleration PSD in input-units^2/Hz.
    """
    f_safe = np.clip(np.asarray(f0_hz, dtype=float), EPS, None)
    dp_safe = np.clip(np.asarray(dp_fds, dtype=float), EPS, None)
    t_safe = max(float(test_duration_s), EPS)
    zeta = max(float(zeta), EPS)
    b = float(b)

    return np.clip(
        f_safe * zeta * np.power(dp_safe / (f_safe * t_safe), 2.0 / b),
        EPS,
        None,
    )


def compute_fds_from_psd_closed_form(
    *,
    f0_hz: np.ndarray,
    psd: np.ndarray,
    zeta: float,
    b: float,
    test_duration_s: float,
) -> np.ndarray:
    """Closed-form direct equation PSD -> DP (for reconstruction check).

    Equation:
        DP(f) = f * T * (G(f) / (f * zeta)) ** (b / 2)
    """
    f_safe = np.clip(np.asarray(f0_hz, dtype=float), EPS, None)
    psd_safe = np.clip(np.asarray(psd, dtype=float), EPS, None)
    t_safe = max(float(test_duration_s), EPS)
    zeta = max(float(zeta), EPS)
    b = float(b)
    denom = np.clip(f_safe * zeta, EPS, None)

    return np.clip(
        f_safe * t_safe * np.power(psd_safe / denom, b / 2.0),
        EPS,
        None,
    )


def invert_fds_closed_form(
    fds: FDSResult,
    *,
    test_duration_s: float,
    strict_metric: bool = True,
) -> PSDResult:
    """Invert an FDS to an equivalent acceleration PSD using a closed-form equation.

    This function:
    - converts Damage -> DP using factor K from `(p_scale, S-N, b)`
    - converts DP -> PSD with `G(f) = f*zeta*(DP/(f*T))^(2/b)`

    Requirements
    ------------
    - `fds.meta['compat']` must exist and include: metric, q, p_scale, sn
    - metric must be `'pv'`
    - `test_duration_s` must be provided

    Notes
    -----
    When the target FDS was computed with compatible settings, the global damage
    scaling carried by `p_scale`, `ref_stress`, and `ref_cycles` cancels in the
    closed-form inversion. Those parameters affect absolute FDS magnitude, but not
    the equivalent inverted PSD.

    This implementation is intended for `metric="pv"` only. Passing
    `strict_metric=False` suppresses the guard, but it does not make the closed-form
    derivation valid for `disp`, `vel`, or `acc`.
    """
    if not np.isfinite(test_duration_s) or float(test_duration_s) <= 0:
        raise ValidationError("test_duration_s must be finite and > 0.")
    t_test = float(test_duration_s)

    compat = (fds.meta or {}).get("compat", {})
    metric = compat.get("metric")
    if strict_metric and metric != "pv":
        raise ValidationError("Closed-form inversion supports only metric='pv'.")

    q = compat.get("q", None)
    p_scale = compat.get("p_scale", None)
    sn = compat.get("sn", None)
    if q is None or p_scale is None or sn is None:
        raise ValidationError("FDS metadata missing required compat fields: q, p_scale, sn.")

    try:
        b = float(sn["slope_k"])
        ref_stress = float(sn["ref_stress"])
        ref_cycles = float(sn["ref_cycles"])
    except Exception as e:
        raise ValidationError(f"Invalid S-N metadata in FDS compat: {e}") from e

    if b <= 0 or ref_stress <= 0 or ref_cycles <= 0:
        raise ValidationError("Invalid S-N parameters (must be > 0).")

    zeta = 1.0 / (2.0 * float(q))
    c_sn = ref_cycles * (ref_stress ** b)

    damage_to_dp = compute_damage_to_dp_factor(p_scale=float(p_scale), b=b, c_sn=c_sn)
    if damage_to_dp <= 0:
        raise ValidationError("Computed damage_to_dp_factor is invalid (<=0).")

    f = np.asarray(fds.f, dtype=float)
    damage = np.asarray(fds.damage, dtype=float)
    if f.ndim != 1 or damage.ndim != 1 or f.shape != damage.shape:
        raise ValidationError("fds.f and fds.damage must be 1D arrays of the same shape.")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(damage))):
        raise ValidationError("fds.f and fds.damage must contain only finite values.")
    if np.any(f <= 0):
        raise ValidationError("fds frequencies must be > 0 Hz.")

    dp = np.clip(damage / float(damage_to_dp), EPS, None)
    psd = compute_psd_from_fds_closed_form(f0_hz=f, dp_fds=dp, zeta=zeta, b=b, test_duration_s=t_test)

    recon_dp = compute_fds_from_psd_closed_form(f0_hz=f, psd=psd, zeta=zeta, b=b, test_duration_s=t_test)
    recon_damage = np.clip(recon_dp * float(damage_to_dp), EPS, None)

    mask = (damage > 0) & (recon_damage > 0)
    med_abs_log10 = float(np.median(np.abs(np.log10(recon_damage[mask]) - np.log10(damage[mask])))) if np.any(mask) else float("nan")

    meta = {
        "compat": {
            "method": "closed_form_hp",
            "metric": metric,
            "q": float(q),
            "zeta": float(zeta),
            "b": float(b),
            "p_scale": float(p_scale),
            "c_sn": float(c_sn),
        },
        "reconstruction": {
            "med_abs_log10": med_abs_log10,
        },
        "damage_to_dp_factor": float(damage_to_dp),
        "provenance": {"source": "invert_fds_closed_form", "equation": "G=f*zeta*(DP/(f*T))^(2/b)"},
    }
    return PSDResult(f=f, psd=psd, meta=meta)
