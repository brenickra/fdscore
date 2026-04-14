"""Compact internal Dirlik fatigue-life utilities.

This module implements the small subset of spectral-fatigue machinery needed by
``fdscore`` for its internal Dirlik spectral-fatigue route:

- uniaxial one-sided response PSD input;
- spectral moments ``m0, m1, m2, m4``;
- expected peak frequency ``m_p``;
- closed-form Dirlik damage intensity and life.

It is intentionally narrow and should not be treated as a general-purpose
spectral-fatigue framework.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from .validate import ValidationError


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Return a version-compatible trapezoidal integral."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _dirlik_spectral_moments(
    *,
    f_hz: np.ndarray,
    psd: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute the spectral moments and ``m_p`` used by Dirlik."""
    f = np.asarray(f_hz, dtype=float).reshape(-1)
    p = np.asarray(psd, dtype=float).reshape(-1)

    if f.size < 2 or p.size < 2 or f.shape != p.shape:
        raise ValidationError("Dirlik inputs require aligned 1D frequency and PSD arrays of length >= 2.")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(p))):
        raise ValidationError("Dirlik inputs must be finite.")
    if not np.all(np.diff(f) > 0):
        raise ValidationError("Dirlik frequency grid must be strictly increasing.")
    if np.any(p < 0.0):
        raise ValidationError("Dirlik PSD input must be non-negative.")

    omega = 2.0 * np.pi * f
    m0 = _trapezoid(p, f)
    m1 = _trapezoid(omega * p, f)
    m2 = _trapezoid((omega**2) * p, f)
    m4 = _trapezoid((omega**4) * p, f)

    if not all(np.isfinite(v) for v in (m0, m1, m2, m4)):
        raise ValidationError("Dirlik spectral moments must be finite.")
    if m0 <= 0.0 or m2 <= 0.0 or m4 <= 0.0:
        raise ValidationError("Dirlik spectral moments m0, m2, and m4 must be > 0.")

    mp = (1.0 / (2.0 * np.pi)) * np.sqrt(m4 / m2)
    if not np.isfinite(mp) or mp <= 0.0:
        raise ValidationError("Dirlik expected peak frequency m_p must be finite and > 0.")

    return m0, m1, m2, m4, mp


def _dirlik_coefficients(
    *,
    m0: float,
    m1: float,
    m2: float,
    m4: float,
) -> tuple[float, float, float, float, float]:
    """Compute the closed-form Dirlik mixture coefficients."""
    if min(m0, m1, m2, m4) <= 0.0:
        raise ValidationError("Dirlik moments must be > 0.")

    sqrt_m0 = np.sqrt(m0)
    c1 = (m1 / m0) * np.sqrt(m2 / m4)
    c2 = m2 / np.sqrt(m0 * m4)

    denom_g1 = 1.0 + c2**2
    if denom_g1 == 0.0:
        raise ValidationError("Dirlik coefficient denominator became zero.")
    g1 = 2.0 * (c1 - c2**2) / denom_g1

    denom_r = 1.0 - c2 - g1 + g1**2
    if denom_r == 0.0:
        raise ValidationError("Dirlik coefficient denominator became zero.")
    r = (c2 - c1 - g1**2) / denom_r

    denom_g2 = 1.0 - r
    if denom_g2 == 0.0:
        raise ValidationError("Dirlik coefficient denominator became zero.")
    g2 = denom_r / denom_g2
    g3 = 1.0 - g1 - g2

    if g1 == 0.0:
        raise ValidationError("Dirlik coefficient g1 became zero.")
    q = 1.25 * (c2 - g3 - g2 * r) / g1

    coeffs = (g1, g2, g3, r, q)
    if not np.all(np.isfinite(coeffs)):
        raise ValidationError("Dirlik coefficients must be finite.")
    return coeffs


def dirlik_damage_intensity(
    *,
    f_hz: np.ndarray,
    psd: np.ndarray,
    C: float,
    k: float,
) -> float:
    """Return Dirlik damage intensity for a one-sided response PSD."""
    try:
        C_val = float(C)
        k_val = float(k)
    except (TypeError, ValueError):
        raise ValidationError("Dirlik parameters C and k must be finite and > 0.") from None
    if not np.isfinite(C_val) or not np.isfinite(k_val) or C_val <= 0.0 or k_val <= 0.0:
        raise ValidationError("Dirlik parameters C and k must be finite and > 0.")

    m0, m1, m2, m4, mp = _dirlik_spectral_moments(f_hz=f_hz, psd=psd)
    g1, g2, g3, r, q = _dirlik_coefficients(m0=m0, m1=m1, m2=m2, m4=m4)

    damage_rate = (mp / C_val) * (
        (sqrt2_m0 := np.sqrt(m0)) ** k_val
        * (
            g1 * (q**k_val) * special.gamma(1.0 + k_val)
            + (np.sqrt(2.0) ** k_val) * special.gamma(1.0 + k_val / 2.0) * (g2 * abs(r) ** k_val + g3)
        )
    )
    if not np.isfinite(damage_rate) or damage_rate <= 0.0:
        raise ValidationError(f"Internal Dirlik returned invalid damage intensity: {damage_rate}")
    return float(damage_rate)


def dirlik_life(
    *,
    f_hz: np.ndarray,
    psd: np.ndarray,
    C: float,
    k: float,
) -> float:
    """Return Dirlik fatigue life for a one-sided response PSD."""
    damage_rate = dirlik_damage_intensity(f_hz=f_hz, psd=psd, C=C, k=k)
    life = 1.0 / damage_rate
    if not np.isfinite(life) or life <= 0.0:
        raise ValidationError(f"Internal Dirlik returned invalid life: {life}")
    return float(life)
