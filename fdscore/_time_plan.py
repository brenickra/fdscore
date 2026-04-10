"""Compatibility checks for reusable time-domain FDS transfer plans.

This module validates whether a stored :class:`fdscore.types.FDSTimePlan`
can be reused for a new call without rebuilding the FFT-domain transfer
matrix. The checks cover sampling configuration, oscillator grid,
damping, response metric, and transfer-matrix shape.
"""

from __future__ import annotations

import numpy as np

from .types import FDSTimePlan
from .validate import ValidationError


def validate_time_plan_compatibility(
    *,
    plan: FDSTimePlan,
    fs: float,
    n_samples: int,
    f0: np.ndarray,
    zeta: float,
    metric: str,
) -> np.ndarray:
    """Validate that an ``FDSTimePlan`` matches a requested time-domain setup.

    Parameters
    ----------
    plan : FDSTimePlan
        Precomputed transfer plan to validate.
    fs : float
        Sampling rate in Hz of the current signal.
    n_samples : int
        Number of samples in the current signal.
    f0 : numpy.ndarray
        Expected oscillator frequency grid in Hz after any Nyquist
        validation or clipping.
    zeta : float
        Expected oscillator damping ratio.
    metric : str
        Expected SDOF response metric.

    Returns
    -------
    numpy.ndarray
        Validated transfer matrix ``H`` stored in the plan.

    Notes
    -----
    Reusing a cached time-domain transfer plan is valid only when the new
    call matches the original sampling and oscillator configuration. A
    mismatch in frequency grid can occur, for example, when plan creation
    and the current call use different Nyquist-clipping behavior.
    """
    if not isinstance(plan, FDSTimePlan):
        raise ValidationError("plan must be an instance of FDSTimePlan.")

    if int(plan.n_samples) != int(n_samples):
        raise ValidationError(f"FDSTimePlan.n_samples mismatch: plan={plan.n_samples}, input={n_samples}")
    if str(plan.metric) != str(metric):
        raise ValidationError(f"FDSTimePlan.metric mismatch: plan={plan.metric}, input={metric}")

    if not np.isclose(float(plan.fs), float(fs), rtol=0.0, atol=1e-12):
        raise ValidationError(f"FDSTimePlan.fs mismatch: plan={plan.fs}, input={fs}")
    if not np.isclose(float(plan.zeta), float(zeta), rtol=0.0, atol=1e-12):
        raise ValidationError(f"FDSTimePlan.zeta mismatch: plan={plan.zeta}, input={zeta}")

    f_plan = np.asarray(plan.f, dtype=float)
    if f_plan.shape != f0.shape or not np.allclose(f_plan, f0, rtol=0.0, atol=1e-12):
        raise ValidationError(
            "FDSTimePlan frequency grid mismatch with provided sdof/fs. "
            "Check sdof, fs, and whether plan creation and the current call used different Nyquist clipping behavior via strict_nyquist."
        )

    h = np.asarray(plan.H)
    n_bins = int(np.fft.rfftfreq(int(n_samples), d=1.0 / float(fs)).size)
    if h.shape != (f0.size, n_bins):
        raise ValidationError(
            "FDSTimePlan.H has unexpected shape. "
            f"Expected {(f0.size, n_bins)}, got {h.shape}."
        )
    if not np.all(np.isfinite(h)):
        raise ValidationError("FDSTimePlan.H must contain only finite values.")
    return h
