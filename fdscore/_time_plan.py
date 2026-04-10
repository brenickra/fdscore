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
