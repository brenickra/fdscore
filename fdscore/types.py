from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

Metric = Literal["pv", "acc", "vel", "disp"]


@dataclass(frozen=True, slots=True)
class SNParams:
    """S-N curve parameters used in Miner damage.

    Conventions:
    - slope_k: exponent k (>0)
    - ref_stress: S_ref (user units, e.g., MPa)
    - ref_cycles: N_ref
    - amplitude_from_range:
        If True, interpret rainflow `range` as 2*amplitude (amplitude = range/2).
        Must be consistent across FDS compute and inversion assumptions.

    Notes
    -----
    Two usage modes are common:

    - Physical: provide `ref_stress`, `ref_cycles`, and an application-specific `p_scale`
      when absolute damage magnitude matters.
    - Normalized: use :meth:`SNParams.normalized` together with `p_scale=1.0`
      when only the relative FDS shape and the equivalent inverted PSD matter.
    """
    slope_k: float
    ref_stress: float = 1.0
    ref_cycles: float = 1.0
    amplitude_from_range: bool = True

    @classmethod
    def normalized(
        cls,
        *,
        slope_k: float,
        amplitude_from_range: bool = True,
    ) -> "SNParams":
        """Return a normalized S-N definition with unit reference values.

        The normalized convention uses:
        - `ref_stress = 1.0`
        - `ref_cycles = 1.0`

        This is useful when the workflow is focused on FDS shape and FDS-to-PSD
        inversion, rather than absolute Miner damage magnitude.
        """
        return cls(
            slope_k=float(slope_k),
            ref_stress=1.0,
            ref_cycles=1.0,
            amplitude_from_range=bool(amplitude_from_range),
        )

    def C(self) -> float:
        """Return C = N_ref * S_ref^k."""
        return float(self.ref_cycles * (self.ref_stress ** self.slope_k))


@dataclass(frozen=True, slots=True)
class SDOFParams:
    """SDOF oscillator grid and response metric.

    Frequency grid can be provided in two ways:
    - Explicit vector `f` (Hz), strictly increasing, all positive.
    - Linear grid via fmin,fmax,df.
    """
    q: float
    metric: Metric = "pv"
    f: Optional[np.ndarray] = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None
    df: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PSDParams:
    """PSD estimation configuration.

    The current implementation supports Welch (`scipy.signal.welch`)
    as the internal estimator.
    """
    method: Literal["welch"] = "welch"
    window: str = "hann"
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    detrend: Literal["constant", "linear", "none"] = "constant"
    scaling: Literal["density"] = "density"
    onesided: bool = True
    fmin: Optional[float] = None
    fmax: Optional[float] = None


@dataclass(frozen=True, slots=True)
class FDSResult:
    """Fatigue Damage Spectrum result."""
    f: np.ndarray
    damage: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FDSTimePlan:
    """Precomputed transfer plan for repeated time-domain FDS calls.

    A plan stores the frequency grid and transfer matrix for a fixed
    `(fs, n_samples, sdof)` configuration and can be reused across
    channels/signals with the same sampling setup.
    """
    fs: float
    n_samples: int
    f: np.ndarray
    zeta: float
    metric: Metric
    H: np.ndarray


@dataclass(frozen=True, slots=True)
class PSDResult:
    """PSD result."""
    f: np.ndarray
    psd: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PSDMetricsResult:
    """Summary metrics computed from an acceleration PSD."""
    rms_acc_g: float
    rms_acc_m_s2: float
    peak_acc_g: float
    peak_acc_m_s2: float
    peak_factor: float
    zero_upcrossing_hz: float
    effective_cycles: float
    rms_vel_m_s: float
    peak_vel_m_s: float
    rms_disp_mm: float
    peak_disp_mm: float
    disp_pk_pk_mm: float
    band_rms_g: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class IterativeInversionParams:
    """Configuration parameters for iterative PSD inversion.

    Use `meta["diagnostics"]` returned by inversion functions to track convergence.
    """

    iters: int = 30
    gamma: float = 0.8
    gain_min: float = 0.2
    gain_max: float = 5.0
    alpha_sharpness: float = 1.0

    floor: float = 1e-30

    smooth_enabled: bool = True
    smooth_window_bins: int = 11
    smooth_every_n_iters: int = 1

    # Prior blending against seed PSD (log-domain)
    prior_blend: float = 0.0
    prior_power: float = 1.0

    # Edge anchoring against seed PSD (low/high frequency edges)
    edge_anchor_hz: float = 0.0
    edge_anchor_blend: float = 0.0

    # Tail/low caps
    tail_cap_start_hz: float = 0.0
    tail_cap_ratio: float = 0.0
    low_cap_ratio: float = 0.0

    # Optional post-smooth + light refine
    post_smooth_window_bins: int = 0
    post_smooth_blend: float = 1.0
    post_refine_iters: int = 0
    post_refine_gamma: float = 0.5
    post_refine_min: float = 0.7
    post_refine_max: float = 2.2
