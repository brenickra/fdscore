"""Typed data containers used throughout the public ``fdscore`` API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

Metric = Literal["pv", "acc", "vel", "disp"]
InputMotion = Literal["acc", "vel", "disp"]


@dataclass(frozen=True, slots=True)
class SNParams:
    """S-N curve parameters used in Miner damage.

    Parameters
    ----------
    slope_k : float
        Fatigue slope exponent :math:`k`. Must be positive.
    ref_stress : float
        Reference stress :math:`S_{ref}` in user units, for example MPa.
    ref_cycles : float
        Reference cycle count :math:`N_{ref}`.
    amplitude_from_range : bool
        If ``True``, interpret the rainflow range as twice the alternating
        amplitude, so the damage-driving amplitude is ``range / 2``. This
        convention must remain consistent across FDS computation and
        inversion.

    Notes
    -----
    Two usage modes are common.

    Physical workflows provide ``ref_stress``, ``ref_cycles``, and an
    application-specific ``p_scale`` when absolute damage magnitude matters.

    Normalized workflows use ``SNParams.normalized(...)`` together with
    ``p_scale=1.0`` when only relative FDS shape and the equivalent
    inverted PSD matter.
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

        Parameters
        ----------
        slope_k : float
            Fatigue slope exponent :math:`k`.
        amplitude_from_range : bool, optional
            Rainflow convention indicating whether the damage-driving
            amplitude is computed from ``range / 2``.

        Returns
        -------
        SNParams
            S-N definition with ``ref_stress = 1`` and ``ref_cycles = 1``.

        Notes
        -----
        This normalized convention is useful when the workflow is focused on
        relative FDS shape and FDS-to-PSD inversion rather than absolute Miner
        damage magnitude.
        """
        return cls(
            slope_k=float(slope_k),
            ref_stress=1.0,
            ref_cycles=1.0,
            amplitude_from_range=amplitude_from_range,
        )

    def C(self) -> float:
        """Return C = N_ref * S_ref^k."""
        return float(self.ref_cycles * (self.ref_stress ** self.slope_k))


@dataclass(frozen=True, slots=True)
class SDOFParams:
    """SDOF oscillator-grid definition and response metric.

    Parameters
    ----------
    q : float
        Oscillator quality factor.
    metric : str, optional
        Response metric reported by the SDOF bank. Accepted values are
        ``"pv"``, ``"acc"``, ``"vel"``, and ``"disp"``.
    f : numpy.ndarray or None, optional
        Explicit frequency grid in Hz. When provided, it must be strictly
        increasing and strictly positive.
    fmin : float or None, optional
        Minimum oscillator frequency in Hz for an implicit linear grid.
    fmax : float or None, optional
        Maximum oscillator frequency in Hz for an implicit linear grid.
    df : float or None, optional
        Frequency increment in Hz for an implicit linear grid.

    Notes
    -----
    The grid can be defined either by an explicit vector ``f`` or by the
    linear-grid tuple ``(fmin, fmax, df)``, but not both.
    """
    q: float
    metric: Metric = "pv"
    f: Optional[np.ndarray] = None
    fmin: Optional[float] = None
    fmax: Optional[float] = None
    df: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PSDParams:
    """PSD-estimation settings used by spectral workflows.

    Parameters
    ----------
    method : {"welch"}, optional
        PSD estimation method. The current implementation supports only Welch.
    window : str, optional
        Window passed to the Welch estimator.
    nperseg : int or None, optional
        Segment length passed to the Welch estimator.
    noverlap : int or None, optional
        Segment overlap passed to the Welch estimator.
    detrend : {"constant", "linear", "none"}, optional
        Detrending mode used during PSD estimation.
    scaling : {"density"}, optional
        PSD scaling convention.
    onesided : bool, optional
        Whether a one-sided PSD is requested.
    fmin : float or None, optional
        Optional lower cropping limit in Hz.
    fmax : float or None, optional
        Optional upper cropping limit in Hz.
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
    """Fatigue Damage Spectrum result container.

    Parameters
    ----------
    f : numpy.ndarray
        Oscillator frequency grid in Hz.
    damage : numpy.ndarray
        Miner-damage values evaluated on ``f``.
    meta : dict, optional
        Auxiliary metadata such as compatibility signatures and provenance.
    """
    f: np.ndarray
    damage: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ERSResult:
    """Extreme-response spectrum result container.

    Parameters
    ----------
    f : numpy.ndarray
        Oscillator frequency grid in Hz.
    response : numpy.ndarray
        Response-spectrum values evaluated on ``f``.
    meta : dict, optional
        Auxiliary metadata such as compatibility signatures and provenance.
    """
    f: np.ndarray
    response: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShockSpectrumPair:
    """Positive and negative shock-spectrum pair.

    Parameters
    ----------
    neg : ERSResult
        Negative-sided response spectrum.
    pos : ERSResult
        Positive-sided response spectrum.
    meta : dict, optional
        Auxiliary metadata associated with the pair.
    """
    neg: ERSResult
    pos: ERSResult
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShockEvent:
    """Single detected shock event in a one-dimensional time history.

    Parameters
    ----------
    peak_index : int
        Sample index of the event peak.
    start_index : int
        Start sample index of the extracted event window.
    stop_index : int
        Stop sample index of the extracted event window.
    peak_time_s : float
        Peak time in seconds.
    start_time_s : float
        Start time in seconds.
    stop_time_s : float
        Stop time in seconds.
    peak_value : float
        Signed peak value.
    peak_abs : float
        Absolute peak magnitude.
    polarity : {"pos", "neg"}
        Peak polarity classification.
    """
    peak_index: int
    start_index: int
    stop_index: int
    peak_time_s: float
    start_time_s: float
    stop_time_s: float
    peak_value: float
    peak_abs: float
    polarity: Literal["pos", "neg"]


@dataclass(frozen=True, slots=True)
class ShockEventSet:
    """Detected shock events together with extraction metadata.

    Parameters
    ----------
    events : tuple of ShockEvent
        Detected events in chronological order.
    fs : float
        Sampling rate in Hz of the source signal.
    n_samples : int
        Total number of samples in the analyzed signal.
    meta : dict, optional
        Auxiliary metadata describing detector settings and provenance.
    """
    events: tuple[ShockEvent, ...]
    fs: float
    n_samples: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RollingERSResult:
    """Rolling response spectra stacked over multiple time windows.

    Parameters
    ----------
    f : numpy.ndarray
        Oscillator frequency grid in Hz.
    t_center_s : numpy.ndarray
        Time coordinate associated with each response row.
    response : numpy.ndarray
        Two-dimensional response matrix with one row per time window.
    meta : dict, optional
        Auxiliary metadata associated with the rolling result.
    """
    f: np.ndarray
    t_center_s: np.ndarray
    response: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HalfSinePulse:
    """Parameterized half-sine acceleration pulse.

    Parameters
    ----------
    amplitude : float
        Unsigned pulse amplitude.
    duration_s : float
        Pulse duration in seconds.
    polarity : {"pos", "neg"}, optional
        Pulse polarity.
    meta : dict, optional
        Auxiliary metadata associated with the pulse definition.
    """
    amplitude: float
    duration_s: float
    polarity: Literal["pos", "neg"] = "pos"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def signed_amplitude(self) -> float:
        return float(self.amplitude) if self.polarity == "pos" else -float(self.amplitude)


@dataclass(frozen=True, slots=True)
class FDSTimePlan:
    """Precomputed transfer plan for repeated time-domain FDS calls.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    n_samples : int
        Number of samples expected in reused time histories.
    f : numpy.ndarray
        Validated oscillator frequency grid in Hz.
    zeta : float
        Oscillator damping ratio implied by the source ``SDOFParams``.
    metric : str
        Response metric encoded in the plan. Accepted values are
        ``"pv"``, ``"acc"``, ``"vel"``, and ``"disp"``.
    H : numpy.ndarray
        Complex transfer matrix used during FFT-domain reconstruction.

    Notes
    -----
    A plan stores the frequency grid and transfer matrix for a fixed
    ``(fs, n_samples, sdof)`` configuration and can be reused across
    channels and signals with the same sampling setup.

    The stored transfer matrix ``H`` is materialized as a ``complex128`` array
    with shape ``(len(f), n_fft_bins)``. This trades memory for speed by
    avoiding transfer rebuilds during repeated ``compute_fds_time(...)`` calls.
    """
    fs: float
    n_samples: int
    f: np.ndarray
    zeta: float
    metric: Metric
    H: np.ndarray


@dataclass(frozen=True, slots=True)
class PSDResult:
    """Power Spectral Density result container.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency grid in Hz.
    psd : numpy.ndarray
        One-sided PSD values evaluated on ``f``.
    meta : dict, optional
        Auxiliary metadata such as provenance and reconstruction diagnostics.
    """
    f: np.ndarray
    psd: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PSDMetricsResult:
    """Summary metrics computed from an acceleration PSD.

    `meta` stores auxiliary details such as band coverage and peak-statistics
    diagnostics used to interpret the scalar outputs.
    """
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


@dataclass(frozen=True, slots=True)
class SineDwellSegment:
    """Single deterministic harmonic dwell segment.

    Parameters
    ----------
    freq_hz : float
        Harmonic excitation frequency in Hz.
    amp : float
        Input amplitude expressed in the units implied by ``input_motion``.
    duration_s : float
        Dwell duration in seconds.
    input_motion : str, optional
        Type of the prescribed harmonic input quantity. Accepted values
        are ``"acc"``, ``"vel"``, and ``"disp"``.
    label : str or None, optional
        Optional user-facing identifier for the segment.
    """

    freq_hz: float
    amp: float
    duration_s: float
    input_motion: InputMotion = "acc"
    label: Optional[str] = None


@dataclass(frozen=True, slots=True)
class IterativeInversionParams:
    """Configuration parameters for iterative PSD inversion.

    Parameters
    ----------
    iters : int, optional
        Number of main-loop iterations.
    gamma : float, optional
        Multiplicative update exponent.
    gain_min : float, optional
        Lower clip applied to oscillator-wise gains.
    gain_max : float, optional
        Upper clip applied to oscillator-wise gains.
    alpha_sharpness : float, optional
        Exponent used to sharpen or soften the influence matrix.
    floor : float, optional
        Minimum PSD floor maintained during iteration.
    smooth_enabled : bool, optional
        Whether periodic smoothing is enabled during the main loop.
    smooth_window_bins : int, optional
        Smoothing-window width in PSD bins.
    smooth_every_n_iters : int, optional
        Smoothing cadence in iterations.
    prior_blend : float, optional
        Blend factor against the seed PSD in log space.
    prior_power : float, optional
        Sensitivity weighting exponent applied to prior blending.
    edge_anchor_hz : float, optional
        Frequency span used for edge anchoring.
    edge_anchor_blend : float, optional
        Blend factor used for low- and high-frequency edge anchoring.
    tail_cap_start_hz : float, optional
        Frequency above which the spectral tail cap becomes active.
    tail_cap_ratio : float, optional
        Maximum allowed high-frequency tail ratio relative to the seed PSD.
    low_cap_ratio : float, optional
        Maximum allowed low-frequency ratio relative to the seed PSD.
    post_smooth_window_bins : int, optional
        Smoothing-window width used in the optional post-smoothing stage.
    post_smooth_blend : float, optional
        Blend factor applied between the best PSD and the post-smoothed PSD.
    post_refine_iters : int, optional
        Number of optional refinement iterations after post-smoothing.
    post_refine_gamma : float, optional
        Update exponent used during the refinement stage.
    post_refine_min : float, optional
        Lower gain clip used during refinement.
    post_refine_max : float, optional
        Upper gain clip used during refinement.

    Notes
    -----
    Use ``meta["diagnostics"]`` returned by inversion functions to track
    convergence.

    Parameter usage is not fully symmetric across engines. The time-domain
    iterative engine currently ignores the spectral-only subset, while
    ``PSDResult.meta["param_usage"]`` records the per-engine interpretation
    explicitly.
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

