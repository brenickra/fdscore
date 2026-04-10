from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .types import SNParams, SDOFParams, ERSResult, FDSResult, SineDwellSegment
from .grid import build_frequency_grid
from .sdof_transfer import build_transfer_psd
from .fds_ops import sum_fds
from .ers_ops import envelope_ers
from .validate import (
    ValidationError,
    validate_sn,
    validate_sdof,
    resolve_p_scale,
    compat_dict,
    ers_compat_dict,
)

PeakMode = Literal["abs"]
SweepSpacing = Literal["linear", "log"]


def _validate_peak_mode(peak_mode: str) -> str:
    if peak_mode != "abs":
        raise ValidationError("peak_mode must be 'abs'.")
    return str(peak_mode)


def _validate_input_motion(input_motion: str) -> str:
    if input_motion not in ("acc", "vel", "disp"):
        raise ValidationError("input_motion must be one of: 'acc', 'vel', 'disp'.")
    return str(input_motion)


def _validate_sweep_spacing(spacing: str) -> str:
    if spacing not in ("linear", "log"):
        raise ValidationError("spacing must be one of: 'linear', 'log'.")
    return str(spacing)


def _validate_harmonic_scalar(*, name: str, value: float, positive: bool = True) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite.")
    if positive and value <= 0.0:
        raise ValidationError(f"{name} must be > 0.")
    if not positive and value < 0.0:
        raise ValidationError(f"{name} must be >= 0.")
    return value


def _base_acceleration_amplitude(*, freq_hz: float, amp: float, input_motion: str) -> float:
    omega = 2.0 * np.pi * float(freq_hz)
    if input_motion == "acc":
        return float(amp)
    if input_motion == "vel":
        return omega * float(amp)
    if input_motion == "disp":
        return (omega * omega) * float(amp)
    raise ValidationError("input_motion must be one of: 'acc', 'vel', 'disp'.")


def _response_amplitude_sine(
    *,
    freq_hz: float,
    amp: float,
    sdof: SDOFParams,
    input_motion: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    validate_sdof(sdof)
    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")
    freq_hz = _validate_harmonic_scalar(name="freq_hz", value=freq_hz, positive=True)
    amp = _validate_harmonic_scalar(name="amp", value=amp, positive=False)
    input_motion = _validate_input_motion(input_motion)

    f0 = build_frequency_grid(sdof)
    zeta = 1.0 / (2.0 * float(sdof.q))
    a_base = _base_acceleration_amplitude(freq_hz=freq_hz, amp=amp, input_motion=input_motion)
    H = build_transfer_psd(
        f_psd_hz=np.asarray([freq_hz], dtype=float),
        f0_hz=f0,
        zeta=zeta,
        metric=sdof.metric,
    )
    response = np.abs(np.asarray(H[:, 0], dtype=np.complex128)) * float(a_base)
    return f0, np.asarray(response, dtype=float), float(a_base)


def _build_sine_sweep_segments(
    *,
    f_start_hz: float,
    f_stop_hz: float,
    amp: float,
    duration_s: float,
    input_motion: str,
    spacing: str,
    n_steps: int,
) -> list[SineDwellSegment]:
    f_start_hz = _validate_harmonic_scalar(name="f_start_hz", value=f_start_hz, positive=True)
    f_stop_hz = _validate_harmonic_scalar(name="f_stop_hz", value=f_stop_hz, positive=True)
    amp = _validate_harmonic_scalar(name="amp", value=amp, positive=False)
    duration_s = _validate_harmonic_scalar(name="duration_s", value=duration_s, positive=True)
    input_motion = _validate_input_motion(input_motion)
    spacing = _validate_sweep_spacing(spacing)

    if f_stop_hz <= f_start_hz:
        raise ValidationError("f_stop_hz must be > f_start_hz.")
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValidationError("n_steps must be an integer > 0.")

    if spacing == "linear":
        edges = np.linspace(f_start_hz, f_stop_hz, n_steps + 1, dtype=float)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        edges = np.geomspace(f_start_hz, f_stop_hz, n_steps + 1, dtype=float)
        centers = np.sqrt(edges[:-1] * edges[1:])

    dt_seg = float(duration_s) / float(n_steps)
    return [
        SineDwellSegment(
            freq_hz=float(fc),
            amp=float(amp),
            duration_s=float(dt_seg),
            input_motion=input_motion,
            label=f"sweep_step_{i}",
        )
        for i, fc in enumerate(centers)
    ]


def compute_ers_sine(
    *,
    freq_hz: float,
    amp: float,
    sdof: SDOFParams,
    input_motion: Literal["acc", "vel", "disp"] = "acc",
    peak_mode: PeakMode = "abs",
) -> ERSResult:
    r"""Compute deterministic ERS for a single harmonic base excitation.

    The returned spectrum is the response-amplitude spectrum of the selected
    SDOF metric. For a pure sinusoid, the absolute peak response is equal to
    the response amplitude, so the ERS is obtained directly from the harmonic
    transfer magnitude.

    Parameters
    ----------
    freq_hz:
        Excitation frequency of the sinusoid in Hz.
    amp:
        Input amplitude expressed in the units implied by ``input_motion``.
    sdof:
        Oscillator-grid definition and response metric.
    input_motion:
        Type of the specified harmonic amplitude. The routine converts the
        input to equivalent base-acceleration amplitude through

        .. math::

           a_{base,amp} =
           \begin{cases}
           amp, & \text{if input\_motion = "acc"} \\
           \omega \, amp, & \text{if input\_motion = "vel"} \\
           \omega^2 \, amp, & \text{if input\_motion = "disp"}
           \end{cases}

        where :math:`\omega = 2 \pi f`.
    peak_mode:
        Peak convention for the ERS. Only ``"abs"`` is currently supported.

    Returns
    -------
    ERSResult
        Response-amplitude spectrum on the oscillator grid.

    Notes
    -----
    The result is deterministic and contains no cycle counting or statistical
    assumptions. It is obtained from the magnitude of the linear SDOF transfer
    function at the single forcing frequency.
    """
    peak_mode = _validate_peak_mode(peak_mode)
    f0, response, a_base = _response_amplitude_sine(
        freq_hz=freq_hz,
        amp=amp,
        sdof=sdof,
        input_motion=input_motion,
    )
    zeta = 1.0 / (2.0 * float(sdof.q))
    meta = {
        "compat": ers_compat_dict(metric=sdof.metric, q=sdof.q, peak_mode=peak_mode, engine="deterministic_sine"),
        "metric": sdof.metric,
        "q": float(sdof.q),
        "zeta": float(zeta),
        "peak_mode": peak_mode,
        "provenance": {
            "source": "compute_ers_sine",
            "freq_hz": float(freq_hz),
            "amp": float(amp),
            "input_motion": str(input_motion),
            "base_acc_amp": float(a_base),
        },
    }
    return ERSResult(f=f0, response=response, meta=meta)


def compute_fds_sine(
    *,
    freq_hz: float,
    amp: float,
    duration_s: float,
    sn: SNParams,
    sdof: SDOFParams,
    input_motion: Literal["acc", "vel", "disp"] = "acc",
    p_scale: float | None = None,
) -> FDSResult:
    r"""Compute deterministic FDS for a single harmonic base excitation.

    This routine evaluates the damage caused by a sinusoidal base input of
    frequency ``freq_hz`` acting for ``duration_s`` seconds. Each oscillator is
    driven at the same forcing frequency and its steady-state response
    amplitude is converted directly to Miner damage.

    Parameters
    ----------
    freq_hz : float
        Excitation frequency of the sinusoid in Hz.
    amp : float
        Input amplitude expressed in the units implied by ``input_motion``.
    duration_s : float
        Exposure duration in seconds.
    sn : SNParams
        S-N curve definition used for Miner damage accumulation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    input_motion : str
        Optional input-motion type. Accepted values are ``"acc"``, ``"vel"``,
        and ``"disp"``. Internally the routine
        converts velocity and displacement amplitudes to equivalent
        base-acceleration amplitude using :math:`\omega amp` and
        :math:`\omega^2 amp`, respectively.
    p_scale : float or None
        Optional response scale factor applied before damage evaluation.

    Returns
    -------
    FDSResult
        Deterministic damage spectrum on the oscillator grid.

    Notes
    -----
    For a single harmonic input, the damage model reduces to

    .. math::

       D(f_0) = N_{cycles}
       \frac{\left(p_{scale} S(f_0)\right)^k}{C}

    where :math:`N_{cycles} = f \, T`, :math:`S(f_0)` is the response load
    amplitude derived from the chosen metric, and :math:`C` is the S-N
    intercept. When ``sn.amplitude_from_range`` is ``False``, the full range
    ``2 S(f_0)`` is used instead of alternating amplitude.

    References
    ----------
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    """
    validate_sn(sn)
    duration_s = _validate_harmonic_scalar(name="duration_s", value=duration_s, positive=True)
    p_scale_resolved = resolve_p_scale(p_scale=p_scale, sn=sn)

    f0, resp_amp, a_base = _response_amplitude_sine(
        freq_hz=freq_hz,
        amp=amp,
        sdof=sdof,
        input_motion=input_motion,
    )

    k = float(sn.slope_k)
    c = float(sn.C())
    n_cycles = float(freq_hz) * float(duration_s)
    load = resp_amp if bool(sn.amplitude_from_range) else (2.0 * resp_amp)
    damage = n_cycles * np.power(float(p_scale_resolved) * load, k) / c

    meta = {
        "compat": compat_dict(sn=sn, metric=sdof.metric, q=sdof.q, p_scale=p_scale_resolved, engine="deterministic_sine"),
        "provenance": {
            "source": "compute_fds_sine",
            "freq_hz": float(freq_hz),
            "amp": float(amp),
            "duration_s": float(duration_s),
            "input_motion": str(input_motion),
            "base_acc_amp": float(a_base),
            "n_cycles": float(n_cycles),
        },
    }
    return FDSResult(f=f0, damage=np.asarray(damage, dtype=float), meta=meta)


def compute_ers_dwell_profile(
    segments: Sequence[SineDwellSegment],
    *,
    sdof: SDOFParams,
    peak_mode: PeakMode = "abs",
) -> ERSResult:
    r"""Compute mission-level ERS as the envelope of harmonic dwell segments.

    Parameters
    ----------
    segments:
        Sequence of deterministic harmonic dwell segments.
    sdof:
        Oscillator-grid definition and response metric.
    peak_mode:
        Peak convention for the ERS. Only ``"abs"`` is currently supported.

    Returns
    -------
    ERSResult
        Pointwise envelope of the ERS results produced by the individual
        segments.

    Notes
    -----
    Each segment is evaluated independently with :func:`compute_ers_sine`, and
    the final result is the pointwise maximum across segments on a compatible
    oscillator grid.
    """
    if len(segments) == 0:
        raise ValidationError("segments must not be empty.")
    results = [
        compute_ers_sine(
            freq_hz=seg.freq_hz,
            amp=seg.amp,
            sdof=sdof,
            input_motion=seg.input_motion,
            peak_mode=peak_mode,
        )
        for seg in segments
    ]
    return envelope_ers(results)


def compute_fds_dwell_profile(
    segments: Sequence[SineDwellSegment],
    *,
    sn: SNParams,
    sdof: SDOFParams,
    p_scale: float | None = None,
) -> FDSResult:
    r"""Compute mission-level FDS as the sum of harmonic dwell-segment damage.

    Parameters
    ----------
    segments : collections.abc.Sequence
        Sequence of :class:`fdscore.types.SineDwellSegment` dwell segments.
    sn : SNParams
        S-N curve definition used for Miner damage accumulation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    p_scale : float or None
        Optional response scale factor applied before damage evaluation.

    Returns
    -------
    FDSResult
        Damage spectrum equal to the sum of the per-segment deterministic FDS
        results.

    Notes
    -----
    The function assumes linear cumulative damage and therefore sums the
    independent damage spectra computed for each dwell segment.

    References
    ----------
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    """
    if len(segments) == 0:
        raise ValidationError("segments must not be empty.")
    results = [
        compute_fds_sine(
            freq_hz=seg.freq_hz,
            amp=seg.amp,
            duration_s=seg.duration_s,
            sn=sn,
            sdof=sdof,
            input_motion=seg.input_motion,
            p_scale=p_scale,
        )
        for seg in segments
    ]
    return sum_fds(results)


def compute_ers_sine_sweep(
    *,
    f_start_hz: float,
    f_stop_hz: float,
    amp: float,
    duration_s: float,
    sdof: SDOFParams,
    input_motion: Literal["acc", "vel", "disp"] = "acc",
    peak_mode: PeakMode = "abs",
    spacing: SweepSpacing = "log",
    n_steps: int = 200,
) -> ERSResult:
    r"""Approximate deterministic ERS for a sine sweep via dwell discretization.

    Parameters
    ----------
    f_start_hz:
        Sweep start frequency in Hz.
    f_stop_hz:
        Sweep stop frequency in Hz.
    amp:
        Input amplitude expressed in the units implied by ``input_motion``.
    duration_s:
        Total sweep duration in seconds.
    sdof:
        Oscillator-grid definition and response metric.
    input_motion:
        Type of the specified harmonic amplitude.
    peak_mode:
        Peak convention for the ERS. Only ``"abs"`` is currently supported.
    spacing:
        Frequency spacing of the dwell discretization, either ``"linear"`` or
        ``"log"``.
    n_steps:
        Number of dwell segments used to approximate the sweep.

    Returns
    -------
    ERSResult
        Deterministic ERS approximation of the sweep.

    Notes
    -----
    The sweep is approximated as a sequence of constant-frequency dwell
    segments of equal duration. Convergence improves as ``n_steps`` increases,
    especially near sharp resonances and for logarithmic sweeps spanning large
    frequency ranges.
    """
    segments = _build_sine_sweep_segments(
        f_start_hz=f_start_hz,
        f_stop_hz=f_stop_hz,
        amp=amp,
        duration_s=duration_s,
        input_motion=input_motion,
        spacing=spacing,
        n_steps=n_steps,
    )
    ers = compute_ers_dwell_profile(segments, sdof=sdof, peak_mode=peak_mode)
    meta = dict(ers.meta)
    meta["provenance"] = {
        "source": "compute_ers_sine_sweep",
        "f_start_hz": float(f_start_hz),
        "f_stop_hz": float(f_stop_hz),
        "amp": float(amp),
        "duration_s": float(duration_s),
        "input_motion": str(input_motion),
        "spacing": str(spacing),
        "n_steps": int(n_steps),
        "discretization": "dwell_profile",
    }
    return ERSResult(f=np.asarray(ers.f, dtype=float), response=np.asarray(ers.response, dtype=float), meta=meta)


def compute_fds_sine_sweep(
    *,
    f_start_hz: float,
    f_stop_hz: float,
    amp: float,
    duration_s: float,
    sn: SNParams,
    sdof: SDOFParams,
    input_motion: Literal["acc", "vel", "disp"] = "acc",
    p_scale: float | None = None,
    spacing: SweepSpacing = "log",
    n_steps: int = 200,
) -> FDSResult:
    r"""Approximate deterministic FDS for a sine sweep via dwell discretization.

    Parameters
    ----------
    f_start_hz : float
        Sweep start frequency in Hz.
    f_stop_hz : float
        Sweep stop frequency in Hz.
    amp : float
        Input amplitude expressed in the units implied by ``input_motion``.
    duration_s : float
        Total sweep duration in seconds.
    sn : SNParams
        S-N curve definition used for Miner damage accumulation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    input_motion : str
        Optional input-motion type. Accepted values are ``"acc"``, ``"vel"``,
        and ``"disp"``.
    p_scale : float or None
        Optional response scale factor applied before damage evaluation.
    spacing : str
        Optional spacing mode for the dwell discretization. Accepted values
        are ``"linear"`` and ``"log"``.
    n_steps : int
        Number of dwell segments used to approximate the sweep.

    Returns
    -------
    FDSResult
        Deterministic FDS approximation of the sweep.

    Notes
    -----
    The result is obtained by discretizing the sweep into harmonic dwell
    segments and summing their individual damage spectra. Convergence with
    respect to ``n_steps`` should be checked whenever the sweep crosses narrow
    resonances or when the requested damage estimate is used quantitatively.

    References
    ----------
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    """
    segments = _build_sine_sweep_segments(
        f_start_hz=f_start_hz,
        f_stop_hz=f_stop_hz,
        amp=amp,
        duration_s=duration_s,
        input_motion=input_motion,
        spacing=spacing,
        n_steps=n_steps,
    )
    fds = compute_fds_dwell_profile(segments, sn=sn, sdof=sdof, p_scale=p_scale)
    meta = dict(fds.meta)
    meta["provenance"] = {
        "source": "compute_fds_sine_sweep",
        "f_start_hz": float(f_start_hz),
        "f_stop_hz": float(f_stop_hz),
        "amp": float(amp),
        "duration_s": float(duration_s),
        "input_motion": str(input_motion),
        "spacing": str(spacing),
        "n_steps": int(n_steps),
        "discretization": "dwell_profile",
    }
    return FDSResult(f=np.asarray(fds.f, dtype=float), damage=np.asarray(fds.damage, dtype=float), meta=meta)
