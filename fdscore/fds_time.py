from __future__ import annotations

import numpy as np

from .types import SNParams, SDOFParams, FDSResult, FDSTimePlan
from .grid import build_frequency_grid
from ._time_plan import validate_time_plan_compatibility
from .validate import (
    ValidationError,
    _finite_positive_float_or_raise,
    _validate_nyquist_with_info,
    compat_dict,
    resolve_p_scale,
    validate_nyquist,
    validate_sdof,
    validate_sn,
)
from .preprocess import preprocess_signal
from .sdof_transfer import build_transfer_matrix
from .rainflow_damage import miner_damage_from_matrix
from ._fds_incremental import fds_incremental


def _fds_from_signal_fft(
    y: np.ndarray,
    *,
    fs: float,
    f0: np.ndarray,
    zeta: float,
    metric: str,
    k: float,
    c: float,
    p_scale: float,
    batch_size: int = 64,
    amplitude_from_range: bool = True,
    H: np.ndarray | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = int(y.size)
    yf = np.fft.rfft(y)

    if H is None:
        H = build_transfer_matrix(fs=float(fs), n=n, f0_hz=f0, zeta=float(zeta), metric=metric)  # type: ignore[arg-type]

    out = np.zeros(H.shape[0], dtype=float)
    bs = max(1, int(batch_size))

    for i0 in range(0, H.shape[0], bs):
        i1 = min(i0 + bs, H.shape[0])
        resp = np.fft.irfft(H[i0:i1] * yf[None, :], n=n, axis=1)
        resp *= float(p_scale)

        dmg_batch = miner_damage_from_matrix(
            resp,
            k=float(k),
            c=float(c),
            amplitude_from_range=bool(amplitude_from_range),
        )
        out[i0:i1] = np.asarray(dmg_batch, dtype=float)

    return out


def prepare_fds_time_plan(
    *,
    fs: float,
    n_samples: int,
    sdof: SDOFParams,
    strict_nyquist: bool = True,
) -> FDSTimePlan:
    r"""Precompute the FFT-domain transfer data for repeated FDS evaluations.

    A time-domain FDS call repeatedly uses the same oscillator grid, damping,
    and FFT-bin transfer matrix when the sampling configuration is fixed.
    ``FDSTimePlan`` stores that reusable state so that repeated calls can skip
    transfer-matrix reconstruction.

    Parameters
    ----------
    fs:
        Sampling rate in Hz.
    n_samples:
        Number of samples in the time histories that will reuse this plan.
    sdof:
        Oscillator-grid definition and chosen response metric.
    strict_nyquist:
        If ``True``, frequencies above Nyquist raise ``ValidationError``. If
        ``False``, the frequency grid is truncated to the valid Nyquist range.

    Returns
    -------
    FDSTimePlan
        Reusable transfer plan containing the validated oscillator grid, the
        implied damping ratio, and the complex FFT-domain transfer matrix.

    Notes
    -----
    The stored matrix ``H`` is materialized as ``complex128`` with shape
    ``(len(f0), n_fft_bins)``. Memory therefore scales approximately as

    .. math::

       len(f_0) \times n_{fft\_bins} \times 16 \text{ bytes}

    so plans trade memory for speed. For example, 400 oscillators and a 4 s
    signal sampled at 1 kHz require roughly 12 MB for the matrix alone.
    """
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")
    fs = _finite_positive_float_or_raise(fs, field="fs")
    if not isinstance(n_samples, (int, np.integer)) or int(n_samples) < 4:
        raise ValidationError("n_samples must be an integer >= 4.")

    f0 = build_frequency_grid(sdof)
    f0 = validate_nyquist(f0, fs=fs, strict=bool(strict_nyquist))
    zeta = 1.0 / (2.0 * float(sdof.q))

    H = build_transfer_matrix(
        fs=fs,
        n=int(n_samples),
        f0_hz=f0,
        zeta=float(zeta),
        metric=sdof.metric,
    )
    return FDSTimePlan(
        fs=fs,
        n_samples=int(n_samples),
        f=np.asarray(f0, dtype=float),
        zeta=float(zeta),
        metric=sdof.metric,
        H=np.asarray(H),
    )


def compute_fds_time(
    x: np.ndarray,
    fs: float,
    sn: SNParams,
    sdof: SDOFParams,
    *,
    p_scale: float | None = None,
    detrend: str = "linear",
    strict_nyquist: bool = True,
    batch_size: int = 64,
    plan: FDSTimePlan | None = None,
    engine: str = "incremental",
    zoh_r_max: float = 0.2,
) -> FDSResult:
    r"""Compute a time-domain Fatigue Damage Spectrum from an input history.

    The returned spectrum contains Miner damage evaluated independently for
    each SDOF oscillator in the grid defined by ``sdof``. The result also
    carries a compatibility signature in ``meta["compat"]`` so that downstream
    operations, especially inversion, can verify that the same fatigue
    conventions were used.

    Pipeline
    --------
    The computation follows this sequence for a base-acceleration time history
    ``x``:

    1. Optionally preprocess the signal with ``preprocess_signal(...)`` using
       the requested ``detrend`` mode.
    2. Validate the oscillator grid against Nyquist and derive the shared
       damping ratio and S-N parameters.
    3. Evaluate the oscillator bank with the selected internal engine:

       * ``engine="incremental"`` integrates each SDOF oscillator
         sample-by-sample using exact ZOH state-transition matrices. For
         oscillators close to Nyquist, the input is adaptively upsampled to
         control ZOH attenuation error.
       * ``engine="fft"`` builds or reuses the FFT-domain transfer matrix,
         applies it to ``rfft(x)``, and reconstructs each oscillator response
         with ``irfft`` in batches.

    4. Apply ASTM-style rainflow counting and Miner's linear damage rule to
       each oscillator response or response reversal stream.
    5. Return ``FDSResult`` together with compatibility and provenance
       metadata describing the selected engine and preprocessing choices.

    Parameters
    ----------
    x : numpy.ndarray
        One-dimensional base-acceleration time history.
    fs : float
        Sampling rate in Hz.
    sn : SNParams
        S-N curve definition used for Miner damage accumulation.
    sdof : SDOFParams
        Oscillator-grid definition and response metric.
    p_scale : float or None
        Optional scale factor applied to each oscillator response before
        rainflow counting. In physical workflows this represents the
        stress-response proportionality used to convert response to the
        fatigue-driving quantity.
    detrend : str
        Optional preprocessing mode passed to ``preprocess_signal(...)``.
        Supported values are ``"linear"``, ``"mean"``, and ``"none"``.
    strict_nyquist : bool
        If ``True``, oscillator frequencies above Nyquist raise
        ``ValidationError``. If ``False``, the grid is truncated to the valid
        Nyquist range and the truncation is recorded in the result metadata.
    batch_size : int
        Number of oscillators processed per inverse FFT batch.
        Only used when ``engine="fft"``.
    plan : FDSTimePlan or None
        Optional precomputed transfer plan created by
        :func:`prepare_fds_time_plan`.
        Only used when ``engine="fft"``.
    engine : {"incremental", "fft"}
        Internal computation engine.

        ``"incremental"`` (default)
            Sample-by-sample SDOF integration via exact ZOH state-transition
            matrices.  Rainflow counting is performed online during integration
            so the full ``(n_osc, n_samples)`` response matrix is never
            materialised.  Provides super-linear speedup over ``"fft"`` for
            long signals and low memory overhead regardless of signal length.

        ``"fft"``
            Original FFT-based engine.  Applies the continuous SDOF frequency
            response function to the signal spectrum and reconstructs each
            oscillator response with ``irfft``.  Use this engine when exact
            bit-for-bit reproducibility with pre-existing results is required.

        .. note::
            The two engines use different SDOF discretisation schemes (ZOH vs.
            continuous FRF on discrete FFT bins) and will therefore produce
            slightly different damage values, particularly for oscillators above
            roughly ``0.3 * fs / 2``.  For the typical configuration of
            ``fs = 1000 Hz`` and ``fmax = 400 Hz`` the discrepancy in Miner
            damage is below 5 % across the full grid and below 1 % below
            150 Hz.

    zoh_r_max : float
        Maximum tolerated ``f0 / Nyquist_effective`` ratio for the
        ``"incremental"`` engine.  Controls the adaptive upsampling that
        corrects the ZOH attenuation error for high-frequency oscillators.
        Ignored when ``engine="fft"``.

        Smaller values yield higher accuracy at the cost of larger upsample
        factors for oscillators near the top of the frequency grid.  Larger
        values prioritise throughput.

        * ``0.30`` - max error about 0.5 dB, upsample up to 3x
        * ``0.20`` - max error about 0.1 dB, upsample up to 4x *(default)*
        * ``0.15`` - max error about 0.05 dB, upsample up to 6x

    Returns
    -------
    FDSResult
        Damage spectrum on the validated oscillator grid. ``meta["compat"]``
        records the fatigue and response conventions required by downstream
        operations.

    Notes
    -----
    The computation assumes a linear SDOF transfer from base acceleration to
    the selected metric and evaluates fatigue on the reconstructed response
    histories. For fixed ``slope_k``, the absolute damage level scales
    globally as

    .. math::

       \frac{p_{scale}^k}{N_{ref} S_{ref}^k}

    Therefore ``p_scale``, ``ref_stress``, and ``ref_cycles`` change the
    magnitude of ``damage(f)`` but not its relative shape. When only the
    spectral shape and a compatible FDS-to-PSD inversion matter, a normalized
    S-N definition together with ``p_scale=1.0`` is usually sufficient.

    The choice of ``detrend`` can materially affect low-frequency damage,
    especially for displacement- and pseudo-velocity-based metrics, because
    offsets and slow drifts are amplified by the low-frequency dynamics of the
    oscillator bank.

    If ``p_scale`` is omitted, ``p_scale=1.0`` is assumed only for normalized
    S-N parameters with ``ref_stress=1`` and ``ref_cycles=1``. Physical
    workflows must pass ``p_scale`` explicitly.

    References
    ----------
    ASTM E1049-85(2017). Standard Practices for Cycle Counting in Fatigue
        Analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    Crandall, S. H., & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press.
    """
    validate_sn(sn)
    validate_sdof(sdof)

    if sdof.metric not in ("pv", "disp", "vel", "acc"):
        raise ValidationError("sdof.metric must be one of: 'pv','disp','vel','acc'.")

    if engine not in ("incremental", "fft"):
        raise ValidationError("engine must be one of: 'incremental', 'fft'.")

    p_scale_resolved = resolve_p_scale(p_scale=p_scale, sn=sn)
    if detrend not in ("linear", "mean", "none"):
        raise ValidationError("detrend must be one of: 'linear', 'mean', 'none'.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError("batch_size must be an int > 0.")
    fs = _finite_positive_float_or_raise(fs, field="fs")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValidationError("x must be a 1D array with length >= 4.")
    if not np.all(np.isfinite(x)):
        raise ValidationError("x must contain only finite values.")

    f_requested = build_frequency_grid(sdof)
    f0, nyquist_info = _validate_nyquist_with_info(f_requested, fs=fs, strict=strict_nyquist)

    zeta = 1.0 / (2.0 * float(sdof.q))
    k = float(sn.slope_k)
    c = float(sn.ref_cycles) * (float(sn.ref_stress) ** k)

    y = preprocess_signal(x, mode=detrend)

    if engine == "incremental":
        damage = fds_incremental(
            y,
            fs=fs,
            f0=f0,
            zeta=float(zeta),
            metric=sdof.metric,
            k=float(k),
            c=float(c),
            p_scale=float(p_scale_resolved),
            amplitude_from_range=bool(sn.amplitude_from_range),
            zoh_r_max=float(zoh_r_max),
        )
        engine_tag = "time_rainflow_incremental_zoh_numba"
        provenance_extra: dict = {"zoh_r_max": float(zoh_r_max)}
    else:
        # engine == "fft"
        if plan is None:
            H = build_transfer_matrix(fs=fs, n=int(y.size), f0_hz=f0, zeta=float(zeta), metric=sdof.metric)  # type: ignore[arg-type]
        else:
            H = validate_time_plan_compatibility(
                plan=plan,
                fs=fs,
                n_samples=int(y.size),
                f0=f0,
                zeta=float(zeta),
                metric=sdof.metric,
            )
        damage = _fds_from_signal_fft(
            y,
            fs=fs,
            f0=f0,
            zeta=float(zeta),
            metric=sdof.metric,
            k=float(k),
            c=float(c),
            p_scale=float(p_scale_resolved),
            batch_size=int(batch_size),
            amplitude_from_range=bool(sn.amplitude_from_range),
            H=H,
        )
        engine_tag = "time_rainflow_fft_numba"
        provenance_extra = {
            "batch_size": int(batch_size),
            "transfer_plan": bool(plan is not None),
        }

    meta = {
        "compat": compat_dict(
            sn=sn,
            metric=sdof.metric,
            q=sdof.q,
            p_scale=p_scale_resolved,
            engine=engine_tag,
        ),
        "provenance": {
            "source": "compute_fds_time",
            "engine": engine,
            "detrend": detrend,
            **provenance_extra,
            **nyquist_info,
        },
    }
    return FDSResult(f=f0, damage=damage, meta=meta)
