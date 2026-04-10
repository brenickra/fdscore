from __future__ import annotations

from math import gamma
import numpy as np

from .types import FDSResult, PSDResult
from .validate import ValidationError, parse_fds_compat

EPS = 1e-30


def compute_damage_to_dp_factor(*, p_scale: float, b: float, c_sn: float) -> float:
    r"""Return the proportionality factor K such that Damage = K * DP.

    Derivation
    ----------
    Henderson & Piersol (1995) show that, under the assumptions of a
    lightly damped linear SDOF oscillator with narrowband Gaussian response,
    the peak stress distribution is approximately Rayleigh (Crandall & Mark,
    1963, p. 117). Integrating Miner's rule against that distribution yields
    (H&P 1995, Eq. 8):

    .. math::

       D = \left(\frac{\nu_0 T}{C}\right)
       \left(\sqrt{2} \, \sigma_S\right)^b
       \Gamma\left(1 + \frac{b}{2}\right)

    where :math:`\nu_0` is the mean zero-upcrossing rate, :math:`T` is
    duration, :math:`\sigma_S` is the stress standard deviation, and
    :math:`C = c` in the S-N curve :math:`N = c S^{-b}`.

    The stress standard deviation is related to the input acceleration PSD
    by (H&P 1995, Eq. 10, assuming :math:`\zeta < 0.1`):

    .. math::

       \sigma_S \approx p_{scale}
       \sqrt{\frac{G_{aa}(f_n)}{16 \pi f_n \zeta}}

    Substituting into the damage expression and isolating the
    environment-dependent part as

    .. math::

       DP(f_n) = f_n T
       \left(\frac{G_{aa}}{f_n \zeta}\right)^{b / 2}

    (H&P 1995, Eq. 12), the remaining factor is

    .. math::

       K = \frac{p_{scale}^b}{C_{SN}}
       \frac{\Gamma\left(1 + b / 2\right)}{(8 \pi)^{b / 2}}

    Here ``p_scale`` plays the role of the stress-velocity proportionality
    constant :math:`k` in H&P Eq. 9, which relates modal velocity to peak
    stress. That relationship, velocity as the stress-relevant quantity, is
    justified by Gaberson & Chalmers (1969) and Crandall (1962); see also
    the theoretical basis in H&P (1995), section "Estimates of Stress in
    Test Items".

    Inherited assumptions
    ---------------------
    - Single dominant resonant mode (SDOF behaviour).
    - Lightly damped response: :math:`\zeta < 0.1` (equivalently :math:`Q > 5`).
    - Input PSD approximately flat over the half-power bandwidth
      :math:`B_r \approx 2 \zeta f_n`.
    - Narrowband, stationary, Gaussian response leading to a Rayleigh peak
      distribution.

    Parameters
    ----------
    p_scale : float
        Stress-response scale factor, analogous to the proportionality
        constant :math:`k` in H&P 1995, Eq. 9. Must be greater than zero.
    b : float
        S-N curve slope exponent. Must be greater than zero.
    c_sn : float
        S-N curve intercept :math:`C = N_{ref} S_{ref}^b`. Must be greater
        than zero.

    Returns
    -------
    float
        Factor :math:`K > 0` such that :math:`Damage = K \, DP`.

    References
    ----------
    Henderson, G. R. & Piersol, A. G. (1995). "Fatigue Damage Related
        Descriptor for Random Vibration Test Environments." Sound and
        Vibration, October 1995, 20-24. Equations 8-12.
    Crandall, S. H. & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press, New York. pp. 113-117.
    Gaberson, H. A. & Chalmers, R. H. (1969). "Modal Velocity as a
        Criterion of Shock Severity." Shock and Vibration Bulletin,
        No. 40, Pt. 2, 31-49.
    Crandall, S. H. (1962). "Relationship between Stress and Velocity in
        Resonant Vibration." Journal of the Acoustical Society of America,
        34(12), 1960-1961.
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
    r"""Convert a Damage Potential (DP) spectrum to an acceleration PSD.

    This is the algebraic inverse of the Damage Potential definition
    (Henderson & Piersol, 1995, Eq. 12):

    .. math::

       DP(f_n) = f_n T
       \left[\frac{G_{aa}(f_n)}{f_n \zeta}\right]^{b / 2}

    Solving for :math:`G_{aa}` yields

    .. math::

       G_{aa}(f_n) = f_n \zeta
       \left[\frac{DP(f_n)}{f_n T}\right]^{2 / b}

    which is the equation implemented here.

    Parameters
    ----------
    f0_hz : numpy.ndarray
        Oscillator natural frequencies in Hz.
    dp_fds : numpy.ndarray
        Damage Potential values at each oscillator frequency.
    zeta : float
        Damping ratio :math:`\zeta = 1 / (2Q)`. The derivation assumes
        :math:`\zeta < 0.1`.
    b : float
        S-N curve slope exponent.
    test_duration_s : float
        Target test duration :math:`T` in seconds.

    Returns
    -------
    numpy.ndarray
        One-sided acceleration PSD :math:`G_{aa}` in input-units squared per
        hertz on the ``f0_hz`` grid.

    Notes
    -----
    The result is the acceleration PSD that, when applied to a base-excited
    SDOF oscillator at each natural frequency :math:`f_n` with damping ratio
    :math:`\zeta`, produces the same DP, and therefore the same fatigue
    damage, as the input DP spectrum over duration :math:`T`.

    References
    ----------
    Henderson, G. R. & Piersol, A. G. (1995). "Fatigue Damage Related
        Descriptor for Random Vibration Test Environments." Sound and
        Vibration, October 1995. Equation 12 (inverted).
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
    r"""Convert an acceleration PSD to a Damage Potential (DP) spectrum.

    Direct application of Henderson & Piersol (1995), Eq. 12:

    .. math::

       DP(f_n) = f_n T
       \left[\frac{G_{aa}(f_n)}{f_n \zeta}\right]^{b / 2}

    This is the forward direction of the closed-form relationship and is
    used primarily for round-trip reconstruction checks after inversion.
    It is not equivalent to a time-domain or spectral FDS computation:
    it assumes the SDOF response is narrowband Gaussian and that peaks
    follow a Rayleigh distribution, which is only valid for lightly damped
    oscillators driven by a stationary random process with a flat PSD over
    the resonance bandwidth.

    Parameters
    ----------
    f0_hz : numpy.ndarray
        Oscillator natural frequencies in Hz.
    psd : numpy.ndarray
        One-sided acceleration PSD :math:`G_{aa}` in units squared per hertz.
    zeta : float
        Damping ratio :math:`\zeta = 1 / (2Q)`. The derivation assumes
        :math:`\zeta < 0.1`.
    b : float
        S-N curve slope exponent.
    test_duration_s : float
        Exposure duration :math:`T` in seconds.

    Returns
    -------
    numpy.ndarray
        Damage Potential :math:`DP(f_n)` on the ``f0_hz`` grid.

    References
    ----------
    Henderson, G. R. & Piersol, A. G. (1995). "Fatigue Damage Related
        Descriptor for Random Vibration Test Environments." Sound and
        Vibration, October 1995. Equation 12.
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
    r"""Invert an FDS to an equivalent acceleration PSD using the closed-form
    Henderson-Piersol method.

    Overview
    --------
    The inversion proceeds in two steps. First, Miner damage is converted to a
    Damage Potential spectrum using the proportionality factor ``K`` derived
    from the S-N parameters and ``p_scale``:

    .. math::

       DP(f) = \frac{Damage(f)}{K}

    Second, the closed-form inverse of H&P (1995), Eq. 12 is applied:

    .. math::

       G_{aa}(f) = f \zeta
       \left[\frac{DP(f)}{f T}\right]^{2 / b}

    The round-trip reconstruction error stored in
    ``meta["reconstruction"]["med_abs_log10"]`` is the median absolute value
    of

    .. math::

       \log_{10}\left(\frac{D_{recon}}{D_{target}}\right)

    Metric restriction
    ------------------
    This method is valid only for ``metric="pv"`` (pseudo-velocity).

    The physical justification follows from two independent lines of
    argument that converge on pseudo-velocity as the stress-relevant
    quantity:

    - Empirical / experimental: Gaberson & Chalmers (1969) established
      that modal velocity is the best single predictor of shock and
      vibration severity across a wide range of structures. Their work is
      explicitly cited by Henderson & Piersol (1995) as the basis for
      using velocity rather than acceleration or displacement to estimate
      stress.
    - Theoretical: Crandall (1962) derived analytically that, for a
      structure vibrating at resonance, peak stress is proportional to
      peak velocity, not peak acceleration or displacement. This result
      holds for geometrically simple structures (beams, plates under
      bending) and is the theoretical foundation cited by H&P (1995) in
      their Eq. 9.

    For a lightly damped SDOF oscillator (:math:`\zeta < 0.1`),
    pseudo-velocity

    .. math::

       PV = 2 \pi f_0 x_{rel}

    and relative velocity :math:`v_{rel}` are approximately equal at
    resonance, because the phase between displacement and velocity at the
    resonant peak makes

    .. math::

       |v_{rel}| \approx \omega_0 |x_{rel}|

    This equivalence justifies using ``metric="pv"`` as a numerically
    convenient proxy for relative velocity in the closed-form derivation.

    Using ``metric="acc"``, ``"vel"``, or ``"disp"`` would require a
    different transfer function at resonance and a different proportionality
    to stress, yielding a structurally different inversion equation that
    is not implemented here. Passing ``strict_metric=False`` suppresses the
    guard but does not make the derivation valid for those metrics.

    Global damage scaling cancellation
    ----------------------------------
    When ``fds`` was computed with compatible settings, the absolute damage
    scaling carried by ``p_scale``, ``ref_stress``, and ``ref_cycles``
    cancels exactly in the inversion: those parameters affect the magnitude
    of ``damage(f)`` and of ``K`` in equal proportion, leaving the inverted
    PSD unchanged. This invariance is verified by
    ``test_closed_form_psd_is_invariant_to_global_damage_scaling``.

    Requirements
    ------------
    - ``fds.meta["compat"]`` must exist and contain ``metric``, ``q``,
      ``p_scale``, and ``sn``.
    - ``metric`` must be ``"pv"`` unless ``strict_metric=False``.
    - ``test_duration_s`` must be the intended test duration, not the
      original signal duration.

    Parameters
    ----------
    fds : FDSResult
        Target FDS result. It must carry ``meta["compat"]`` produced by any
        ``compute_fds_*`` function in this library.
    test_duration_s : float
        Duration of the equivalent test :math:`T` in seconds. This is the
        denominator in the DP-to-PSD step and directly controls the amplitude
        of the inverted PSD.
    strict_metric : bool
        If ``True`` (default), raise ``ValidationError`` when
        ``fds.meta["compat"]["metric"] != "pv"``.

    Returns
    -------
    PSDResult
        Equivalent acceleration PSD on the same frequency grid as ``fds.f``.
        ``meta["reconstruction"]`` contains the round-trip log10 error used
        for quality assessment.

    References
    ----------
    Henderson, G. R. & Piersol, A. G. (1995). "Fatigue Damage Related
        Descriptor for Random Vibration Test Environments." Sound and
        Vibration, October 1995, 20-24. Equations 11-12.
    Gaberson, H. A. & Chalmers, R. H. (1969). "Modal Velocity as a
        Criterion of Shock Severity." Shock and Vibration Bulletin,
        No. 40, Pt. 2, 31-49.
    Crandall, S. H. (1962). "Relationship between Stress and Velocity in
        Resonant Vibration." Journal of the Acoustical Society of America,
        34(12), 1960-1961.
    Crandall, S. H. & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press, New York. pp. 113-117.
    """
    if not np.isfinite(test_duration_s) or float(test_duration_s) <= 0:
        raise ValidationError("test_duration_s must be finite and > 0.")
    t_test = float(test_duration_s)

    compat = parse_fds_compat((fds.meta or {}).get("compat", {}))
    if strict_metric and compat.metric != "pv":
        raise ValidationError("Closed-form inversion supports only metric='pv'.")

    b = float(compat.sn.slope_k)
    ref_stress = float(compat.sn.ref_stress)
    ref_cycles = float(compat.sn.ref_cycles)
    q = float(compat.q)
    p_scale = float(compat.p_scale)
    metric = compat.metric

    if b <= 0 or ref_stress <= 0 or ref_cycles <= 0:
        raise ValidationError("Invalid S-N parameters (must be > 0).")
    if not np.isfinite(q) or q <= 0:
        raise ValidationError("Invalid q in FDS compat metadata (must be finite and > 0).")

    zeta = 1.0 / (2.0 * q)
    c_sn = ref_cycles * (ref_stress ** b)

    damage_to_dp = compute_damage_to_dp_factor(p_scale=p_scale, b=b, c_sn=c_sn)
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
            "q": q,
            "zeta": float(zeta),
            "b": float(b),
            "p_scale": p_scale,
            "c_sn": float(c_sn),
        },
        "reconstruction": {
            "med_abs_log10": med_abs_log10,
        },
        "damage_to_dp_factor": float(damage_to_dp),
        "provenance": {"source": "invert_fds_closed_form", "equation": "G=f*zeta*(DP/(f*T))^(2/b)"},
    }
    return PSDResult(f=f, psd=psd, meta=meta)
