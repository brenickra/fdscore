from __future__ import annotations

import numpy as np

from .types import Metric


def h_baseacc_to_x(freq_hz: np.ndarray, fn_hz: np.ndarray, zeta: float) -> np.ndarray:
    r"""Return the base-acceleration-to-relative-displacement receptance.

    For a linear SDOF oscillator under base acceleration input
    :math:`a_{base}(t)`, the relative displacement transfer function is

    .. math::

       H_x(\omega; \omega_0, \zeta) =
       -\frac{1}{(\omega_0^2 - \omega^2) + 2 \zeta \omega_0 j \omega}

    such that

    .. math::

       X_{rel}(\omega) = H_x(\omega) A_{base}(\omega)

    This receptance is the common starting point for all response metrics
    implemented in this module.

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Excitation-frequency vector in Hz. Broadcasting is supported.
    fn_hz : numpy.ndarray
        Natural-frequency vector of the oscillators in Hz. Broadcasting is
        supported against ``freq_hz``.
    zeta : float
        Damping ratio of the oscillator. Must be interpreted consistently with
        the ``Q`` values used elsewhere in the library through
        :math:`\zeta = 1 / (2Q)`.

    Returns
    -------
    numpy.ndarray
        Complex receptance from base acceleration to relative displacement.
        The returned shape is the NumPy broadcasted shape of ``freq_hz`` and
        ``fn_hz``.

    Notes
    -----
    The sign convention follows the usual base-excitation equation of motion
    written in relative coordinates. The returned quantity has units of
    displacement per acceleration.

    References
    ----------
    Crandall, S. H., & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press.
    """
    w = 2.0 * np.pi * np.asarray(freq_hz, dtype=float)
    w0 = 2.0 * np.pi * np.asarray(fn_hz, dtype=float)
    return -1.0 / ((w0 * w0 - w * w) + 2.0 * zeta * w0 * (1j * w))


def build_transfer_matrix(
    *,
    fs: float,
    n: int,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
) -> np.ndarray:
    r"""Build the FFT-domain transfer matrix for a chosen SDOF response metric.

    The input signal is a base-acceleration time history. This routine builds
    the complex transfer matrix used to map the FFT of that signal to the FFT
    of the oscillator response before reconstruction with ``numpy.fft.irfft``.

    Metric definition
    -----------------
    Let :math:`A_{base}(\omega)` be the FFT of the base acceleration and let
    :math:`H_x(\omega)` be the relative-displacement receptance returned by
    :func:`h_baseacc_to_x`. The implemented response metrics are

    .. math::

       X_{rel} = H_x A_{base}

    .. math::

       V_{rel} = j \omega X_{rel}

    .. math::

       A_{rel} = -\omega^2 X_{rel}

    .. math::

       A_{abs} = A_{base} + A_{rel}
               = \left(1 - \omega^2 H_x\right) A_{base}

    with the library mapping

    - ``metric="disp"`` -> relative displacement :math:`X_{rel}`
    - ``metric="vel"`` -> relative velocity :math:`V_{rel}`
    - ``metric="acc"`` -> absolute acceleration :math:`A_{abs}`
    - ``metric="pv"`` -> pseudo-velocity
      :math:`PV = \omega_0 X_{rel} = 2 \pi f_0 X_{rel}`

    Parameters
    ----------
    fs : float
        Sampling rate in Hz of the time history that will be transformed.
    n : int
        Number of time samples. This defines the FFT grid through
        ``numpy.fft.rfftfreq``.
    f0_hz : numpy.ndarray
        Oscillator natural frequencies in Hz.
    zeta : float
        Damping ratio of the oscillators.
    metric : str
        Response metric to build. Supported values are ``"disp"``, ``"vel"``,
        ``"acc"``, and ``"pv"``.

    Returns
    -------
    numpy.ndarray
        Complex transfer matrix with shape ``(len(f0_hz), n_fft_bins)``. Each
        row maps the base-acceleration spectrum to the selected metric for one
        oscillator frequency.

    Notes
    -----
    The distinction between ``"acc"`` and ``"vel"`` is physically important:
    ``"acc"`` is the absolute acceleration of the mass, while ``"vel"`` is the
    relative velocity across the spring-damper coordinates.

    ``"pv"`` is not the same quantity as relative velocity away from resonance.
    For lightly damped oscillators near resonance, however,
    :math:`PV = \omega_0 X_{rel}` is a useful approximation to peak relative
    velocity. That approximation underlies the closed-form inversion workflow
    implemented elsewhere in the library.

    References
    ----------
    Crandall, S. H. (1962). "Relationship between Stress and Velocity in
        Resonant Vibration." Journal of the Acoustical Society of America,
        34(12), 1960-1961.
    Crandall, S. H., & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press.
    Gaberson, H. A., & Chalmers, R. H. (1969). "Modal Velocity as a Criterion
        of Shock Severity." Shock and Vibration Bulletin, No. 40, Pt. 2,
        31-49.
    """
    f_fft = np.fft.rfftfreq(int(n), d=1.0 / float(fs))
    f0 = np.asarray(f0_hz, dtype=float)
    Hx = h_baseacc_to_x(f_fft[None, :], f0[:, None], float(zeta))

    w = 2.0 * np.pi * f_fft[None, :]
    if metric == "disp":
        H = Hx
    elif metric == "vel":
        H = (1j * w) * Hx
    elif metric == "acc":
        H = 1.0 + (-w * w) * Hx
    elif metric == "pv":
        H = (2.0 * np.pi * f0[:, None]) * Hx
    else:  # pragma: no cover
        raise ValueError(f"Unknown metric: {metric}")

    return H


def build_transfer_psd(
    *,
    f_psd_hz: np.ndarray,
    f0_hz: np.ndarray,
    zeta: float,
    metric: Metric,
) -> np.ndarray:
    r"""Build the frequency-response matrix used in PSD-domain workflows.

    This is the PSD-domain counterpart of :func:`build_transfer_matrix`. It
    returns the complex transfer function evaluated directly on a user-provided
    one-sided PSD frequency grid instead of the FFT bins implied by a time
    history.

    Metric definition
    -----------------
    The same metric mapping as :func:`build_transfer_matrix` is used here:
    relative displacement, relative velocity, absolute acceleration, or
    pseudo-velocity. If ``H`` denotes the returned matrix, the response PSD for
    each oscillator is obtained from

    .. math::

       P_{resp}(f; f_0) = \left| H(f; f_0) \right|^2 P_{base}(f)

    Parameters
    ----------
    f_psd_hz : numpy.ndarray
        One-sided input PSD frequency grid in Hz.
    f0_hz : numpy.ndarray
        Oscillator natural frequencies in Hz.
    zeta : float
        Damping ratio of the oscillators.
    metric : str
        Response metric to build. Supported values are ``"disp"``, ``"vel"``,
        ``"acc"``, and ``"pv"``.

    Returns
    -------
    numpy.ndarray
        Complex transfer matrix with shape ``(len(f0_hz), len(f_psd_hz))``.
        Each row corresponds to one oscillator and each column to one PSD
        frequency bin.

    Notes
    -----
    The returned transfer is intended for one-sided acceleration PSDs. As in
    the time-domain matrix, ``"acc"`` refers to absolute acceleration of the
    oscillator mass, not to relative acceleration.

    References
    ----------
    Crandall, S. H., & Mark, W. D. (1963). Random Vibrations in Mechanical
        Systems. Academic Press.
    """
    f_psd = np.asarray(f_psd_hz, dtype=float).reshape(-1)
    f0 = np.asarray(f0_hz, dtype=float).reshape(-1)

    Hx = h_baseacc_to_x(f_psd[None, :], f0[:, None], float(zeta))
    w = 2.0 * np.pi * f_psd[None, :]

    if metric == "disp":
        return Hx
    if metric == "vel":
        return (1j * w) * Hx
    if metric == "acc":
        return 1.0 + (-w * w) * Hx
    if metric == "pv":
        return (2.0 * np.pi * f0[:, None]) * Hx
    raise ValueError(f"Unknown metric: {metric}")
