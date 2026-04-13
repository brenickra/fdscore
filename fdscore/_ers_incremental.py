"""Incremental engine for time-domain ERS computation.

This internal module implements the ``engine="incremental"`` path used by
``fdscore.ers_time.compute_ers_time``. The goal is the same as in the
incremental FDS engine: avoid materializing the full oscillator-response
matrix produced by the legacy FFT workflow.

Strategy
--------
Instead of reconstructing every oscillator response with inverse FFT and
then extracting its peak, the incremental ERS engine:

1. Integrates each SDOF oscillator sample-by-sample using exact
   zero-order-hold state-transition matrices.
2. Evaluates the requested response metric online.
3. Tracks the absolute peak for each oscillator without storing the full
   time history.

Oscillators near Nyquist are integrated with adaptive upsampling so that
the ZOH amplitude error stays controlled in the same spirit as the
incremental FDS engine.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._fds_incremental import (
    _ZOH_R_MAX_DEFAULT,
    _build_iir_coefficients,
    _compute_upsample_factors,
    _upsample_signal,
)
from .types import Metric


@njit(cache=False, fastmath=True, parallel=True)
def _final_states_numba(
    x: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Return final oscillator states after one record with zero initial state."""
    n = x.size
    n_osc = phi.shape[0]
    states = np.zeros((n_osc, 2), dtype=np.float64)

    for i in prange(n_osc):
        p00 = phi[i, 0, 0]
        p01 = phi[i, 0, 1]
        p10 = phi[i, 1, 0]
        p11 = phi[i, 1, 1]
        g0 = gamma[i, 0]
        g1 = gamma[i, 1]

        z = 0.0
        zdot = 0.0
        for j in range(n):
            a_base = x[j]
            z_new = p00 * z + p01 * zdot + g0 * a_base
            zdot_new = p10 * z + p11 * zdot + g1 * a_base
            z = z_new
            zdot = zdot_new

        states[i, 0] = z
        states[i, 1] = zdot

    return states


def _periodic_initial_states(
    x: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Solve the periodic initial state that matches circular convolution.

    The FFT baseline implicitly assumes periodic extension of the input
    record. The corresponding state-space initial condition satisfies
    ``s0 = A_period @ s0 + b_period`` after one full record. This helper
    computes that periodic state so the incremental engine aligns with
    the FFT-domain baseline instead of using rest initial conditions.
    """
    final_zero = _final_states_numba(x, phi, gamma)
    init = np.zeros_like(final_zero)

    n = int(x.size)
    eye = np.eye(2, dtype=np.float64)
    for i in range(phi.shape[0]):
        a_period = np.linalg.matrix_power(np.asarray(phi[i], dtype=np.float64), n)
        init[i, :] = np.linalg.solve(eye - a_period, final_zero[i])

    return init


@njit(cache=False, fastmath=True, parallel=True)
def _integrate_peaks_numba(
    x: np.ndarray,
    phi: np.ndarray,
    gamma: np.ndarray,
    cz: np.ndarray,
    czdot: np.ndarray,
    initial_state: np.ndarray,
    sample_stride: int,
) -> np.ndarray:
    """Integrate a group of oscillators and track absolute peaks online.

    Parameters
    ----------
    x : numpy.ndarray
        Input signal, potentially upsampled for the current oscillator
        group.
    phi : numpy.ndarray
        State-transition matrices with shape ``(n_osc, 2, 2)``.
    gamma : numpy.ndarray
        Input-coupling vectors with shape ``(n_osc, 2)``.
    cz : numpy.ndarray
        Displacement output coefficients with shape ``(n_osc,)``.
    czdot : numpy.ndarray
        Velocity output coefficients with shape ``(n_osc,)``.
    initial_state : numpy.ndarray
        Initial oscillator states with shape ``(n_osc, 2)``.
    sample_stride : int
        Peak-extraction stride. A value of ``1`` evaluates every sample.
        For upsampled groups, the peak is evaluated only at samples that
        coincide with the original-rate time grid.

    Returns
    -------
    numpy.ndarray
        Peak absolute response for each oscillator in the group.
    """
    n = x.size
    n_osc = phi.shape[0]
    peaks = np.zeros(n_osc, dtype=np.float64)

    for i in prange(n_osc):
        p00 = phi[i, 0, 0]
        p01 = phi[i, 0, 1]
        p10 = phi[i, 1, 0]
        p11 = phi[i, 1, 1]
        g0 = gamma[i, 0]
        g1 = gamma[i, 1]
        cz_i = cz[i]
        czd_i = czdot[i]

        z = initial_state[i, 0]
        zdot = initial_state[i, 1]
        peak = 0.0

        for j in range(n):
            a_base = x[j]

            if j % sample_stride == 0:
                value = cz_i * z + czd_i * zdot
                mag = abs(value)
                if mag > peak:
                    peak = mag

            z_new = p00 * z + p01 * zdot + g0 * a_base
            zdot_new = p10 * z + p11 * zdot + g1 * a_base
            z = z_new
            zdot = zdot_new

        peaks[i] = peak

    return peaks


def ers_incremental(
    y: np.ndarray,
    *,
    fs: float,
    f0: np.ndarray,
    zeta: float,
    metric: Metric,
    zoh_r_max: float = _ZOH_R_MAX_DEFAULT,
) -> np.ndarray:
    """Compute ERS peaks via incremental ZOH SDOF integration.

    Parameters
    ----------
    y : numpy.ndarray
        Preprocessed base-acceleration history.
    fs : float
        Sampling rate in Hz.
    f0 : numpy.ndarray
        Validated oscillator frequency grid in Hz.
    zeta : float
        Oscillator damping ratio.
    metric : Metric
        Response metric. One of ``"pv"``, ``"disp"``, ``"vel"``, or
        ``"acc"``.
    zoh_r_max : float, optional
        Maximum tolerated ``f0 / Nyquist_effective`` ratio after adaptive
        upsampling.

    Returns
    -------
    numpy.ndarray
        Peak absolute response for each oscillator frequency.

    Notes
    -----
    The peak is evaluated on the original sample grid even when a group
    of oscillators is integrated with an upsampled input signal. This
    keeps the incremental engine aligned with the current FFT baseline,
    which reconstructs responses on the original record length.
    """
    fs = float(fs)
    zeta = float(zeta)
    r_max = float(zoh_r_max)

    peaks = np.zeros(len(f0), dtype=np.float64)
    factors = _compute_upsample_factors(f0, fs, r_max)

    y_cache: dict[int, np.ndarray] = {}

    for factor in np.unique(factors):
        indices = np.where(factors == factor)[0]
        f0_group = f0[indices]
        fs_eff = fs * float(factor)
        dt_eff = 1.0 / fs_eff

        phi, gamma, cz, czdot = _build_iir_coefficients(
            f0_group,
            zeta,
            dt_eff,
            metric,
        )
        phi = np.ascontiguousarray(phi, dtype=np.float64)
        gamma = np.ascontiguousarray(gamma, dtype=np.float64)

        if factor not in y_cache:
            y_cache[factor] = _upsample_signal(y, int(factor))
        y_up = y_cache[int(factor)]

        initial_state = _periodic_initial_states(y_up, phi, gamma)

        peaks_group = _integrate_peaks_numba(
            y_up,
            phi,
            gamma,
            np.ascontiguousarray(cz, dtype=np.float64),
            np.ascontiguousarray(czdot, dtype=np.float64),
            np.ascontiguousarray(initial_state, dtype=np.float64),
            int(factor),
        )
        peaks[indices] = peaks_group

    return peaks
