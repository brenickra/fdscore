from __future__ import annotations

import numpy as np
from numba import njit, prange

from .validate import ValidationError


@njit(cache=False, fastmath=True)
def _extract_reversals_values_numba(series: np.ndarray) -> np.ndarray:
    """Extract reversal values for ASTM-style rainflow counting.

    The first and last points are always retained so that residual half cycles
    can be closed consistently after the main stack-reduction pass.

    Parameters
    ----------
    series:
        One-dimensional response history.

    Returns
    -------
    ndarray
        Sequence of reversal values containing the first point, all strict
        turning points, and the final point.

    Notes
    -----
    Flat segments are skipped by ignoring repeated consecutive values. The
    implementation is designed for the compact reversal representation used by
    the Numba rainflow kernels in this module.
    """
    n = series.size
    if n < 2:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    m = 0

    first = series[0]
    x = series[1]
    d_last = x - first

    out[m] = first
    m += 1

    for i in range(2, n):
        x_next = series[i]
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0.0:
            out[m] = x
            m += 1
        x = x_next
        d_last = d_next

    out[m] = x
    m += 1
    return out[:m]


@njit(cache=False, fastmath=True)
def _miner_damage_numba(series: np.ndarray, k: float, c: float, use_amplitude_from_range: bool) -> float:
    r"""Compute Miner damage from a single history using a rainflow stack.

    The algorithm first reduces the input history to reversal points and then
    applies an ASTM E1049-style stack procedure. Closed ranges contribute full
    cycles, while unresolved residual ranges at the end of the pass contribute
    half cycles. Damage is accumulated under Miner's linear damage rule as

    .. math::

       D = \sum_j \phi_j \frac{S_j^k}{C}

    where :math:`\phi_j` is ``1.0`` for a closed cycle and ``0.5`` for a half
    cycle.

    Parameters
    ----------
    series:
        One-dimensional response history.
    k:
        S-N curve slope exponent.
    c:
        S-N curve intercept :math:`C`.
    use_amplitude_from_range:
        If ``True``, interpret the rainflow range as twice the alternating
        amplitude and use :math:`S = range / 2`. If ``False``, use the full
        range directly as the damage-driving load quantity.

    Returns
    -------
    float
        Miner damage accumulated over the full history.

    Notes
    -----
    This kernel is intentionally low level and performs no input validation.
    Validation is handled by the public wrappers.

    References
    ----------
    ASTM E1049-85(2017). *Standard Practices for Cycle Counting in Fatigue
        Analysis*.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." *Journal of Applied
        Mechanics*, 12(3), A159-A164.
    """
    rev = _extract_reversals_values_numba(series)
    n_rev = rev.size
    if n_rev < 2:
        return 0.0

    points = np.empty(n_rev, dtype=np.float64)
    start = 0
    end = 0
    dmg = 0.0

    for i in range(n_rev):
        points[end] = rev[i]
        end += 1

        while (end - start) >= 3:
            x1 = points[end - 3]
            x2 = points[end - 2]
            x3 = points[end - 1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)

            if X < Y:
                break

            load = 0.5 * Y if use_amplitude_from_range else Y
            if load > 0.0:
                if (end - start) == 3:
                    dmg += 0.5 * (load ** k) / c
                else:
                    dmg += 1.0 * (load ** k) / c

            if (end - start) == 3:
                start += 1
            else:
                last = points[end - 1]
                end -= 3
                points[end] = last
                end += 1

    while (end - start) > 1:
        rng = abs(points[start + 1] - points[start])
        load = 0.5 * rng if use_amplitude_from_range else rng
        if load > 0.0:
            dmg += 0.5 * (load ** k) / c
        start += 1

    return dmg


@njit(cache=False, parallel=True, fastmath=True)
def _miner_damage_matrix_numba(signals: np.ndarray, k: float, c: float, use_amplitude_from_range: bool) -> np.ndarray:
    n = signals.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in prange(n):
        out[i] = _miner_damage_numba(signals[i], k, c, use_amplitude_from_range)
    return out


def miner_damage_from_signal(
    signal: np.ndarray,
    *,
    k: float,
    c: float,
    amplitude_from_range: bool = True,
) -> float:
    r"""Compute Miner damage for a single response history.

    This is the public, validated entry point for the ASTM-style rainflow and
    Miner's-rule implementation used by the library.

    Parameters
    ----------
    signal : numpy.ndarray
        One-dimensional response history.
    k : float
        S-N curve slope exponent.
    c : float
        S-N curve intercept :math:`C`.
    amplitude_from_range : bool
        If ``True``, each rainflow range is converted to alternating amplitude
        by dividing by two before applying the S-N relationship. If ``False``,
        the full range is used directly.

    Returns
    -------
    float
        Total Miner damage for the history.

    Notes
    -----
    This wrapper validates dimensionality, finiteness, and positivity of the
    fatigue parameters before delegating to the Numba kernel.

    References
    ----------
    ASTM E1049-85(2017). Standard Practices for Cycle Counting in Fatigue
        Analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    """
    s = np.asarray(signal, dtype=np.float64)
    if s.ndim != 1:
        raise ValidationError("signal must be a 1D array.")
    if not np.all(np.isfinite(s)):
        raise ValidationError("signal must contain only finite values.")
    k_val = float(k)
    c_val = float(c)
    if not np.isfinite(k_val) or k_val <= 0.0:
        raise ValidationError("k must be finite and > 0.")
    if not np.isfinite(c_val) or c_val <= 0.0:
        raise ValidationError("c must be finite and > 0.")
    return float(_miner_damage_numba(s, k_val, c_val, bool(amplitude_from_range)))



def miner_damage_from_matrix(
    signals: np.ndarray,
    *,
    k: float,
    c: float,
    amplitude_from_range: bool = True,
) -> np.ndarray:
    r"""Compute Miner damage for a batch of response histories.

    Each row of ``signals`` is interpreted as an independent response history,
    and damage is accumulated row by row using the same ASTM-style rainflow and
    Miner's-rule logic as :func:`miner_damage_from_signal`.

    Parameters
    ----------
    signals : numpy.ndarray
        Two-dimensional array with shape ``(n_signals, n_samples)``.
    k : float
        S-N curve slope exponent.
    c : float
        S-N curve intercept :math:`C`.
    amplitude_from_range : bool
        If ``True``, convert each rainflow range to alternating amplitude by
        dividing by two before evaluating damage. If ``False``, use the full
        range directly.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of Miner damage values with length
        ``n_signals``.

    Notes
    -----
    This wrapper performs shape and finiteness validation and then dispatches
    the row-wise computation to a parallel Numba kernel.

    References
    ----------
    ASTM E1049-85(2017). Standard Practices for Cycle Counting in Fatigue
        Analysis.
    Miner, M. A. (1945). "Cumulative Damage in Fatigue." Journal of Applied
        Mechanics, 12(3), A159-A164.
    """
    m = np.asarray(signals, dtype=np.float64)
    if m.ndim != 2:
        raise ValidationError("signals must be a 2D array with shape (n_signals, n_samples).")
    if not np.all(np.isfinite(m)):
        raise ValidationError("signals must contain only finite values.")
    k_val = float(k)
    c_val = float(c)
    if not np.isfinite(k_val) or k_val <= 0.0:
        raise ValidationError("k must be finite and > 0.")
    if not np.isfinite(c_val) or c_val <= 0.0:
        raise ValidationError("c must be finite and > 0.")
    return _miner_damage_matrix_numba(m, k_val, c_val, bool(amplitude_from_range))
