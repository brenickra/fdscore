from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=False, fastmath=True)
def _extract_reversals_values_numba(series: np.ndarray) -> np.ndarray:
    """Return reversal values (including first and last) for rainflow counting."""
    n = series.size
    if n < 2:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    m = 0

    x_last = series[0]
    x = series[1]
    d_last = x - x_last

    out[m] = x_last
    m += 1

    for i in range(2, n):
        x_next = series[i]
        if x_next == x:
            continue
        d_next = x_next - x
        if d_last * d_next < 0.0:
            out[m] = x
            m += 1
        x_last = x
        x = x_next
        d_last = d_next

    out[m] = x
    m += 1
    return out[:m]


@njit(cache=False, fastmath=True)
def _miner_damage_numba(series: np.ndarray, k: float, c: float, use_amplitude_from_range: bool) -> float:
    """Compute Miner damage using ASTM-style rainflow on reversal points."""
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
    """Public wrapper (Numba-backed)."""
    s = np.asarray(signal, dtype=np.float64)
    return float(_miner_damage_numba(s, float(k), float(c), bool(amplitude_from_range)))


def miner_damage_from_matrix(
    signals: np.ndarray,
    *,
    k: float,
    c: float,
    amplitude_from_range: bool = True,
) -> np.ndarray:
    """Vectorized Miner damage for a matrix of signals (n_signals, n_samples)."""
    m = np.asarray(signals, dtype=np.float64)
    return _miner_damage_matrix_numba(m, float(k), float(c), bool(amplitude_from_range))
