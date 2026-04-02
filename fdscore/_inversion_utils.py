from __future__ import annotations

import numpy as np


def moving_average_reflect(x: np.ndarray, win: int) -> np.ndarray:
    win = int(win)
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(xp, kernel, mode="valid")


def smooth_psd_log10(P: np.ndarray, win: int, floor: float) -> np.ndarray:
    P = np.clip(P, floor, None)
    logP = np.log10(P)
    logP_s = moving_average_reflect(logP, win)
    return np.power(10.0, logP_s)


def blend_log_curves(cur: np.ndarray, ref: np.ndarray, weight: np.ndarray, floor: float) -> np.ndarray:
    w = np.clip(np.asarray(weight, dtype=float), 0.0, 1.0)
    return np.exp(
        (1.0 - w) * np.log(np.clip(cur, floor, None))
        + w * np.log(np.clip(ref, floor, None))
    )


def build_edge_taper_weights(f_psd: np.ndarray, edge_hz: float) -> np.ndarray:
    edge = float(edge_hz)
    w = np.zeros_like(f_psd, dtype=float)
    if edge <= 0.0:
        return w
    fmin = float(np.min(f_psd))
    fmax = float(np.max(f_psd))
    low = f_psd <= (fmin + edge)
    high = f_psd >= (fmax - edge)
    if np.any(low):
        w[low] = np.maximum(w[low], 1.0 - (f_psd[low] - fmin) / edge)
    if np.any(high):
        w[high] = np.maximum(w[high], 1.0 - (fmax - f_psd[high]) / edge)
    return np.clip(w, 0.0, 1.0)


def apply_edge_caps(
    P: np.ndarray,
    f_psd: np.ndarray,
    *,
    tail_cap_start_hz: float,
    tail_cap_ratio: float,
    low_cap_ratio: float,
    floor: float,
) -> np.ndarray:
    p = np.clip(np.asarray(P, dtype=float).copy(), floor, None)

    low_cap = float(low_cap_ratio)
    if low_cap > 0.0 and p.size > 1:
        p[0] = min(p[0], p[1] * low_cap)

    t_ratio = float(tail_cap_ratio)
    if t_ratio > 0.0 and p.size > 1:
        idx = int(np.searchsorted(f_psd, float(tail_cap_start_hz)))
        idx = max(1, min(idx, p.size - 1))
        for i in range(idx + 1, p.size):
            lim = p[i - 1] * t_ratio
            if p[i] > lim:
                p[i] = lim

    return np.clip(p, floor, None)
