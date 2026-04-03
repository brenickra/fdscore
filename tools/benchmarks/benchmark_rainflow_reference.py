from __future__ import annotations

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rainflow

from fdscore import synthesize_time_from_psd
from fdscore.rainflow_damage import miner_damage_from_matrix, miner_damage_from_signal


def reference_damage(signal: np.ndarray, *, k: float, c: float, amplitude_from_range: bool) -> float:
    dmg = 0.0
    for rng, _mean, count, _i0, _i1 in rainflow.extract_cycles(np.asarray(signal, dtype=float)):
        if rng <= 0.0:
            continue
        load = 0.5 * float(rng) if amplitude_from_range else float(rng)
        if load <= 0.0:
            continue
        dmg += float(count) * (load ** float(k)) / float(c)
    return float(dmg)


def build_corpus(n_signals: int = 8, n_samples: int = 120_000) -> np.ndarray:
    t = np.linspace(0.0, n_samples / 1000.0, n_samples, endpoint=False)
    rng = np.random.default_rng(2026)

    f_psd = np.linspace(0.0, 400.0, 401)
    psd = (
        2.0e-4 * np.exp(-0.5 * ((f_psd - 24.0) / 9.0) ** 2)
        + 4.5e-4 * np.exp(-0.5 * ((f_psd - 90.0) / 20.0) ** 2)
        + 2.0e-4 * np.exp(-0.5 * ((f_psd - 180.0) / 32.0) ** 2)
        + 1.0e-6
    )
    psd[0] = 0.0

    signals: list[np.ndarray] = []
    for i in range(n_signals):
        phase = 0.2 * i
        sig = (
            0.55 * np.sin(2.0 * np.pi * (9.0 + i) * t + phase)
            + 0.18 * np.sin(2.0 * np.pi * (37.0 + 2.0 * i) * t + 0.7 * phase)
            + 0.06 * np.cumsum(rng.normal(size=n_samples)) / np.sqrt(n_samples)
        )
        sig += 0.35 * synthesize_time_from_psd(
            f_psd_hz=f_psd,
            psd=psd,
            fs=1000.0,
            duration_s=n_samples / 1000.0,
            seed=100 + i,
        )
        signals.append(np.asarray(sig, dtype=float))
    return np.vstack(signals)


def main() -> None:
    k = 6.0
    c = 1.0e6
    amplitude_from_range = True
    signals = build_corpus()

    miner_damage_from_signal(signals[0], k=k, c=c, amplitude_from_range=amplitude_from_range)
    miner_damage_from_matrix(signals[:1], k=k, c=c, amplitude_from_range=amplitude_from_range)

    t0 = time.perf_counter()
    ref = np.array(
        [reference_damage(sig, k=k, c=c, amplitude_from_range=amplitude_from_range) for sig in signals],
        dtype=float,
    )
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours_scalar = np.array(
        [miner_damage_from_signal(sig, k=k, c=c, amplitude_from_range=amplitude_from_range) for sig in signals],
        dtype=float,
    )
    t_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    ours_batch = miner_damage_from_matrix(signals, k=k, c=c, amplitude_from_range=amplitude_from_range)
    t_batch = time.perf_counter() - t0

    max_abs_diff_scalar = float(np.max(np.abs(ours_scalar - ref)))
    max_abs_diff_batch = float(np.max(np.abs(ours_batch - ref)))

    print("Rainflow reference benchmark")
    print(f"  signals: {signals.shape[0]}")
    print(f"  samples per signal: {signals.shape[1]}")
    print(f"  rainflow reference: {t_ref:.3f} s")
    print(f"  fdscore scalar:     {t_scalar:.3f} s")
    print(f"  fdscore batch:      {t_batch:.3f} s")
    print(f"  scalar speedup:     {t_ref / max(t_scalar, 1e-12):.2f}x")
    print(f"  batch speedup:      {t_ref / max(t_batch, 1e-12):.2f}x")
    print(f"  max abs diff scalar: {max_abs_diff_scalar:.6e}")
    print(f"  max abs diff batch:  {max_abs_diff_batch:.6e}")


if __name__ == "__main__":
    main()
