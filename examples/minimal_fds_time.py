from __future__ import annotations

import numpy as np

from fdscore import SNParams, SDOFParams, compute_fds_time, prepare_fds_time_plan

from ._common import build_multitone_signal


def main() -> None:
    fs = 1000.0
    duration_s = 4.0

    sn = SNParams(slope_k=4.0)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=5.0, fmax=200.0, df=5.0)

    x = build_multitone_signal(fs=fs, duration_s=duration_s)
    y = 0.7 * x + 0.02 * np.sin(2.0 * np.pi * 95.0 * np.arange(x.size) / fs)

    plan = prepare_fds_time_plan(fs=fs, n_samples=x.size, sdof=sdof)
    fds_x = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="mean", plan=plan)
    fds_y = compute_fds_time(y, fs, sn=sn, sdof=sdof, detrend="mean", plan=plan)

    idx_x = int(np.argmax(fds_x.damage))
    idx_y = int(np.argmax(fds_y.damage))

    print("Time-domain FDS example")
    print(f"  channels: 2")
    print(f"  oscillator bins: {fds_x.f.size}")
    print(f"  channel X peak: f={fds_x.f[idx_x]:.1f} Hz, damage={fds_x.damage[idx_x]:.6e}")
    print(f"  channel Y peak: f={fds_y.f[idx_y]:.1f} Hz, damage={fds_y.damage[idx_y]:.6e}")
    print(f"  plan metric: {plan.metric}, n_fft_bins={plan.H.shape[1]}")


if __name__ == "__main__":
    main()
