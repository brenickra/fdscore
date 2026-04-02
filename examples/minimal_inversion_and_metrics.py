from __future__ import annotations

import numpy as np

from fdscore import (
    SNParams,
    SDOFParams,
    compute_fds_time,
    compute_psd_metrics,
    invert_fds_closed_form,
    synthesize_time_from_psd,
)

from ._common import build_example_psd, median_abs_log10


def main() -> None:
    fs = 1000.0
    duration_s = 12.0

    f_ref, p_ref = build_example_psd(fs=fs, fmax_hz=300.0)
    x = synthesize_time_from_psd(f_psd_hz=f_ref, psd=p_ref, fs=fs, duration_s=duration_s, seed=7)

    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=5.0)

    sn_norm = SNParams(slope_k=4.0)
    fds_norm = compute_fds_time(x, fs, sn=sn_norm, sdof=sdof, detrend="mean")
    psd_norm = invert_fds_closed_form(fds_norm, test_duration_s=duration_s)
    metrics = compute_psd_metrics(psd_norm, duration_s=duration_s, acc_unit="g")

    sn_phys = SNParams(slope_k=4.0, ref_stress=10.0, ref_cycles=1.0e6)
    p_scale_phys = 300.0
    fds_phys = compute_fds_time(x, fs, sn=sn_phys, sdof=sdof, detrend="mean", p_scale=p_scale_phys)
    psd_phys = invert_fds_closed_form(fds_phys, test_duration_s=duration_s)

    err_psd = median_abs_log10(psd_norm.psd, psd_phys.psd)
    ratio_peak = float(np.median(psd_phys.psd / np.clip(psd_norm.psd, 1e-30, None)))

    print("Closed-form inversion and PSD metrics example")
    print(f"  inverted PSD bins: {psd_norm.f.size}")
    print(f"  normalized RMS: {metrics.rms_acc_g:.6f} g")
    print(f"  normalized peak: {metrics.peak_acc_g:.6f} g")
    print(f"  normalized displacement pk-pk: {metrics.disp_pk_pk_mm:.6f} mm")
    print(f"  median abs log10 difference (normalized vs physical inversion): {err_psd:.6e}")
    print(f"  median PSD ratio (physical / normalized): {ratio_peak:.6f}")


if __name__ == "__main__":
    main()
