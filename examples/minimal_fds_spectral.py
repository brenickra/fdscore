from __future__ import annotations

from fdscore import (
    PSDParams,
    SDOFParams,
    SNParams,
    compute_fds_spectral_psd,
    compute_fds_spectral_time,
    synthesize_time_from_psd,
)

from ._common import build_example_psd, median_abs_log10


def main() -> None:
    fs = 1000.0
    duration_s = 16.0

    sn = SNParams(slope_k=4.0)
    sdof = SDOFParams(q=10.0, metric="disp", fmin=10.0, fmax=200.0, df=5.0)
    psd_params = PSDParams(method="welch", window="hann", nperseg=2048, noverlap=1024, detrend="constant")

    f_psd, p_ref = build_example_psd(fs=fs, fmax_hz=300.0)
    x = synthesize_time_from_psd(f_psd_hz=f_psd, psd=p_ref, fs=fs, duration_s=duration_s, seed=42)

    fds_from_psd = compute_fds_spectral_psd(
        f_psd_hz=f_psd,
        psd_baseacc=p_ref,
        duration_s=duration_s,
        sn=sn,
        sdof=sdof,
    )
    fds_from_time = compute_fds_spectral_time(
        x,
        fs,
        sn=sn,
        sdof=sdof,
        psd=psd_params,
        duration_s=duration_s,
    )

    err = median_abs_log10(fds_from_psd.damage, fds_from_time.damage)

    print("Spectral FDS example")
    print(f"  oscillator bins: {fds_from_psd.f.size}")
    print(f"  reference PSD bins: {f_psd.size}")
    print(f"  median abs log10 difference (PSD vs time->Welch): {err:.6f}")


if __name__ == "__main__":
    main()
