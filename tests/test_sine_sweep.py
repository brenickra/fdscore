import numpy as np

from fdscore import (
    SNParams,
    SDOFParams,
    compute_ers_sine,
    compute_fds_sine,
    compute_ers_sine_sweep,
    compute_fds_sine_sweep,
)


def test_compute_ers_sine_sweep_with_one_step_matches_center_sine_linear():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    ers_sweep = compute_ers_sine_sweep(
        f_start_hz=20.0,
        f_stop_hz=60.0,
        amp=1.5,
        duration_s=10.0,
        sdof=sdof,
        spacing="linear",
        n_steps=1,
    )
    ers_sine = compute_ers_sine(freq_hz=40.0, amp=1.5, sdof=sdof)

    assert np.allclose(ers_sweep.response, ers_sine.response)


def test_compute_fds_sine_sweep_with_one_step_matches_center_sine_log():
    sn = SNParams.normalized(slope_k=4.0)
    sdof = SDOFParams(q=8.0, metric="pv", fmin=10.0, fmax=120.0, df=10.0)
    center = np.sqrt(20.0 * 80.0)

    fds_sweep = compute_fds_sine_sweep(
        f_start_hz=20.0,
        f_stop_hz=80.0,
        amp=2.0,
        duration_s=12.0,
        sn=sn,
        sdof=sdof,
        spacing="log",
        n_steps=1,
    )
    fds_sine = compute_fds_sine(freq_hz=center, amp=2.0, duration_s=12.0, sn=sn, sdof=sdof)

    assert np.allclose(fds_sweep.damage, fds_sine.damage)


def test_compute_fds_sine_sweep_scales_linearly_with_duration():
    sn = SNParams.normalized(slope_k=5.0)
    sdof = SDOFParams(q=12.0, metric="pv", fmin=20.0, fmax=120.0, df=20.0)

    fds_short = compute_fds_sine_sweep(
        f_start_hz=30.0,
        f_stop_hz=90.0,
        amp=1.0,
        duration_s=5.0,
        sn=sn,
        sdof=sdof,
        spacing="linear",
        n_steps=20,
    )
    fds_long = compute_fds_sine_sweep(
        f_start_hz=30.0,
        f_stop_hz=90.0,
        amp=1.0,
        duration_s=15.0,
        sn=sn,
        sdof=sdof,
        spacing="linear",
        n_steps=20,
    )

    assert np.allclose(fds_long.damage, 3.0 * fds_short.damage)


def test_compute_ers_sine_sweep_covers_endpoints_more_than_single_mid_frequency():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    ers_sweep = compute_ers_sine_sweep(
        f_start_hz=20.0,
        f_stop_hz=80.0,
        amp=1.0,
        duration_s=10.0,
        sdof=sdof,
        spacing="linear",
        n_steps=40,
    )
    ers_mid = compute_ers_sine(freq_hz=50.0, amp=1.0, sdof=sdof)

    idx_20 = int(np.argmin(np.abs(ers_sweep.f - 20.0)))
    idx_80 = int(np.argmin(np.abs(ers_sweep.f - 80.0)))

    assert ers_sweep.response[idx_20] >= ers_mid.response[idx_20]
    assert ers_sweep.response[idx_80] >= ers_mid.response[idx_80]
