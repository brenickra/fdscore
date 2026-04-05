import numpy as np

from fdscore import SDOFParams, compute_ers_sine
from fdscore.sdof_transfer import build_transfer_psd


def test_compute_ers_sine_matches_transfer_amplitude_for_acc_input():
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    ers = compute_ers_sine(freq_hz=40.0, amp=2.5, sdof=sdof, input_motion="acc")

    zeta = 1.0 / (2.0 * sdof.q)
    H = build_transfer_psd(
        f_psd_hz=np.asarray([40.0]),
        f0_hz=ers.f,
        zeta=zeta,
        metric=sdof.metric,
    )
    expected = np.abs(H[:, 0]) * 2.5

    assert np.allclose(ers.response, expected)
    assert ers.meta["metric"] == "acc"
    assert ers.meta["peak_mode"] == "abs"


def test_compute_ers_sine_converts_velocity_input_to_equivalent_base_acceleration():
    sdof = SDOFParams(q=12.0, metric="pv", fmin=20.0, fmax=120.0, df=20.0)
    freq_hz = 30.0
    base_acc_amp = 15.0
    vel_amp = base_acc_amp / (2.0 * np.pi * freq_hz)

    ers_acc = compute_ers_sine(freq_hz=freq_hz, amp=base_acc_amp, sdof=sdof, input_motion="acc")
    ers_vel = compute_ers_sine(freq_hz=freq_hz, amp=vel_amp, sdof=sdof, input_motion="vel")

    assert np.allclose(ers_acc.response, ers_vel.response)
