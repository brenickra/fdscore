import numpy as np

from fdscore import SNParams, SDOFParams, compute_ers_sine, compute_fds_sine


def test_compute_fds_sine_matches_manual_damage_from_response_amplitude():
    sn = SNParams(slope_k=4.0, ref_stress=1.0, ref_cycles=1.0, amplitude_from_range=True)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0)

    ers = compute_ers_sine(freq_hz=25.0, amp=3.0, sdof=sdof, input_motion="acc")
    fds = compute_fds_sine(
        freq_hz=25.0,
        amp=3.0,
        duration_s=12.0,
        sn=sn,
        sdof=sdof,
        input_motion="acc",
    )

    expected_cycles = 25.0 * 12.0
    expected_damage = expected_cycles * np.power(ers.response, sn.slope_k) / sn.C()

    assert np.allclose(fds.damage, expected_damage)


def test_compute_fds_sine_scales_linearly_with_duration():
    sn = SNParams.normalized(slope_k=5.0)
    sdof = SDOFParams(q=8.0, metric="pv", fmin=20.0, fmax=120.0, df=20.0)

    fds_short = compute_fds_sine(freq_hz=40.0, amp=2.0, duration_s=5.0, sn=sn, sdof=sdof)
    fds_long = compute_fds_sine(freq_hz=40.0, amp=2.0, duration_s=15.0, sn=sn, sdof=sdof)

    assert np.allclose(fds_long.damage, 3.0 * fds_short.damage)


def test_compute_fds_sine_respects_amplitude_vs_range_convention():
    sdof = SDOFParams(q=15.0, metric="pv", fmin=10.0, fmax=90.0, df=10.0)
    sn_amp = SNParams.normalized(slope_k=3.0, amplitude_from_range=True)
    sn_rng = SNParams.normalized(slope_k=3.0, amplitude_from_range=False)

    fds_amp = compute_fds_sine(freq_hz=30.0, amp=1.2, duration_s=8.0, sn=sn_amp, sdof=sdof)
    fds_rng = compute_fds_sine(freq_hz=30.0, amp=1.2, duration_s=8.0, sn=sn_rng, sdof=sdof)

    assert np.allclose(fds_rng.damage, (2.0 ** sn_amp.slope_k) * fds_amp.damage)
