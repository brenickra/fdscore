import pytest
import numpy as np

from fdscore import (
    SNParams,
    SDOFParams,
    SineDwellSegment,
    compute_ers_sine,
    compute_fds_sine,
    compute_ers_dwell_profile,
    compute_fds_dwell_profile,
    envelope_ers,
    ValidationError,
)


def test_compute_fds_dwell_profile_matches_sum_of_segment_fds():
    sn = SNParams.normalized(slope_k=4.0)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0)
    segments = [
        SineDwellSegment(freq_hz=20.0, amp=1.0, duration_s=5.0),
        SineDwellSegment(freq_hz=40.0, amp=0.8, duration_s=12.0),
        SineDwellSegment(freq_hz=70.0, amp=1.3, duration_s=4.0),
    ]

    mission = compute_fds_dwell_profile(segments, sn=sn, sdof=sdof)
    direct = sum(
        compute_fds_sine(freq_hz=s.freq_hz, amp=s.amp, duration_s=s.duration_s, sn=sn, sdof=sdof).damage
        for s in segments
    )

    assert np.allclose(mission.damage, direct)


def test_compute_ers_dwell_profile_matches_pointwise_envelope():
    sdof = SDOFParams(q=12.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0)
    segments = [
        SineDwellSegment(freq_hz=20.0, amp=1.0, duration_s=5.0),
        SineDwellSegment(freq_hz=60.0, amp=0.5, duration_s=7.0),
        SineDwellSegment(freq_hz=80.0, amp=1.6, duration_s=3.0),
    ]

    mission = compute_ers_dwell_profile(segments, sdof=sdof)
    expected = np.maximum.reduce([
        compute_ers_sine(freq_hz=s.freq_hz, amp=s.amp, sdof=sdof).response
        for s in segments
    ])

    assert np.allclose(mission.response, expected)


def test_envelope_ers_rejects_metric_mismatch():
    ers_acc = compute_ers_sine(
        freq_hz=20.0,
        amp=1.0,
        sdof=SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=100.0, df=10.0),
    )
    ers_pv = compute_ers_sine(
        freq_hz=20.0,
        amp=1.0,
        sdof=SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=100.0, df=10.0),
    )

    with pytest.raises(ValidationError, match="metric"):
        envelope_ers([ers_acc, ers_pv])
