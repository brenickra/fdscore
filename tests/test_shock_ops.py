import numpy as np
import pytest

from fdscore import (
    ERSResult,
    SDOFParams,
    ShockSpectrumPair,
    compute_pvss_time,
    compute_srs_time,
    envelope_pvss,
    envelope_srs,
)
from fdscore.validate import ValidationError


def _signal(fs: float = 4096.0, duration_s: float = 1.0, *, amp1: float = 1.0, amp2: float = -0.8) -> tuple[np.ndarray, float]:
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = np.zeros_like(t)
    m1 = (t >= 0.2) & (t < 0.21)
    m2 = (t >= 0.55) & (t < 0.558)
    x[m1] += amp1 * np.sin(np.pi * (t[m1] - 0.2) / 0.01)
    x[m2] += amp2 * np.sin(np.pi * (t[m2] - 0.55) / 0.008)
    return x, fs


def test_envelope_srs_on_one_sided_results_matches_pointwise_max():
    x1, fs = _signal(amp1=1.0, amp2=-0.8)
    x2, _ = _signal(amp1=1.5, amp2=-0.5)
    sdof = SDOFParams(q=10.0, metric="acc", fmin=10.0, fmax=200.0, df=10.0)

    s1 = compute_srs_time(x1, fs, sdof, detrend="none", peak_mode="abs")
    s2 = compute_srs_time(x2, fs, sdof, detrend="none", peak_mode="abs")
    out = envelope_srs([s1, s2])

    assert isinstance(out, ERSResult)
    assert np.allclose(out.response, np.maximum(s1.response, s2.response))
    assert out.meta["provenance"]["source"] == "envelope_srs"


def test_envelope_pvss_on_both_sided_pairs_matches_sidewise_max():
    x1, fs = _signal(amp1=1.0, amp2=-0.8)
    x2, _ = _signal(amp1=0.7, amp2=-1.2)
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)

    p1 = compute_pvss_time(x1, fs, sdof, detrend="none", peak_mode="both")
    p2 = compute_pvss_time(x2, fs, sdof, detrend="none", peak_mode="both")
    out = envelope_pvss([p1, p2])

    assert isinstance(out, ShockSpectrumPair)
    assert np.allclose(out.neg.response, np.maximum(p1.neg.response, p2.neg.response))
    assert np.allclose(out.pos.response, np.maximum(p1.pos.response, p2.pos.response))
    assert out.meta["peak_mode"] == "both"
    assert out.meta["provenance"]["source"] == "envelope_pvss"


def test_envelope_srs_rejects_wrong_ers_kind():
    f = np.array([10.0, 20.0, 40.0])
    bad = ERSResult(
        f=f,
        response=np.ones_like(f),
        meta={"compat": {"engine": "x", "metric": "acc", "q": 10.0, "peak_mode": "abs", "ers_kind": "response_spectrum"}},
    )

    with pytest.raises(ValidationError, match="shock_response_spectrum"):
        envelope_srs([bad])


def test_envelope_pvss_rejects_mixed_result_forms():
    x, fs = _signal()
    sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)
    one = compute_pvss_time(x, fs, sdof, detrend="none", peak_mode="abs")
    both = compute_pvss_time(x, fs, sdof, detrend="none", peak_mode="both")

    with pytest.raises(ValidationError, match="ERSResult"):
        envelope_pvss([one, both])
