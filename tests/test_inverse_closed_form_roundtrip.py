import numpy as np
import pytest
from fdscore import FDSResult, invert_fds_closed_form
from fdscore.inverse_closed_form import compute_psd_from_fds_closed_form, compute_fds_from_psd_closed_form
from fdscore.validate import ValidationError


def test_closed_form_roundtrip_dp_psd():
    rng = np.random.default_rng(0)
    f = np.linspace(5.0, 200.0, 50)
    dp = 10 ** rng.uniform(-8, -2, size=f.size)
    zeta = 1.0/(2.0*10.0)
    b = 3.0
    T = 3600.0

    psd = compute_psd_from_fds_closed_form(f0_hz=f, dp_fds=dp, zeta=zeta, b=b, test_duration_s=T)
    dp2 = compute_fds_from_psd_closed_form(f0_hz=f, psd=psd, zeta=zeta, b=b, test_duration_s=T)

    assert np.allclose(dp, dp2, rtol=1e-10, atol=1e-30)


def test_closed_form_rejects_noncurrent_compat_schema():
    f = np.array([10.0, 20.0, 30.0])
    damage = np.array([1e-6, 2e-6, 3e-6])
    fds = FDSResult(
        f=f,
        damage=damage,
        meta={
            "compat": {
                "metric": "pv",
                "q": 10.0,
                "p_scale": 1.0,
                "sn": {
                    "k": 3.0,
                    "Sref": 1.0,
                    "Nref": 1.0,
                    "range2amp": True,
                },
            }
        },
    )

    with pytest.raises(ValidationError):
        invert_fds_closed_form(fds, test_duration_s=3600.0)


def test_closed_form_rejects_nonpositive_q_in_compat_metadata():
    f = np.array([10.0, 20.0, 30.0])
    damage = np.array([1e-6, 2e-6, 3e-6])
    fds = FDSResult(
        f=f,
        damage=damage,
        meta={
            "compat": {
                "engine": "time_rainflow_fft_numba",
                "metric": "pv",
                "q": 0.0,
                "p_scale": 1.0,
                "sn": {
                    "slope_k": 3.0,
                    "ref_stress": 1.0,
                    "ref_cycles": 1.0,
                    "amplitude_from_range": True,
                },
                "fds_kind": "damage_spectrum",
            }
        },
    )

    with pytest.raises(ValidationError, match="Invalid q"):
        invert_fds_closed_form(fds, test_duration_s=3600.0)
