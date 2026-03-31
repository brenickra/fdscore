import numpy as np
from fdscore.inverse_closed_form import compute_psd_from_fds_closed_form, compute_fds_from_psd_closed_form

def test_closed_form_roundtrip_dp_psd():
    rng = np.random.default_rng(0)
    f = np.linspace(5.0, 200.0, 50)
    dp = 10 ** rng.uniform(-8, -2, size=f.size)
    zeta = 1.0/(2.0*10.0)
    b = 3.0
    T = 3600.0

    psd = compute_psd_from_fds_closed_form(f0_hz=f, dp_fds=dp, zeta=zeta, b=b, test_duration_s=T)
    dp2 = compute_fds_from_psd_closed_form(f0_hz=f, psd=psd, zeta=zeta, b=b, test_duration_s=T)

    # Should match closely (numerical eps)
    assert np.allclose(dp, dp2, rtol=1e-10, atol=1e-30)
