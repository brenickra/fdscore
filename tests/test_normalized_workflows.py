import numpy as np
import pytest

from fdscore import SNParams, SDOFParams, ValidationError, compute_fds_time, compute_fds_spectral_psd, invert_fds_closed_form


def test_snparams_normalized_factory():
    sn = SNParams.normalized(slope_k=6.0, amplitude_from_range=False)

    assert sn.slope_k == 6.0
    assert sn.ref_stress == 1.0
    assert sn.ref_cycles == 1.0
    assert sn.amplitude_from_range is False
    assert sn.C() == 1.0


def test_closed_form_psd_is_invariant_to_global_damage_scaling():
    fs = 1000.0
    t = np.arange(0.0, 3.0, 1.0 / fs)
    x = 0.08 * np.sin(2.0 * np.pi * 23.0 * t) + 0.03 * np.sin(2.0 * np.pi * 91.0 * t)

    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=150.0, df=5.0, metric="pv")

    sn_norm = SNParams.normalized(slope_k=5.0)
    fds_norm = compute_fds_time(
        x,
        fs,
        sn=sn_norm,
        sdof=sdof,
        detrend="none",
        p_scale=1.0,
        batch_size=16,
    )

    sn_phys = SNParams(slope_k=5.0, ref_stress=120.0, ref_cycles=2.5e6)
    fds_phys = compute_fds_time(
        x,
        fs,
        sn=sn_phys,
        sdof=sdof,
        detrend="none",
        p_scale=6500.0,
        batch_size=16,
    )

    mask = (fds_norm.damage > 0.0) & (fds_phys.damage > 0.0)
    ratio = fds_phys.damage[mask] / fds_norm.damage[mask]
    expected = (6500.0 ** 5.0) / (2.5e6 * (120.0 ** 5.0))

    assert np.allclose(ratio, expected, rtol=1e-12, atol=1e-15)

    psd_norm = invert_fds_closed_form(fds_norm, test_duration_s=3600.0)
    psd_phys = invert_fds_closed_form(fds_phys, test_duration_s=3600.0)

    assert np.allclose(psd_norm.f, psd_phys.f, rtol=0.0, atol=0.0)
    assert np.allclose(psd_norm.psd, psd_phys.psd, rtol=1e-12, atol=1e-30)


def test_nonunit_sn_requires_explicit_pscale_in_time_fds():
    fs = 500.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    x = 0.1 * np.sin(2.0 * np.pi * 30.0 * t)

    sn = SNParams(slope_k=4.0, ref_stress=120.0, ref_cycles=1e6)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=80.0, df=10.0, metric="pv")

    with pytest.raises(ValidationError):
        compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="none")


def test_nonunit_sn_requires_explicit_pscale_in_spectral_fds():
    sn = SNParams(slope_k=4.0, ref_stress=120.0, ref_cycles=1e6)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=80.0, df=10.0, metric="pv")

    f = np.linspace(0.0, 200.0, 201)
    psd = np.full_like(f, 1e-6)

    with pytest.raises(ValidationError):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof)
