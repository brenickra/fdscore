import numpy as np
import pytest

import fdscore.fds_spectral as spectral_mod
import fdscore.psd_welch as welch_mod
from fdscore import PSDParams, SDOFParams, SNParams, ValidationError, compute_fds_spectral_psd, compute_fds_spectral_time, compute_psd_welch
from fdscore.validate import validate_sdof, validate_sn


def test_compute_fds_spectral_psd_rejects_material_negative_input():
    f = np.linspace(1.0, 100.0, 100)
    psd = np.full_like(f, 1e-4)
    psd[10] = -1e-3
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    with pytest.raises(ValidationError, match="psd_baseacc"):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof, p_scale=1.0)


def test_compute_psd_welch_clamps_tiny_negative_numerical_noise(monkeypatch):
    def fake_welch(*args, **kwargs):
        f = np.array([0.0, 1.0, 2.0])
        pxx = np.array([1e-6, -1e-15, 2e-6])
        return f, pxx

    monkeypatch.setattr(welch_mod, "welch", fake_welch)

    x = np.linspace(0.0, 1.0, 16)
    params = PSDParams(method="welch", detrend="constant")
    f, pxx = compute_psd_welch(x, fs=16.0, psd=params)

    assert np.all(pxx >= 0.0)
    assert pxx[1] == 0.0


def test_compute_psd_welch_rejects_material_negative_output(monkeypatch):
    def fake_welch(*args, **kwargs):
        f = np.array([0.0, 1.0, 2.0])
        pxx = np.array([1e-6, -1e-3, 2e-6])
        return f, pxx

    monkeypatch.setattr(welch_mod, "welch", fake_welch)

    x = np.linspace(0.0, 1.0, 16)
    params = PSDParams(method="welch", detrend="constant")

    with pytest.raises(ValidationError, match="welch PSD"):
        compute_psd_welch(x, fs=16.0, psd=params)


def test_compute_fds_spectral_psd_raises_validation_for_invalid_life(monkeypatch):
    class DummySpectralData:
        def __init__(self, input):
            self.input = input

    class DummyDirlik:
        def __init__(self, sd):
            self.sd = sd

        def get_life(self, C, k):
            return 0.0

    class DummyFLife:
        SpectralData = DummySpectralData
        Dirlik = DummyDirlik

    monkeypatch.setattr(spectral_mod, "_require_flife", lambda: DummyFLife)

    f = np.linspace(1.0, 100.0, 100)
    psd = np.full_like(f, 1e-4)
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    with pytest.raises(ValidationError, match="invalid life"):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof, p_scale=1.0)


def test_compute_fds_spectral_psd_rejects_non_boolean_sn_amplitude_flag():
    f = np.linspace(1.0, 100.0, 100)
    psd = np.full_like(f, 1e-4)
    sn = SNParams.normalized(slope_k=3.0, amplitude_from_range="false")
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")

    with pytest.raises(ValidationError, match="amplitude_from_range"):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof, p_scale=1.0)


def test_validate_sn_rejects_non_numeric_scalar_inputs_with_validation_error():
    with pytest.raises(ValidationError, match=r"SNParams\.slope_k must be finite and > 0"):
        validate_sn(SNParams(slope_k="abc"))


def test_validate_sdof_rejects_non_numeric_scalar_inputs_with_validation_error():
    with pytest.raises(ValidationError, match=r"SDOFParams\.q must be finite and > 0"):
        validate_sdof(SDOFParams(q="abc", fmin=5.0, fmax=50.0, df=5.0))

    with pytest.raises(ValidationError, match=r"SDOFParams \(fmin,fmax,df\) must be finite"):
        validate_sdof(SDOFParams(q=10.0, fmin="abc", fmax=50.0, df=5.0))


def test_compute_fds_spectral_psd_rejects_mixed_sdof_grid_inputs():
    f = np.linspace(1.0, 100.0, 100)
    psd = np.full_like(f, 1e-4)
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(
        q=10.0,
        metric="pv",
        f=np.array([5.0, 10.0, 20.0, 40.0]),
        fmin=5.0,
        fmax=40.0,
        df=5.0,
    )

    with pytest.raises(ValidationError, match=r"either sdof\.f OR \(fmin, fmax, df\), not both"):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof, p_scale=1.0)


def test_compute_fds_spectral_psd_rejects_invalid_sdof_metric():
    f = np.linspace(1.0, 100.0, 100)
    psd = np.full_like(f, 1e-4)
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="force")

    with pytest.raises(ValidationError, match=r"sdof\.metric must be one of"):
        compute_fds_spectral_psd(f, psd, duration_s=60.0, sn=sn, sdof=sdof, p_scale=1.0)


def test_compute_fds_spectral_time_rejects_twosided_welch_request():
    fs = 256.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 20 * t)
    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=5.0, fmax=50.0, df=5.0, metric="pv")
    psd_params = PSDParams(method="welch", window="hann", detrend="constant", onesided=False)

    with pytest.raises(ValidationError, match=r"requires PSDParams\.onesided=True"):
        compute_fds_spectral_time(x, fs, sn=sn, sdof=sdof, psd=psd_params, p_scale=1.0)
