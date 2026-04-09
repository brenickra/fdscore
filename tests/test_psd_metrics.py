import numpy as np
import pytest

import fdscore.metrics as metrics_mod
from fdscore import PSDResult, ValidationError, compute_psd_metrics


def test_compute_psd_metrics_constant_g2_hz():
    f = np.linspace(0.0, 100.0, 1001)
    psd = np.full_like(f, 1e-4)

    m = compute_psd_metrics(psd, f_hz=f, duration_s=3600.0, acc_unit="g")

    assert np.isclose(m.rms_acc_g, 0.1, rtol=1e-6, atol=1e-12)
    assert np.isclose(m.rms_acc_m_s2, 0.1 * 9.80665, rtol=1e-6, atol=1e-12)
    assert np.isfinite(m.peak_factor)
    assert np.isfinite(m.peak_acc_g)
    assert np.isclose(m.band_rms_g["rms_g_5_20Hz"], np.sqrt(1e-4 * 15.0), rtol=2e-3)
    assert np.isclose(m.band_rms_g["rms_g_20_80Hz"], np.sqrt(1e-4 * 60.0), rtol=2e-3)
    assert np.isclose(m.band_rms_g["rms_g_80_200Hz"], np.sqrt(1e-4 * 20.0), rtol=2e-3)
    assert np.isnan(m.band_rms_g["rms_g_200_400Hz"])


def test_compute_psd_metrics_without_duration_has_nan_peaks():
    f = np.linspace(0.0, 50.0, 501)
    psd = np.full_like(f, 2e-5)

    m = compute_psd_metrics(psd, f_hz=f, acc_unit="m/s2")

    assert np.isfinite(m.rms_acc_m_s2)
    assert np.isnan(m.peak_factor)
    assert np.isnan(m.peak_acc_m_s2)
    assert np.isnan(m.peak_vel_m_s)
    assert np.isnan(m.peak_disp_mm)
    assert np.isnan(m.disp_pk_pk_mm)


def test_compute_psd_metrics_accepts_psdresult_with_explicit_unit():
    f = np.linspace(0.0, 80.0, 801)
    psd = np.full_like(f, 5e-6)
    p = PSDResult(f=f, psd=psd, meta={})

    m = compute_psd_metrics(p, duration_s=600.0, acc_unit="m/s2")
    assert np.isfinite(m.rms_acc_m_s2)
    assert m.meta["input_acc_unit"] == "m/s2"


def test_compute_psd_metrics_requires_unit_when_not_inferable():
    f = np.linspace(0.0, 10.0, 101)
    psd = np.full_like(f, 1e-4)

    with pytest.raises(ValidationError):
        compute_psd_metrics(psd, f_hz=f, duration_s=10.0)

    p = PSDResult(f=f, psd=psd, meta={})
    with pytest.raises(ValidationError):
        compute_psd_metrics(p, duration_s=10.0)


def test_compute_psd_metrics_supports_custom_scale_factor():
    f = np.linspace(0.0, 20.0, 201)
    psd = np.full_like(f, 1.0)

    m = compute_psd_metrics(psd, f_hz=f, acc_to_m_s2=2.0)
    expected = np.sqrt(20.0 * (2.0**2))
    assert np.isclose(m.rms_acc_m_s2, expected, rtol=1e-6, atol=1e-12)

def test_integrate_trapz_falls_back_when_numpy_has_no_trapezoid(monkeypatch):
    monkeypatch.delattr(metrics_mod.np, "trapezoid", raising=False)

    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])

    area = metrics_mod._integrate_trapz(y, x)

    assert np.isclose(area, 1.0, rtol=1e-12, atol=1e-12)

def test_compute_psd_metrics_reports_partial_band_coverage_in_meta():
    f = np.linspace(0.0, 100.0, 1001)
    psd = np.full_like(f, 1e-4)

    m = compute_psd_metrics(psd, f_hz=f, duration_s=10.0, acc_unit="g")

    coverage = m.meta["band_coverage"]["rms_g_80_200Hz"]
    assert coverage["has_data"] is True
    assert coverage["partial_coverage"] is True
    assert coverage["covered_hz"] == (80.0, 100.0)
    assert np.isclose(coverage["coverage_fraction"], 20.0 / 120.0, rtol=1e-12, atol=1e-12)

    coverage_empty = m.meta["band_coverage"]["rms_g_200_400Hz"]
    assert coverage_empty["has_data"] is False
    assert coverage_empty["covered_hz"] is None


def test_compute_psd_metrics_custom_band_keys_are_stable_and_distinct():
    f = np.linspace(0.0, 30.0, 301)
    psd = np.full_like(f, 1e-4)

    m = compute_psd_metrics(psd, f_hz=f, acc_unit="g", bands_hz=[(5.9, 20.9), (5.0, 20.0)])

    assert "rms_g_5p9_20p9Hz" in m.band_rms_g
    assert "rms_g_5_20Hz" in m.band_rms_g


def test_compute_psd_metrics_exposes_effective_cycle_floor_usage():
    f = np.linspace(0.0, 1.0, 1001)
    psd = np.full_like(f, 1e-4)

    m = compute_psd_metrics(psd, f_hz=f, duration_s=0.01, acc_unit="g")

    peak_meta = m.meta["peak_statistics"]["acc"]
    assert peak_meta["n_eff_floor_applied"] is True
    assert peak_meta["effective_cycles_raw"] < np.e
    assert np.isclose(m.effective_cycles, np.e, rtol=1e-12, atol=1e-12)


def test_compute_psd_metrics_clamps_tiny_negative_numerical_noise():
    f = np.linspace(0.0, 20.0, 201)
    psd = np.full_like(f, 1e-4)
    psd[10] = -1e-15

    m = compute_psd_metrics(psd, f_hz=f, acc_unit="g")

    assert np.isfinite(m.rms_acc_g)


def test_compute_psd_metrics_rejects_material_negative_psd():
    f = np.linspace(0.0, 20.0, 201)
    psd = np.full_like(f, 1e-4)
    psd[10] = -1e-3

    with pytest.raises(ValidationError, match="PSD input"):
        compute_psd_metrics(psd, f_hz=f, acc_unit="g")

