import numpy as np
import pytest

from fdscore import ValidationError, synthesize_time_from_psd


def _one_sided_periodogram(x: np.ndarray, *, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    X = np.fft.rfft(x, n=n)
    f = np.fft.rfftfreq(n, d=1.0 / float(fs))

    P = (np.abs(X) ** 2) / (float(fs) * float(n))
    if (n % 2) == 0 and P.size > 2:
        P[1:-1] *= 2.0
    elif P.size > 1:
        P[1:] *= 2.0

    return f, P


def test_synthesize_time_from_psd_rejects_material_negative_psd():
    with pytest.raises(ValidationError, match="negative values below numerical tolerance"):
        synthesize_time_from_psd(
            f_psd_hz=np.array([0.0, 1.0, 2.0], dtype=float),
            psd=np.array([0.0, -1.0e-3, 1.0], dtype=float),
            fs=8.0,
            duration_s=1.0,
            seed=0,
            nfft=8,
            remove_mean=False,
        )


def test_synthesize_time_from_psd_preserves_dc_and_nyquist_for_even_nfft():
    fs = 8.0
    psd = np.array([2.0, 1.0, 1.0, 1.0, 3.0], dtype=float)
    f_psd = np.arange(psd.size, dtype=float)

    x = synthesize_time_from_psd(
        f_psd_hz=f_psd,
        psd=psd,
        fs=fs,
        duration_s=1.0,
        seed=0,
        nfft=8,
        remove_mean=False,
    )

    f_out, P_out = _one_sided_periodogram(x, fs=fs)
    assert np.allclose(f_out, f_psd)
    assert np.allclose(P_out, psd, rtol=0.0, atol=1e-12)


def test_synthesize_time_from_psd_preserves_last_positive_bin_for_odd_nfft():
    fs = 9.0
    psd = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    f_psd = np.arange(psd.size, dtype=float)

    x = synthesize_time_from_psd(
        f_psd_hz=f_psd,
        psd=psd,
        fs=fs,
        duration_s=1.0,
        seed=0,
        nfft=9,
        remove_mean=False,
    )

    f_out, P_out = _one_sided_periodogram(x, fs=fs)
    assert np.allclose(f_out, f_psd)
    assert np.allclose(P_out, psd, rtol=0.0, atol=1e-12)
