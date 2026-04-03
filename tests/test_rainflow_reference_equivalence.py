import numpy as np
import pytest
import rainflow

from fdscore import synthesize_time_from_psd
from fdscore.rainflow_damage import miner_damage_from_matrix, miner_damage_from_signal


def _reference_damage(signal: np.ndarray, *, k: float, c: float, amplitude_from_range: bool) -> float:
    dmg = 0.0
    for rng, _mean, count, _i0, _i1 in rainflow.extract_cycles(np.asarray(signal, dtype=float)):
        if rng <= 0.0:
            continue
        load = 0.5 * float(rng) if amplitude_from_range else float(rng)
        if load <= 0.0:
            continue
        dmg += float(count) * (load ** float(k)) / float(c)
    return float(dmg)


def _signal_corpus(n: int = 4096) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    rng = np.random.default_rng(1234)

    f_psd = np.linspace(0.0, 200.0, 201)
    psd = (
        2.0e-4 * np.exp(-0.5 * ((f_psd - 20.0) / 8.0) ** 2)
        + 5.0e-4 * np.exp(-0.5 * ((f_psd - 75.0) / 18.0) ** 2)
        + 2.5e-4 * np.exp(-0.5 * ((f_psd - 130.0) / 20.0) ** 2)
        + 1.0e-6
    )
    psd[0] = 0.0

    return {
        "multisine": (
            0.8 * np.sin(2.0 * np.pi * 7.0 * t)
            + 0.35 * np.sin(2.0 * np.pi * 31.0 * t + 0.4)
            + 0.18 * np.sin(2.0 * np.pi * 63.0 * t + 1.3)
        ),
        "plateaus": np.repeat(np.array([0.0, 1.0, 1.0, 0.2, -0.4, -0.4, 0.8, 0.0], dtype=float), n // 8),
        "amplitude_modulated": (1.0 + 0.45 * np.sin(2.0 * np.pi * 2.0 * t)) * np.sin(2.0 * np.pi * 27.0 * t),
        "burst": (
            0.18 * np.sin(2.0 * np.pi * 12.0 * t)
            + 1.2 * np.exp(-0.5 * ((t - 0.52) / 0.035) ** 2) * np.sin(2.0 * np.pi * 70.0 * t)
        ),
        "random_walk": np.cumsum(rng.normal(loc=0.0, scale=0.025, size=n)),
        "gaussian_psd_synth": synthesize_time_from_psd(
            f_psd_hz=f_psd,
            psd=psd,
            fs=400.0,
            duration_s=n / 400.0,
            seed=77,
        ),
    }


@pytest.mark.parametrize("k", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("amplitude_from_range", [True, False])
def test_miner_damage_matches_reference_rainflow_over_signal_corpus(k: float, amplitude_from_range: bool):
    c = 1234.5
    corpus = _signal_corpus()

    miner_damage_from_signal(np.array([0.0, 1.0, 0.0, -1.0, 0.0]), k=k, c=c, amplitude_from_range=amplitude_from_range)

    for name, signal in corpus.items():
        got = miner_damage_from_signal(signal, k=k, c=c, amplitude_from_range=amplitude_from_range)
        ref = _reference_damage(signal, k=k, c=c, amplitude_from_range=amplitude_from_range)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-15), name


@pytest.mark.parametrize("amplitude_from_range", [True, False])
def test_miner_damage_matrix_matches_reference_rainflow(amplitude_from_range: bool):
    k = 6.0
    c = 98765.0
    corpus = _signal_corpus()
    names = list(corpus)
    signals = np.vstack([corpus[name] for name in names]).astype(float, copy=False)

    miner_damage_from_matrix(signals[:1], k=k, c=c, amplitude_from_range=amplitude_from_range)

    got = miner_damage_from_matrix(signals, k=k, c=c, amplitude_from_range=amplitude_from_range)
    ref = np.array(
        [
            _reference_damage(corpus[name], k=k, c=c, amplitude_from_range=amplitude_from_range)
            for name in names
        ],
        dtype=float,
    )

    assert np.allclose(got, ref, rtol=1e-12, atol=1e-15)
