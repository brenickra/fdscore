# fdscore

**Numerical Python library for Fatigue Damage Spectrum (FDS) computation and
FDS-to-PSD inversion.**

<!--
[![PyPI version](https://img.shields.io/pypi/v/fdscore.svg)](https://pypi.org/project/fdscore/)
[![Python versions](https://img.shields.io/pypi/pyversions/fdscore.svg)](https://pypi.org/project/fdscore/)
-->
[![Documentation](https://readthedocs.org/projects/fdscore/badge/?version=latest)](https://fdscore.readthedocs.io/en/latest/)
[![CI](https://github.com/brenickra/fdscore/actions/workflows/ci.yml/badge.svg)](https://github.com/brenickra/fdscore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`fdscore` provides time-domain, spectral, deterministic, and shock-oriented
workflows for vibration fatigue analysis. The library combines FDS
computation, equivalent PSD inversion, reusable transfer plans, PSD summary
metrics, and explicit public result models intended for engineering use.

## Documentation

- User documentation: https://fdscore.readthedocs.io/en/latest/
- API reference: https://fdscore.readthedocs.io/en/latest/api.html
- Theory: https://fdscore.readthedocs.io/en/latest/theory.html
- Core concepts: https://fdscore.readthedocs.io/en/latest/concepts.html
- Public contracts: [CONTRACTS.md](CONTRACTS.md)

## Main capabilities

- Time-domain FDS using FFT-domain SDOF response reconstruction with
  rainflow/Miner damage accumulation
- Spectral FDS using Dirlik through the optional `FLife` dependency
- Closed-form FDS-to-PSD inversion for pseudo-velocity (`pv`)
- Iterative inversion with spectral and time-domain predictors
- Deterministic sine, dwell-profile, and dwell-discretized sweep workflows
- Generic ERS plus dedicated shock workflows for SRS, PVSS, event detection,
  rolling spectra, and half-sine reduction
- PSD estimation, Gaussian time synthesis, and PSD summary metrics
- Normalized and physical fatigue parameter workflows with explicit
  compatibility metadata

## Installation

Install the core package from PyPI:

```bash
pip install fdscore
```

Enable the spectral workflows as well:

```bash
pip install "fdscore[spectral]"
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np

from fdscore import (
    SNParams,
    SDOFParams,
    compute_fds_time,
    invert_fds_closed_form,
    synthesize_time_from_psd,
)

fs = 1000.0
duration_s = 12.0

f_psd = np.array([1.0, 20.0, 80.0, 150.0, 300.0])
psd = np.array([1.0e-4, 2.0e-3, 5.0e-3, 2.0e-3, 5.0e-4])

x = synthesize_time_from_psd(
    f_psd_hz=f_psd,
    psd=psd,
    fs=fs,
    duration_s=duration_s,
    seed=7,
)

sn = SNParams.normalized(slope_k=4.0)
sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)

fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, detrend="mean")
psd_eq = invert_fds_closed_form(fds, test_duration_s=duration_s)

print(fds.damage[:3])
print(psd_eq.psd[:3])
```

`fds.damage` is the predicted Miner damage spectrum on the oscillator bank
defined by `sdof`. `psd_eq.psd` is the equivalent one-sided acceleration PSD
that reproduces that damage target under the closed-form inversion
assumptions.

## Workflow overview

`fdscore` is organized around a small set of public parameter and result
models:

- `SNParams`, `SDOFParams`, `PSDParams`, and `IterativeInversionParams`
- `FDSResult`, `ERSResult`, `PSDResult`, `PSDMetricsResult`, `FDSTimePlan`
- `ShockSpectrumPair`, `ShockEventSet`, `RollingERSResult`, and
  `HalfSinePulse`

The public API is documented in the RTD site and exposed from the top-level
`fdscore` namespace. Internal helper modules are intentionally kept out of the
public reference.

## Examples

Runnable examples are available in [examples/README.md](examples/README.md):

- `python -m examples.minimal_fds_time`
- `python -m examples.minimal_fds_spectral`
- `python -m examples.minimal_inversion_and_metrics`

## Scope

Use the README as the entry point. Use the RTD site for the complete user
guide, API reference, theory notes, assumptions, compatibility rules, and
examples. Use [CONTRACTS.md](CONTRACTS.md) for the stable engineering contract
behind the public API.

## License

MIT License. See [LICENSE](LICENSE) for details.
