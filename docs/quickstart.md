# Quickstart

## Basic Usage

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

print(fds.f.shape, fds.damage.shape)
print(psd_eq.f.shape, psd_eq.psd.shape)
```

## What the Output Means

`fds.damage` contains the predicted accumulated Miner damage for each natural
frequency in the SDOF oscillator bank defined by `sdof`. `psd_eq.psd`
contains the equivalent one-sided acceleration PSD that reproduces that
damage target under the assumptions of the Henderson-Piersol closed-form
inversion, on the same frequency grid as `fds.f`.
