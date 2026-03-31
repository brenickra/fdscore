# Experiments

This folder contains exploratory scripts for method comparison and diagnostics.
These scripts are auxiliary tooling and are not part of the `fdscore` package API.

## Gaussian multi-metric tester

Script:
- `gaussian_multi_metric_tester.py`

What it does:
- Synthesizes one Gaussian acceleration signal.
- Computes FDS with:
  - time-domain rainflow (`compute_fds_time`)
  - spectral Dirlik from time (`compute_fds_spectral_time`)
  - spectral Dirlik from explicit PSD (`compute_fds_spectral_psd`)
- Runs iterative spectral inversion for each metric.
- Runs closed-form inversion for `metric=\"pv\"` only.
- Computes PSD summary metrics with `compute_psd_metrics`.
- Saves CSV/PNG/TXT outputs under `_outputs_gaussian_metric_check`.

Run:
```powershell
py -3 tools\experiments\gaussian_multi_metric_tester.py
```

Useful environment overrides:
- `SHOW_PLOTS=0|1` (default `0`)
- `TEST_METRICS=pv,disp,vel,acc`
- `ITER_ITERS`, `ITER_GAMMA`, `ITER_GAIN_MIN`, `ITER_GAIN_MAX`, `ITER_SMOOTH_WIN`
- `FS_HZ`, `DURATION_S`, `TARGET_RMS_G`, `P_SCALE`
- `GAUSS_TEST_OUT_DIR=<path>`

Notes:
- Spectral paths require `FLife`.
- Percent errors can be misleading when damage values are near numerical floor.
  Prefer the log-domain metrics (`med_abs_log10`) for robust interpretation.
