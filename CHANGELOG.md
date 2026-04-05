# Changelog

## Unreleased
- Restored PSD metrics integration compatibility with NumPy 1.x while keeping `numpy>=1.24` as the supported floor
- Normalized legacy S-N compatibility metadata in closed-form inversion so all inversion routes accept the same serialized FDS signatures
- Removed the unused frequency-grid argument from the internal compatibility metadata builder and documented the distinct semantics of FDS algebra vs inversion compatibility
- Relaxed inversion compatibility checks for `q` and `p_scale` to tolerate tiny numeric drift while still rejecting material mismatches
- Accepted NumPy integer scalar types in `prepare_fds_time_plan(...)` for `n_samples`
- Relaxed `FDSTimePlan` zeta compatibility tolerance and improved plan grid mismatch diagnostics around Nyquist clipping behavior
- Exposed PSD metrics band coverage and Gaussian effective-cycle floor diagnostics in metadata while stabilizing custom band key naming
- Hardened spectral PSD sanitization by clamping only tiny negative numerical noise, rejecting materially negative PSD inputs, and turning invalid FLife life values into explicit `ValidationError`s
- Strengthened iterative inversion diagnostics with explicit stage metadata, effective smoothing-window reporting, and a non-recursive post-refine guard
- Improved FDS algebra provenance so scaled and summed spectra retain structured input provenance instead of only the first-result trail

## 0.2.4
- Added `rainflow` to development dependencies so CI installs the external reference backend for those tests
- Added rainflow reference-equivalence tests against the external `rainflow` package
- Added a local benchmark script for comparing external `rainflow` against the internal Numba-backed rainflow/Miner core

## 0.2.3
- Clarified public method assumptions and limits for Gaussian time synthesis, Dirlik vs rainflow, closed-form `pv` scope, and `FDSTimePlan` memory cost

## 0.2.2
- Added runnable minimal workflow examples under `examples/` for time-domain FDS, spectral FDS, and closed-form inversion with PSD metrics

## 0.2.1
- Factored shared smoothing, blending, and taper helpers out of the iterative inversion engines into a common internal module
- Documented per-engine `IterativeInversionParams` usage and exposed it in `PSDResult.meta["param_usage"]`

## 0.2.0
- Added `SNParams.normalized(...)` for explicit normalized workflows (`k` with unit references)
- Updated `SNParams` defaults to `ref_stress=1` and `ref_cycles=1`, making `SNParams(slope_k=...)` a normalized definition by default
- Removed the implicit legacy `p_scale=6500` behavior from public FDS APIs
- `compute_fds_time(...)` and `compute_fds_spectral_*` now assume `p_scale=1` only for normalized S-N definitions and require explicit `p_scale` for physical S-N workflows
- Clarified the role of `p_scale`, `ref_stress`, and `ref_cycles` in README, contracts, and public docstrings
- Documented that global damage scaling affects FDS magnitude but not the equivalent inverted PSD in compatible workflows

## 0.1.12
- Prepared repository metadata for public release
- Added root `.gitignore` and MIT `LICENSE`
- Updated package metadata with author and classifiers
- Added optional `spectral` dependency extra for `FLife`
- Reworked public documentation and docstrings to remove project-internal wording

## 0.1.11
- Simplified `compute_psd_metrics(...)` unit handling: acceleration unit must now be provided explicitly via `acc_unit` or `acc_to_m_s2`
- Removed implicit unit inference from `PSDResult.meta` in PSD metrics
- Updated tests and contracts to enforce explicit-unit behavior

## 0.1.10
- Added `compute_psd_metrics(...)` for PSD summary metrics (RMS, Gaussian peak, velocity/displacement, and band-limited RMS)
- Added `PSDMetricsResult` dataclass to represent metric outputs
- Added test coverage for unit handling, peak behavior, and band metrics
- Updated README and contracts with PSD metrics API

## 0.1.9
- Added `FDSTimePlan` and `prepare_fds_time_plan(...)` to reuse precomputed transfer matrices across repeated `compute_fds_time(...)` calls
- Added optional `plan` argument to `compute_fds_time(...)` (backward-compatible)
- Removed experimental native C rainflow backend and related tests; time-domain rainflow path is Numba-only
- Removed unused `rainflow` runtime dependency from package metadata

## 0.1.8
- Added time-domain iterative PSD inversion (`invert_fds_iterative_time`) using random-phase synthesis + `compute_fds_time`
- Added PSD->time synthesis helper (`synthesize_time_from_psd`)

## 0.1.7
- Added iterative PSD inversion (spectral) with regularization (`invert_fds_iterative_spectral`)
- Added `IterativeInversionParams` knob dataclass

## 0.1.6
- Added spectral FDS core via Dirlik (requires `FLife`)
- Added `compute_psd_welch` helper and `compute_fds_spectral_*` APIs
- Added PSD-domain SDOF transfer builder for metrics disp/vel/acc/pv

## 0.1.5
- Added `CONTRACTS.md` documenting the public API and non-goals
- Clarified README and module docstrings (no changes to numerical results)

## 0.1.4
- Added support for metrics `disp/vel/acc` (derived); kept `pv` canonical





