# Core Concepts

This page introduces the main parameter models, result objects, and workflow
conventions used across `fdscore`.

## Parameter Models

The public API is built around a small set of frozen dataclasses that define
fatigue, oscillator, PSD, and inversion settings.

### `SNParams`

`SNParams` defines the S-N fatigue curve used by Miner damage accumulation.

```python
from fdscore import SNParams

sn = SNParams.normalized(slope_k=6.0)
sn_physical = SNParams(slope_k=6.0, ref_stress=120.0, ref_cycles=1e6)
```

Key ideas:

- `slope_k` controls the fatigue exponent.
- `ref_stress` and `ref_cycles` define the reference point of the S-N curve.
- `amplitude_from_range` controls whether rainflow ranges are interpreted as
  `2 * amplitude` or as amplitude directly.

### `SDOFParams`

`SDOFParams` defines the oscillator bank used by FDS, ERS, SRS, and PVSS
workflows.

```python
from fdscore import SDOFParams

sdof = SDOFParams(q=10.0, metric="pv", fmin=10.0, fmax=200.0, df=10.0)
```

The main fields are:

- `q`: quality factor, with damping ratio `zeta = 1 / (2Q)`.
- `metric`: `"pv"`, `"acc"`, `"vel"`, or `"disp"`.
- frequency definition: either explicit `f` or linear `fmin`, `fmax`, `df`.

### `PSDParams`

`PSDParams` configures Welch PSD estimation and related spectral preprocessing
choices used by workflows such as `compute_psd_welch(...)` and
`compute_fds_spectral_time(...)`.

### `IterativeInversionParams`

`IterativeInversionParams` controls the multiplicative update logic used by
the iterative inversion engines. It groups convergence, smoothing, prior, and
regularization settings into one explicit object instead of scattering them
across many keyword arguments.

### `SineDwellSegment`

`SineDwellSegment` defines one harmonic dwell used by deterministic mission
profiles and dwell-discretized sine sweeps.

## Result Objects

The library returns frozen dataclasses rather than ad-hoc dictionaries. The
main public result types are:

| Object | Role |
| --- | --- |
| `FDSResult` | Miner damage spectrum over oscillator frequency |
| `ERSResult` | Extreme response spectrum for the selected metric |
| `PSDResult` | One-sided acceleration PSD plus diagnostics |
| `PSDMetricsResult` | Scalar PSD-derived metrics such as RMS and peak estimates |
| `ShockSpectrumPair` | Explicit negative/positive sided shock spectrum pair |
| `ShockEventSet` | Detected shock windows and detector metadata |
| `RollingERSResult` | Event-by-event stacked spectra |
| `HalfSinePulse` | Equivalent half-sine pulse fit |
| `FDSTimePlan` | Reusable FFT-domain transfer plan for repeated FDS calls |

## Compatibility Metadata

Most results expose a structured compatibility signature in `meta["compat"]`.
This signature is used to validate safe composition and inversion.

Examples:

- `sum_fds(...)` requires compatible FDS semantics and the same oscillator
  grid.
- `invert_fds_closed_form(...)` requires compatible fatigue semantics but does
  not require the candidate PSD grid to match the FDS grid.
- shock envelope helpers require compatible shock-spectrum semantics.

See [compatibility.md](compatibility.md) for the detailed rules.

## Normalized and Physical Workflows

`fdscore` supports two equally valid fatigue parameterizations.

### Normalized Workflow

Use a normalized S-N definition when the main interest is FDS shape,
comparison, and equivalent PSD inversion.

```python
sn = SNParams.normalized(slope_k=4.0)
```

In this case, `p_scale=1.0` is the natural default.

### Physical Workflow

Use a physical S-N definition when the absolute Miner damage level matters.

```python
sn = SNParams(slope_k=6.0, ref_stress=120.0, ref_cycles=1e6)
p_scale_physical = 300.0
```

For fixed `slope_k`, the combination of `ref_stress`, `ref_cycles`, and
`p_scale` acts as a global damage scaling factor. It changes the magnitude of
the FDS but not its relative shape.

## Reusable Transfer Plans

`FDSTimePlan` exists for repeated time-domain FDS evaluations on signals that
share the same sampling setup and oscillator definition.

```python
from fdscore import prepare_fds_time_plan

plan = prepare_fds_time_plan(fs=fs, n_samples=len(x), sdof=sdof)
```

The plan stores the full FFT-domain transfer matrix `H`, which improves reuse
performance at the cost of memory. This is usually worthwhile when evaluating
multiple channels or repeated realizations over the same grid.

## Spectrum Families in fdscore

The library uses several related but distinct spectrum concepts:

- `FDS`: accumulated fatigue damage per oscillator.
- `ERS`: extreme response spectrum for a selected response metric.
- `SRS`: shock response spectrum, implemented through a dedicated recursive
  shock backend and restricted to `sdof.metric="acc"`.
- `PVSS`: pseudo-velocity shock spectrum, also implemented through the shock
  backend and restricted to `sdof.metric="pv"`.
- `PSD`: one-sided acceleration power spectral density.

Those objects are intentionally separate because they serve different physical
purposes and obey different composition rules.
