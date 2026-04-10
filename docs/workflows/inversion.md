# Inversion Workflows

`fdscore` provides three FDS-to-PSD inversion routes: one closed-form engine
and two iterative engines.

## Closed-Form Inversion

`invert_fds_closed_form(...)` implements the Henderson-Piersol style
closed-form inversion from FDS to equivalent acceleration PSD.

```python
from fdscore import invert_fds_closed_form

psd = invert_fds_closed_form(fds, test_duration_s=24 * 3600.0)
```

This route is compact, fast, and deterministic, but it is restricted to
`metric="pv"`.

Use it when:

- your target FDS is defined in pseudo-velocity;
- the closed-form assumptions are acceptable;
- you want the simplest equivalent PSD workflow.

## Iterative Spectral Inversion

`invert_fds_iterative_spectral(...)` matches a target FDS by repeatedly
updating a candidate PSD and evaluating the predicted damage with the spectral
Dirlik route.

```python
from fdscore import IterativeInversionParams, invert_fds_iterative_spectral

params = IterativeInversionParams(iters=50, gamma=0.5)

psd = invert_fds_iterative_spectral(
    target=fds,
    f_psd_hz=f_seed,
    psd_seed=psd_seed,
    duration_s=3600.0,
    sn=sn,
    sdof=sdof,
    p_scale=1.0,
    params=params,
)
```

This route requires the optional spectral dependency because its predictor is
built on `compute_fds_spectral_psd(...)`.

## Iterative Time-Domain Inversion

`invert_fds_iterative_time(...)` also updates a candidate PSD
multiplicatively, but its predictor synthesizes Gaussian time histories and
recomputes FDS in the time domain.

```python
from fdscore import invert_fds_iterative_time

psd = invert_fds_iterative_time(
    target=fds,
    f_psd_hz=f_seed,
    psd_seed=psd_seed,
    fs=fs,
    duration_s=3600.0,
    sn=sn,
    sdof=sdof,
    p_scale=1.0,
    params=params,
    n_realizations=3,
    seed=42,
)
```

This route is attractive when the time-domain fatigue route is the main
reference and you want the predictor to include synthesis plus time-domain FDS
evaluation explicitly.

## Shared Parameter Object

Both iterative engines use `IterativeInversionParams`, but they do not consume
exactly the same subset of fields.

- the spectral engine uses the full regularization set;
- the time-domain engine uses the common update, smoothing, and prior controls
  but currently ignores tail caps and post-processing fields.

Each returned `PSDResult` exposes `meta["param_usage"]` so callers can inspect
which fields were actually consumed.

## Choosing an Engine

Use these rules of thumb:

- `invert_fds_closed_form(...)`: first choice for compact `pv` workflows.
- `invert_fds_iterative_spectral(...)`: use when spectral fatigue prediction
  is acceptable and you want a PSD-only iterative predictor.
- `invert_fds_iterative_time(...)`: use when the time-domain route is the main
  reference and you want the predictor to include synthesis plus rainflow.

## Diagnostics

All inversion routes return `PSDResult`. The metadata is intentionally rich:

- reconstruction or convergence diagnostics;
- compatibility metadata;
- parameter usage reporting;
- predictor-policy diagnostics for the time-domain iterative route.

For the mathematical derivations behind the closed-form route, see
[theory.md](../theory.md).
