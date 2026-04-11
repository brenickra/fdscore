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

### Closed-Form Assumptions

The closed-form route should be interpreted as an equivalent stationary-Gaussian
projection, not as a universal inverse for arbitrary measured vibration
histories.

The Henderson-Piersol formulation assumes:

- a stationary Gaussian excitation process, described in the source paper as a
  "strongly mixed random process";
- a linear SDOF-dominated response;
- light damping, typically `zeta < 0.1` or equivalently `Q > 5`;
- a PSD that is approximately flat across the half-power bandwidth of the
  oscillator, with `B_r \approx 2 \zeta f_n`;
- Rayleigh-distributed response peaks, consistent with narrowband Gaussian
  response.

These assumptions are what make the compact closed-form inversion possible. If
they are not a reasonable approximation for the intended use case, the
iterative engines are the safer choice.

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

## Methodological Consistency Between FDS Generation and Inversion

The most important methodological choice is not only which inversion engine to
use, but also how that engine relates to the way the input FDS was computed.

### Internally Consistent Gaussian Path

`compute_fds_spectral_psd(...)` together with `invert_fds_closed_form(...)`
forms an internally consistent path when the target environment is treated as
stationary and Gaussian.

In this pairing:

- the FDS is computed from a PSD through a spectral fatigue model;
- the inversion maps that damage back to an equivalent PSD under the same
  general stationary-Gaussian framework.

This is the cleanest path when the objective is test specification or compact
equivalent-environment construction under classical random-vibration
assumptions.

### Internally Consistent Time-Domain Path

`compute_fds_time(...)` together with `invert_fds_iterative_time(...)` forms
the time-domain-consistent path.

In this pairing:

- the target FDS is computed by rainflow counting on oscillator responses;
- candidate PSDs are evaluated by synthesizing time histories and recomputing
  FDS with the same time-domain fatigue route.

This path is the most methodologically aligned when the main concern is
faithfulness to the signal-level damage mechanism captured by rainflow.

### Crossed Path: Time-Domain FDS With Closed-Form Inversion

Using `compute_fds_time(...)` to generate the target FDS and then inverting it
with `invert_fds_closed_form(...)` is valid as an engineering approximation,
but it introduces an explicit methodological asymmetry.

The time-domain FDS can reflect signal characteristics that a PSD does not
carry directly, including:

- non-stationarity;
- non-Rayleigh peak statistics;
- sequence effects and temporal clustering of load cycles;
- other time-structure effects that influence accumulated damage.

The closed-form inversion then projects that richer damage target back into the
space of an equivalent stationary Gaussian process. The result is therefore the
PSD of an idealized random environment that would reproduce the same damage
under the Henderson-Piersol assumptions.

This is not a bug. It is often a deliberate and useful engineering choice
because the closed-form route is simple, fast, and practical for deriving a
test PSD from a measured damage target. But it must be interpreted as an
equivalent Gaussian test environment, not as a full statistical reconstruction
of the original signal.

Depending on the source history, this equivalent PSD may be:

- conservative for signals with heavy tails or unusually severe peak
  statistics;
- non-conservative for signals whose temporal organization reduces effective
  damage relative to a Gaussian stationary surrogate.

## Choosing an Engine

Use these rules of thumb:

- `invert_fds_closed_form(...)`: first choice for compact `pv` workflows when
  a stationary-Gaussian equivalent PSD is the desired result.
- `invert_fds_iterative_spectral(...)`: use when spectral fatigue prediction
  is acceptable and you want a PSD-only iterative predictor.
- `invert_fds_iterative_time(...)`: use when the time-domain route is the main
  reference and you want the predictor to include synthesis plus rainflow.

## Choosing the Right FDS-to-PSD Path

Use these pairings as the main decision framework:

- real-signal characterization with maximum consistency:
  `compute_fds_time(...)` + `invert_fds_iterative_time(...)`
- stationary-Gaussian specification workflow:
  `compute_fds_spectral_psd(...)` + `invert_fds_closed_form(...)`
- rapid engineering conversion from measured damage target to equivalent test
  PSD:
  `compute_fds_time(...)` + `invert_fds_closed_form(...)`

The third path is often the most practical in industry, but it should be read
as a projection into an equivalent Gaussian test space rather than as an exact
recovery of the original signal statistics.

## Diagnostics

All inversion routes return `PSDResult`. The metadata is intentionally rich:

- reconstruction or convergence diagnostics;
- compatibility metadata;
- parameter usage reporting;
- predictor-policy diagnostics for the time-domain iterative route.

For the mathematical derivations behind the closed-form route, see
[theory.md](../theory.md). For the consolidated assumptions and interpretation
limits of the inversion workflows, see
[assumptions-and-limits.md](../assumptions-and-limits.md).
