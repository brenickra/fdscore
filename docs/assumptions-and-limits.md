# Assumptions and Limits

This page summarizes the main modeling assumptions and operational limits of
the current `fdscore` workflows.

## Time Synthesis

`synthesize_time_from_psd(...)` generates stationary Gaussian random-phase
realizations. It is appropriate for controlled studies, examples, and
iterative predictors, but it is not a general representation of arbitrary
measured non-stationary vibration environments.

## Spectral Fatigue

The spectral fatigue route uses Dirlik through `FLife`. This is a spectral
approximation, not the same algorithm as time-domain rainflow counting on a
realized signal. Absolute agreement between the two routes is therefore not
guaranteed.

`compute_fds_spectral_time(...)` adds a second modeling layer because it first
estimates the PSD with Welch and only then applies Dirlik. Its result depends
on both the PSD estimation settings and the spectral fatigue model.

## Spectral Random ERS

`compute_ers_spectral_psd(...)` and `compute_ers_spectral_time(...)` model the
ERS as an expected extreme response of a stationary Gaussian process over a
specified duration.

This route assumes:

- the PSD is an adequate descriptor of the environment;
- the response process is sufficiently close to stationary Gaussian behavior;
- the selected `duration_s` is physically meaningful for the expected maximum;
- the PSD remains representative near the top of the oscillator grid.

The spectral/random ERS is therefore not the same quantity as
`compute_ers_time(...)`, which returns the maximum observed in a realized time
history.

When a PSD has been cropped below the original Nyquist limit, the top end of
the oscillator grid can be biased low because the response moments lose part of
the high-frequency tail. The spectral ERS APIs apply an automatic edge
correction by default when `nyquist_hz` is known. This correction is intended
to mitigate PSD-export truncation, not to extrapolate arbitrary unknown
high-frequency physics.

## PSD Validity

Explicit spectral PSD inputs are expected to be non-negative. Tiny negative
values consistent with numerical noise are clamped to zero, but materially
negative PSD values raise `ValidationError`.

## Closed-Form Inversion Scope

`invert_fds_closed_form(...)` is implemented for `metric="pv"`. Other metrics
should use the iterative inversion engines.

## Closed-Form Inversion Assumptions

The Henderson-Piersol closed-form route is derived under a specific set of
assumptions that should be treated as part of the method, not as optional
background context.

The method assumes:

- stationary Gaussian excitation;
- a linear SDOF-dominated response;
- light damping, typically `zeta < 0.1` or `Q > 5`;
- an input PSD that is approximately flat over the oscillator half-power
  bandwidth;
- Rayleigh-distributed response peaks, as expected for narrowband Gaussian
  response.

For lightly damped systems this approximation is often useful and practical,
but it is still an approximation. In particular, the classical simplification
used in the closed-form derivation is tied to the light-damping regime.

## Cross-Engine Methodological Asymmetry

An important limitation is not numerical but methodological: different FDS
engines do not embed the same physical assumptions.

`compute_fds_time(...)` can preserve damage-relevant effects of the original
signal that are not fully represented by a PSD alone, including
non-stationarity, non-Rayleigh peak behavior, and temporal organization of
cycles.

If that richer time-domain FDS is then inverted with
`invert_fds_closed_form(...)`, the result is an equivalent stationary Gaussian
PSD that reproduces the same damage target under the closed-form assumptions.

This crossed workflow is often useful in practice, but it should be understood
as an engineering equivalence, not as a full reconstruction of the original
signal statistics.

For a more detailed decision framework, see
[workflows/inversion.md](workflows/inversion.md).

## `FDSTimePlan` Memory Tradeoff

`FDSTimePlan` stores the full complex FFT-domain transfer matrix `H`. This
trades memory for speed and is often worthwhile for repeated evaluations on
the same sampling setup, but it should still be treated as an explicit memory
decision rather than a free optimization.

## Generic ERS vs Shock Backend

`compute_ers_time(...)` is the generic FFT-domain ERS engine.

`compute_srs_time(...)` and `compute_pvss_time(...)` use a separate recursive
shock backend. This separation is intentional so transient shock behavior can
evolve without changing the generic ERS workflow.

## Rolling Shock Constraints

The rolling shock APIs are event-window based. They currently operate on
detected event windows rather than on a fixed-stride moving window and do not
support `peak_mode="both"`.
