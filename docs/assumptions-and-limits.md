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

## PSD Validity

Explicit spectral PSD inputs are expected to be non-negative. Tiny negative
values consistent with numerical noise are clamped to zero, but materially
negative PSD values raise `ValidationError`.

## Closed-Form Inversion Scope

`invert_fds_closed_form(...)` is implemented for `metric="pv"`. Other metrics
should use the iterative inversion engines.

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
