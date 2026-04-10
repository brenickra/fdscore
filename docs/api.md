# API Reference

This page documents the stable public API exposed from the top-level
`fdscore` namespace. Internal helper modules, compatibility plumbing,
and implementation details are intentionally excluded from this
reference.

The structure below follows the public workflows described in
`README.md` and `CONTRACTS.md`.

```{eval-rst}
.. currentmodule:: fdscore
```

## Configuration Models

These classes define the main numerical settings and workflow inputs used
throughout the library.

```{eval-rst}
.. autoclass:: SNParams

.. autoclass:: SDOFParams

.. autoclass:: PSDParams

.. autoclass:: IterativeInversionParams

.. autoclass:: SineDwellSegment
```

## Result Models

These classes carry the structured outputs returned by the main fatigue,
ERS, PSD, and shock workflows.

```{eval-rst}
.. autoclass:: FDSResult

.. autoclass:: ERSResult

.. autoclass:: PSDResult

.. autoclass:: PSDMetricsResult

.. autoclass:: FDSTimePlan

.. autoclass:: ShockSpectrumPair

.. autoclass:: ShockEvent

.. autoclass:: ShockEventSet

.. autoclass:: RollingERSResult

.. autoclass:: HalfSinePulse
```

## Fatigue and Inversion Workflows

These are the main FDS computation, algebra, and inversion entry points.

```{eval-rst}
.. autofunction:: compute_fds_time

.. autofunction:: prepare_fds_time_plan

.. autofunction:: compute_fds_spectral_psd

.. autofunction:: compute_fds_spectral_time

.. autofunction:: scale_fds

.. autofunction:: sum_fds

.. autofunction:: invert_fds_closed_form

.. autofunction:: invert_fds_iterative_spectral

.. autofunction:: invert_fds_iterative_time
```

## PSD Analysis and Signal Synthesis

These functions estimate acceleration PSDs, derive scalar PSD metrics,
and synthesize stationary Gaussian realizations.

```{eval-rst}
.. autofunction:: compute_psd_welch

.. autofunction:: compute_psd_metrics

.. autofunction:: synthesize_time_from_psd
```

## Deterministic Harmonic Workflows

These APIs cover single-tone, dwell-profile, and dwell-discretized sweep
analyses.

```{eval-rst}
.. autofunction:: compute_ers_sine

.. autofunction:: compute_fds_sine

.. autofunction:: compute_ers_dwell_profile

.. autofunction:: compute_fds_dwell_profile

.. autofunction:: compute_ers_sine_sweep

.. autofunction:: compute_fds_sine_sweep
```

## ERS Workflows

These functions compute and combine generic extreme-response spectra.

```{eval-rst}
.. autofunction:: compute_ers_time

.. autofunction:: envelope_ers
```

## Shock Workflows

These functions cover transient shock analysis, event-based workflows,
rolling spectra, sided envelopes, and half-sine reduction.

```{eval-rst}
.. autofunction:: compute_srs_time

.. autofunction:: compute_pvss_time

.. autofunction:: detect_shock_events

.. autofunction:: compute_rolling_srs_time

.. autofunction:: compute_rolling_pvss_time

.. autofunction:: envelope_srs

.. autofunction:: envelope_pvss

.. autofunction:: fit_half_sine_to_pvss

.. autofunction:: synthesize_half_sine_pulse
```
