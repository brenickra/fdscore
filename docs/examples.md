# Examples

The repository ships small runnable examples built directly on top of the
public `fdscore` API. Run them from the repository root with `python -m ...`.

## Available Examples

### Minimal Time-Domain FDS

```bash
python -m examples.minimal_fds_time
```

This example shows:

- time-domain FDS on multiple channels;
- reuse of a compatible `FDSTimePlan`;
- basic normalized-workflow usage.

### Minimal Spectral FDS

```bash
python -m examples.minimal_fds_spectral
```

This example shows:

- spectral FDS from an explicit PSD;
- spectral FDS from a synthesized time history;
- the optional `spectral` dependency path.

It requires:

```bash
pip install -e ".[spectral]"
```

### Minimal Inversion and Metrics

```bash
python -m examples.minimal_inversion_and_metrics
```

This example shows:

- closed-form `pv` inversion;
- PSD summary metrics;
- the equivalence of normalized and physical workflows for compatible
  inversion shape recovery.

## Notes

- The examples use synthetic PSDs and time histories so they run without
  external data files.
- The spectral example depends on `FLife`.
- The inversion example uses the closed-form `pv` route because it is the most
  compact minimal workflow with no optional dependencies.
