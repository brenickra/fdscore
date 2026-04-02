# Examples

These scripts are small, runnable workflows built on top of the public `fdscore` API.
Run them from the repository root with `python -m ...`.

## Available examples

- `python -m examples.minimal_fds_time`
  - time-domain FDS on two channels using a reusable `FDSTimePlan`

- `python -m examples.minimal_fds_spectral`
  - spectral FDS from an explicit PSD and from a synthesized time history
  - requires spectral support: `pip install -e .[spectral]`

- `python -m examples.minimal_inversion_and_metrics`
  - closed-form `pv` inversion and PSD summary metrics
  - shows that normalized and physical workflows produce the same inverted PSD

## Notes

- The examples use synthetic signals and PSDs so they run without external files.
- The spectral example depends on `FLife`.
- The inversion example uses the closed-form `pv` route because it is the most compact
  minimal workflow with no optional dependencies.
