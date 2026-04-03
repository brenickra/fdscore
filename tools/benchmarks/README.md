# Benchmarks

These scripts are optional local benchmarks and are not part of the main test suite.

## Available benchmark

- `python tools/benchmarks/benchmark_rainflow_reference.py`
  - compares the external `rainflow` package against the internal Numba-backed
    rainflow/Miner implementation in `fdscore.rainflow_damage`
  - reports both scalar-per-signal and matrix-batch timings

## Notes

- Requires the development dependency `rainflow`.
- The first Numba call is warmed up before timing.
- Timings are machine-dependent and should be treated as indicative, not contractual.
