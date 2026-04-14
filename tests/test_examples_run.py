from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_minimal_fds_time_example_runs():
    runpy.run_module("examples.minimal_fds_time", run_name="__main__")


def test_minimal_inversion_and_metrics_example_runs():
    runpy.run_module("examples.minimal_inversion_and_metrics", run_name="__main__")


def test_minimal_fds_spectral_example_runs():
    runpy.run_module("examples.minimal_fds_spectral", run_name="__main__")
