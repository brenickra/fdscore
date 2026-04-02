import runpy

import pytest


def test_minimal_fds_time_example_runs():
    runpy.run_module("examples.minimal_fds_time", run_name="__main__")


def test_minimal_inversion_and_metrics_example_runs():
    runpy.run_module("examples.minimal_inversion_and_metrics", run_name="__main__")


def test_minimal_fds_spectral_example_runs_when_spectral_available():
    pytest.importorskip("FLife")
    runpy.run_module("examples.minimal_fds_spectral", run_name="__main__")
