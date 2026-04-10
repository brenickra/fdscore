import importlib
import sys


EXTRA_NAMESPACE_ATTRS = [
    "annotations",
    "types",
    "validate",
    "grid",
    "fds_ops",
    "ers_ops",
    "shock_ops",
    "fds_time",
    "fds_spectral",
    "deterministic",
    "ers_time",
    "shock",
    "shock_events",
    "shock_rolling",
    "shock_half_sine",
    "psd_welch",
    "inverse_closed_form",
    "inverse_iterative_spectral",
    "inverse_iterative_time",
    "synth_time",
    "metrics",
    "preprocess",
    "sdof_transfer",
    "rainflow_damage",
    "_inversion_utils",
    "_psd_utils",
    "_shock_iir",
    "_time_plan",
    "_shock_signal",
]


def _fresh_import_fdscore():
    for name in list(sys.modules):
        if name == "fdscore" or name.startswith("fdscore."):
            sys.modules.pop(name)
    return importlib.import_module("fdscore")


def test_base_import_hides_undeclared_package_namespace_attrs():
    fdscore = _fresh_import_fdscore()

    missing = [name for name in fdscore.__all__ if not hasattr(fdscore, name)]
    leaked = [name for name in EXTRA_NAMESPACE_ATTRS if hasattr(fdscore, name)]

    assert missing == []
    assert leaked == []


def test_explicit_submodule_imports_still_work_after_namespace_cleanup():
    fdscore = _fresh_import_fdscore()

    assert not hasattr(fdscore, "validate")
    assert not hasattr(fdscore, "_shock_iir")

    validate_mod = importlib.import_module("fdscore.validate")
    shock_iir_mod = importlib.import_module("fdscore._shock_iir")

    assert validate_mod.ValidationError is fdscore.ValidationError
    assert hasattr(validate_mod, "validate_sdof")
    assert hasattr(shock_iir_mod, "_shock_response_spectrum_iir")
