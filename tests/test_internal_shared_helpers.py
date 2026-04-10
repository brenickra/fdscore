import fdscore._shock_signal as shock_signal_mod
import fdscore._time_plan as time_plan_mod
import fdscore.ers_time as ers_time_mod
import fdscore.fds_time as fds_time_mod
import fdscore.shock as shock_mod
import fdscore.shock_events as shock_events_mod


def test_time_plan_validation_helper_lives_in_shared_internal_module():
    assert fds_time_mod.validate_time_plan_compatibility is time_plan_mod.validate_time_plan_compatibility
    assert ers_time_mod.validate_time_plan_compatibility is time_plan_mod.validate_time_plan_compatibility
    assert "_validate_plan_compatibility" not in fds_time_mod.__dict__
    assert "_validate_plan_compatibility" not in ers_time_mod.__dict__


def test_shock_preprocess_helper_lives_in_shared_internal_module():
    assert shock_mod.preprocess_shock_signal is shock_signal_mod.preprocess_shock_signal
    assert shock_events_mod.preprocess_shock_signal is shock_signal_mod.preprocess_shock_signal
    assert "_preprocess_shock_signal" not in shock_mod.__dict__
    assert "_preprocess_shock_signal" not in shock_events_mod.__dict__
