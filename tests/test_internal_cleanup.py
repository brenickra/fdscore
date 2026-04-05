from fdscore.types import IterativeInversionParams


def test_iterative_inversion_params_uses_slots():
    params = IterativeInversionParams()
    assert not hasattr(params, "__dict__")
