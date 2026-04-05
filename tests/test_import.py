def test_import():
    import fdscore
    assert hasattr(fdscore, "SNParams")
    assert hasattr(fdscore, "ERSResult")
    assert hasattr(fdscore, "prepare_fds_time_plan")
    assert hasattr(fdscore, "compute_psd_metrics")
    assert hasattr(fdscore, "compute_ers_sine")
    assert hasattr(fdscore, "compute_fds_dwell_profile")
    assert hasattr(fdscore, "envelope_ers")
