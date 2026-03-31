def test_import():
    import fdscore
    assert hasattr(fdscore, "SNParams")
    assert hasattr(fdscore, "prepare_fds_time_plan")
    assert hasattr(fdscore, "compute_psd_metrics")
