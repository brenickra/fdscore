import numpy as np

from fdscore import SNParams, SDOFParams, compute_fds_time, scale_fds, sum_fds


def test_scale_fds_wraps_input_provenance_without_mutating_compat():
    fs = 256.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x = 0.1 * np.sin(2 * np.pi * 20 * t)

    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=40.0, df=10.0, metric="pv")
    fds = compute_fds_time(x, fs, sn=sn, sdof=sdof, p_scale=1.0, detrend="none", batch_size=8)

    scaled = scale_fds(fds, 3.5)

    assert np.allclose(scaled.damage, fds.damage * 3.5)
    assert scaled.meta["compat"] == fds.meta["compat"]
    assert scaled.meta["provenance"]["source"] == "scale_fds"
    assert scaled.meta["provenance"]["factor"] == 3.5
    assert scaled.meta["provenance"]["input"] == fds.meta["provenance"]
    assert fds.meta["provenance"]["source"] == "compute_fds_time"


def test_sum_fds_records_all_inputs_and_weights_in_provenance():
    fs = 256.0
    t = np.arange(0, 1.0, 1.0 / fs)
    x1 = 0.1 * np.sin(2 * np.pi * 20 * t)
    x2 = 0.08 * np.sin(2 * np.pi * 30 * t)

    sn = SNParams(slope_k=3.0)
    sdof = SDOFParams(q=10.0, fmin=10.0, fmax=40.0, df=10.0, metric="pv")
    fds1 = compute_fds_time(x1, fs, sn=sn, sdof=sdof, p_scale=1.0, detrend="none", batch_size=8)
    fds2 = compute_fds_time(x2, fs, sn=sn, sdof=sdof, p_scale=1.0, detrend="none", batch_size=8)

    summed = sum_fds([fds1, fds2], weights=[2.0, 0.0])

    assert np.allclose(summed.damage, 2.0 * fds1.damage)
    assert summed.meta["compat"] == fds1.meta["compat"]

    prov = summed.meta["provenance"]
    assert prov["source"] == "sum_fds"
    assert prov["n_inputs"] == 2
    assert prov["n_nonzero"] == 1
    assert prov["weights"] == [2.0, 0.0]
    assert len(prov["inputs"]) == 2
    assert prov["inputs"][0]["weight"] == 2.0
    assert prov["inputs"][1]["weight"] == 0.0
    assert prov["inputs"][0]["provenance"] == fds1.meta["provenance"]
    assert prov["inputs"][1]["provenance"] == fds2.meta["provenance"]
