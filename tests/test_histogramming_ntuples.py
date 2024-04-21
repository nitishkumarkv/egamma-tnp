import os

import dask_awkward as dak
import numpy as np
import uproot

from egamma_tnp import TagNProbeFromNTuples
from egamma_tnp.utils import (
    fill_nd_cutncount_histograms,
    fill_nd_mll_histograms,
    fill_pt_eta_phi_cutncount_histograms,
    fill_pt_eta_phi_mll_histograms,
)

fileset = {"sample": {"files": {os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree"}}}


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def test_histogramming_funcs_custom_vars():
    import egamma_tnp

    events = uproot.dask({os.path.abspath("tests/samples/TnPNTuples.root"): "fitter_tree"})

    passing_probe_evens = events[events.passHltEle30WPTightGsf == 1]
    failing_probe_evens = events[events.passHltEle30WPTightGsf == 0]

    passing_probes = dak.zip(
        {
            "pt": passing_probe_evens.el_pt,
            "eta": passing_probe_evens.el_eta,
            "phi": passing_probe_evens.el_phi,
            "r9": passing_probe_evens.el_r9,
            "pair_mass": passing_probe_evens.pair_mass,
        }
    ).compute()
    failing_probes = dak.zip(
        {
            "pt": failing_probe_evens.el_pt,
            "eta": failing_probe_evens.el_eta,
            "phi": failing_probe_evens.el_phi,
            "r9": failing_probe_evens.el_r9,
            "pair_mass": failing_probe_evens.pair_mass,
        }
    ).compute()

    egamma_tnp.config.set("r9bins", np.linspace(0.1, 1.05, 100).tolist())

    hcnc1d = fill_pt_eta_phi_cutncount_histograms(
        passing_probes,
        failing_probes,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        delayed=False,
    )
    hmll1d = fill_pt_eta_phi_mll_histograms(
        passing_probes,
        failing_probes,
        delayed=False,
    )
    hcnc3d = fill_nd_cutncount_histograms(
        passing_probes,
        failing_probes,
        vars=["eta", "r9"],
        delayed=False,
    )
    hmll3d = fill_nd_mll_histograms(
        passing_probes,
        failing_probes,
        vars=["eta", "r9"],
        delayed=False,
    )

    assert_histograms_equal(hcnc1d["eta"]["entire"]["passing"], hcnc3d["passing"][-2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hcnc1d["eta"]["entire"]["failing"], hcnc3d["failing"][-2.5j:2.5j, sum], flow=False)

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, :], flow=False)

    assert_histograms_equal(hcnc1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, sum], flow=False)
    assert_histograms_equal(hcnc1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, sum], flow=False)

    assert_histograms_equal(hcnc1d["eta"]["entire"]["passing"], hmll1d["eta"]["entire"]["passing"][:, sum], flow=False)
    assert_histograms_equal(hcnc1d["eta"]["entire"]["failing"], hmll1d["eta"]["entire"]["failing"][:, sum], flow=False)

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"][:, sum], hcnc3d["passing"][-2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"][:, sum], hcnc3d["failing"][-2.5j:2.5j, sum], flow=False)

    assert_histograms_equal(hcnc3d["passing"], hmll3d["passing"][..., sum], flow=False)
    assert_histograms_equal(hcnc3d["failing"], hmll3d["failing"][..., sum], flow=False)


def test_histogramming_default_vars():
    tag_n_probe = TagNProbeFromNTuples(
        fileset,
        "passHltEle30WPTightGsf",
        cutbased_id="passingCutBasedTight122XV1",
        use_sc_eta=True,
        tags_pt_cut=30,
    )

    hcnc1d = tag_n_probe.get_tnp_histograms(
        cut_and_count=True,
        pt_eta_phi_1d=True,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        compute=True,
    )["sample"]
    hmll1d = tag_n_probe.get_tnp_histograms(
        cut_and_count=False,
        pt_eta_phi_1d=True,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=35,
        compute=True,
    )["sample"]
    hcnc3d = tag_n_probe.get_tnp_histograms(
        cut_and_count=True,
        pt_eta_phi_1d=False,
        compute=True,
    )["sample"]
    hmll3d = tag_n_probe.get_tnp_histograms(
        cut_and_count=False,
        pt_eta_phi_1d=False,
        compute=True,
    )["sample"]

    assert_histograms_equal(hcnc1d["pt"]["barrel"]["passing"], hcnc3d["passing"][:, -1.4442j:1.4442j:sum, sum], flow=False)
    assert_histograms_equal(hcnc1d["pt"]["barrel"]["failing"], hcnc3d["failing"][:, -1.4442j:1.4442j:sum, sum], flow=False)
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_loweta"]["passing"], hcnc3d["passing"][:, -2j:-1.566j:sum, sum] + hcnc3d["passing"][:, 1.566j:2j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_loweta"]["failing"], hcnc3d["failing"][:, -2j:-1.566j:sum, sum] + hcnc3d["failing"][:, 1.566j:2j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_higheta"]["passing"], hcnc3d["passing"][:, -2.5j:-2j:sum, sum] + hcnc3d["passing"][:, 2j:2.5j:sum, sum], flow=False
    )
    assert_histograms_equal(
        hcnc1d["pt"]["endcap_higheta"]["failing"], hcnc3d["failing"][:, -2.5j:-2j:sum, sum] + hcnc3d["failing"][:, 2j:2.5j:sum, sum], flow=False
    )
    assert_histograms_equal(hcnc1d["eta"]["entire"]["passing"], hcnc3d["passing"][35j::sum, -2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hcnc1d["eta"]["entire"]["failing"], hcnc3d["failing"][35j::sum, -2.5j:2.5j, sum], flow=False)
    assert_histograms_equal(hcnc1d["phi"]["entire"]["passing"], hcnc3d["passing"][35j::sum, -2.5j:2.5j:sum, :], flow=False)
    assert_histograms_equal(hcnc1d["phi"]["entire"]["failing"], hcnc3d["failing"][35j::sum, -2.5j:2.5j:sum, :], flow=False)

    assert_histograms_equal(hmll1d["pt"]["barrel"]["passing"], hmll3d["passing"][:, -1.4442j:1.4442j:sum, sum, :], flow=False)
    assert_histograms_equal(hmll1d["pt"]["barrel"]["failing"], hmll3d["failing"][:, -1.4442j:1.4442j:sum, sum, :], flow=False)
    assert_histograms_equal(
        hmll1d["pt"]["endcap_loweta"]["passing"], hmll3d["passing"][:, -2j:-1.566j:sum, sum, :] + hmll3d["passing"][:, 1.566j:2j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_loweta"]["failing"], hmll3d["failing"][:, -2j:-1.566j:sum, sum, :] + hmll3d["failing"][:, 1.566j:2j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_higheta"]["passing"], hmll3d["passing"][:, -2.5j:-2j:sum, sum, :] + hmll3d["passing"][:, 2j:2.5j:sum, sum, :], flow=False
    )
    assert_histograms_equal(
        hmll1d["pt"]["endcap_higheta"]["failing"], hmll3d["failing"][:, -2.5j:-2j:sum, sum, :] + hmll3d["failing"][:, 2j:2.5j:sum, sum, :], flow=False
    )
    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][35j::sum, -2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][35j::sum, -2.5j:2.5j, sum, :], flow=False)
    assert_histograms_equal(hmll1d["phi"]["entire"]["passing"], hmll3d["passing"][35j::sum, -2.5j:2.5j:sum, :, :], flow=False)
    assert_histograms_equal(hmll1d["phi"]["entire"]["failing"], hmll3d["failing"][35j::sum, -2.5j:2.5j:sum, :, :], flow=False)


def test_histogramming_custom_vars():
    import egamma_tnp

    tag_n_probe = TagNProbeFromNTuples(
        fileset,
        "passHltEle30WPTightGsf",
        cutbased_id="passingCutBasedTight122XV1",
        use_sc_eta=True,
        tags_pt_cut=30,
    )

    egamma_tnp.config.set("r9bins", np.linspace(0.1, 1.05, 100).tolist())

    hmll1d = tag_n_probe.get_tnp_histograms(
        cut_and_count=False,
        pt_eta_phi_1d=True,
        eta_regions_pt={
            "barrel": [0.0, 1.4442],
            "endcap_loweta": [1.566, 2.0],
            "endcap_higheta": [2.0, 2.5],
        },
        plateau_cut=0,
        compute=True,
    )["sample"]

    hmll3d = tag_n_probe.get_tnp_histograms(
        cut_and_count=False,
        vars=["eta", "r9"],
        pt_eta_phi_1d=False,
        compute=True,
    )["sample"]

    assert_histograms_equal(hmll1d["eta"]["entire"]["passing"], hmll3d["passing"][-2.5j:2.5j, sum, :], flow=True)
    assert_histograms_equal(hmll1d["eta"]["entire"]["failing"], hmll3d["failing"][-2.5j:2.5j, sum, :], flow=True)