import numpy as np
import uproot
from hist import intervals


def get_ratio_histogram(
    passing_probes, failing_or_all_probes, denominator_type="failing"
):
    """Get the ratio (efficiency) of the passing over passing + failing probes.
    NaN values are replaced with 0.

    Parameters
    ----------
        passing_probes : hist.Hist
            The histogram of the passing probes.
        failing_or_all_probes : hist.Hist
            The histogram of the failing or passing + failing probes.
        denominator_type : str, optional
            The type of the denominator histogram.
            Can be either "failing" or "all".
            The default is "failing".

    Returns
    -------
        ratio : hist.Hist
            The ratio histogram.
        yerr : numpy.ndarray
            The y error of the ratio histogram.
    """
    if denominator_type == "failing":
        all_probes = passing_probes + failing_or_all_probes
    elif denominator_type == "all":
        all_probes = failing_or_all_probes
    else:
        raise ValueError("Invalid denominator type. Must be either 'failing' or 'all'.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = passing_probes / all_probes
    ratio[:] = np.nan_to_num(ratio.values())
    yerr = intervals.ratio_uncertainty(
        passing_probes.values(), all_probes.values(), uncertainty_type="efficiency"
    )

    return ratio, yerr


def fill_1d_cutncount_histograms(
    passing_probes,
    failing_probes,
    plateau_cut=None,
    eta_regions_pt=None,
    eta_regions_eta=None,
    eta_regions_phi=None,
    delayed=True,
):
    """Get the Pt, Eta and Phi histograms of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the failing probes.
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions_pt : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Pt histograms will be split into those eta regions.
            The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
        eta_regions_eta : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Eta histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        eta_regions_phi : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Phi histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        delayed : bool, optional
            Whether the probes arrays are delayed (dask-awkward) or not.
            The default is True.

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"var": {"name": {"passing": passing_probes, "failing": failing_probes}, ...}, ...}`
            where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
            Each `"name"` is the name of eta region specified by the user.
            `passing_probes` and `failing_probes` are `hist.Hist` or `hist.dask.Hist` objects.
            These are the histograms of the passing and failing probes respectively.
    """
    import hist

    if delayed:
        from hist.dask import Hist
    else:
        from hist import Hist

    import egamma_tnp

    if plateau_cut is None:
        plateau_cut = 0
    if eta_regions_pt is None:
        eta_regions_pt = {
            "barrel": [0.0, 1.4442],
            "endcap": [1.566, 2.5],
        }
    if eta_regions_eta is None:
        eta_regions_eta = {"entire": [0.0, 2.5]}
    if eta_regions_phi is None:
        eta_regions_phi = {"entire": [0.0, 2.5]}

    ptbins = egamma_tnp.config.get("ptbins")
    etabins = egamma_tnp.config.get("etabins")
    phibins = egamma_tnp.config.get("phibins")

    pt_pass = passing_probes.pt
    pt_fail = failing_probes.pt
    eta_pass = passing_probes.eta
    eta_fail = failing_probes.eta
    phi_pass = passing_probes.phi
    phi_fail = failing_probes.phi

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass = pt_pass > plateau_cut
    plateau_mask_fail = pt_fail > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (
            abs(eta_pass) < region_pt[1]
        )
        eta_mask_pt_fail = (abs(eta_fail) > region_pt[0]) & (
            abs(eta_fail) < region_pt[1]
        )
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_fail = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_pass.fill(pt_pass[eta_mask_pt_pass])
        hpt_fail.fill(pt_fail[eta_mask_pt_fail])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "failing": hpt_fail}

    for name_eta, region_eta in eta_regions_eta.items():
        eta_mask_eta_pass = (abs(eta_pass) > region_eta[0]) & (
            abs(eta_pass) < region_eta[1]
        )
        eta_mask_eta_fail = (abs(eta_fail) > region_eta[0]) & (
            abs(eta_fail) < region_eta[1]
        )
        heta_pass = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            storage=hist.storage.Weight(),
        )
        heta_fail = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            storage=hist.storage.Weight(),
        )
        heta_pass.fill(eta_pass[plateau_mask_pass & eta_mask_eta_pass])
        heta_fail.fill(eta_fail[plateau_mask_fail & eta_mask_eta_fail])

        histograms["eta"][name_eta] = {"passing": heta_pass, "failing": heta_fail}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (
            abs(eta_pass) < region_phi[1]
        )
        eta_mask_phi_fail = (abs(eta_fail) > region_phi[0]) & (
            abs(eta_fail) < region_phi[1]
        )
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            storage=hist.storage.Weight(),
        )
        hphi_fail = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            storage=hist.storage.Weight(),
        )
        hphi_pass.fill(phi_pass[plateau_mask_pass & eta_mask_phi_pass])
        hphi_fail.fill(phi_fail[plateau_mask_fail & eta_mask_phi_fail])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "failing": hphi_fail}

    return histograms


def fill_2d_mll_histograms(
    passing_probes,
    failing_probes,
    plateau_cut=None,
    eta_regions_pt=None,
    eta_regions_eta=None,
    eta_regions_phi=None,
    delayed=True,
):
    """Get the 2D histograms of Pt, Eta and Phi vs mll of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the failing probes.
        plateau_cut : int or float, optional
            The Pt threshold to use to ensure that we are on the efficiency plateau for eta and phi histograms.
            The default None, meaning that no extra cut is applied and the activation region is included in those histograms.
        eta_regions_pt : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Pt histograms will be split into those eta regions.
            The default is to avoid the ECAL transition region meaning |eta| < 1.4442 or 1.566 < |eta| < 2.5.
        eta_regions_eta : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Eta histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        eta_regions_phi : dict, optional
            A dictionary of the form `{"name": [etamin, etamax], ...}`
            where name is the name of the region and etamin and etamax are the absolute eta bounds.
            The Phi histograms will be split into those eta regions.
            The default is to use the entire |eta| < 2.5 region.
        delayed : bool, optional
            Whether the probes arrays are delayed (dask-awkward) or not.
            The default is True.

    Returns
    -------
        histograms : dict
            A dictionary of the form `{"var": {"name": {"passing": passing_probes, "failing": failing_probes}, ...}, ...}`
            where `"var"` can be `"pt"`, `"eta"`, or `"phi"`.
            Each `"name"` is the name of eta region specified by the user.
            `passing_probes` and `failing_probes` are `hist.Hist` or `hist.dask.Hist` objects.
            These are the histograms of the passing and failing probes respectively.
    """
    import hist

    if delayed:
        from hist.dask import Hist
    else:
        from hist import Hist

    import egamma_tnp

    if plateau_cut is None:
        plateau_cut = 0
    if eta_regions_pt is None:
        eta_regions_pt = {
            "barrel": [0.0, 1.4442],
            "endcap": [1.566, 2.5],
        }
    if eta_regions_eta is None:
        eta_regions_eta = {"entire": [0.0, 2.5]}
    if eta_regions_phi is None:
        eta_regions_phi = {"entire": [0.0, 2.5]}

    ptbins = egamma_tnp.config.get("ptbins")
    etabins = egamma_tnp.config.get("etabins")
    phibins = egamma_tnp.config.get("phibins")

    pt_pass = passing_probes.pt
    pt_fail = failing_probes.pt
    eta_pass = passing_probes.eta
    eta_fail = failing_probes.eta
    phi_pass = passing_probes.phi
    phi_fail = failing_probes.phi
    mll_pass = passing_probes.pair_mass
    mll_fail = failing_probes.pair_mass

    histograms = {}
    histograms["pt"] = {}
    histograms["eta"] = {}
    histograms["phi"] = {}

    plateau_mask_pass = pt_pass > plateau_cut
    plateau_mask_fail = pt_fail > plateau_cut

    for name_pt, region_pt in eta_regions_pt.items():
        eta_mask_pt_pass = (abs(eta_pass) > region_pt[0]) & (
            abs(eta_pass) < region_pt[1]
        )
        eta_mask_pt_fail = (abs(eta_fail) > region_pt[0]) & (
            abs(eta_fail) < region_pt[1]
        )
        hpt_pass = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_fail = Hist(
            hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hpt_pass.fill(pt_pass[eta_mask_pt_pass], mll_pass[eta_mask_pt_pass])
        hpt_fail.fill(pt_fail[eta_mask_pt_fail], mll_fail[eta_mask_pt_fail])

        histograms["pt"][name_pt] = {"passing": hpt_pass, "failing": hpt_fail}

    for name_eta, region_eta in eta_regions_eta.items():
        eta_mask_eta_pass = (abs(eta_pass) > region_eta[0]) & (
            abs(eta_pass) < region_eta[1]
        )
        eta_mask_eta_fail = (abs(eta_fail) > region_eta[0]) & (
            abs(eta_fail) < region_eta[1]
        )
        heta_pass = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        heta_fail = Hist(
            hist.axis.Variable(etabins, name="eta", label="eta"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        eta_mask_pass = plateau_mask_pass & eta_mask_eta_pass
        eta_mask_fail = plateau_mask_fail & eta_mask_eta_fail
        heta_pass.fill(eta_pass[eta_mask_pass], mll_pass[eta_mask_pass])
        heta_fail.fill(eta_fail[eta_mask_fail], mll_fail[eta_mask_fail])

        histograms["eta"][name_eta] = {"passing": heta_pass, "failing": heta_fail}

    for name_phi, region_phi in eta_regions_phi.items():
        eta_mask_phi_pass = (abs(eta_pass) > region_phi[0]) & (
            abs(eta_pass) < region_phi[1]
        )
        eta_mask_phi_fail = (abs(eta_fail) > region_phi[0]) & (
            abs(eta_fail) < region_phi[1]
        )
        hphi_pass = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        hphi_fail = Hist(
            hist.axis.Variable(phibins, name="phi", label="phi"),
            hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
            storage=hist.storage.Weight(),
        )
        phi_mask_pass = plateau_mask_pass & eta_mask_phi_pass
        phi_mask_fail = plateau_mask_fail & eta_mask_phi_fail
        hphi_pass.fill(phi_pass[phi_mask_pass], mll_pass[phi_mask_pass])
        hphi_fail.fill(phi_fail[phi_mask_fail], mll_fail[phi_mask_fail])

        histograms["phi"][name_phi] = {"passing": hphi_pass, "failing": hphi_fail}

    return histograms


def fill_3d_cutncount_histograms(
    passing_probes,
    failing_probes,
    delayed=True,
):
    """Get the 3D (Pt, Eta, Phi) histogram of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, and `phi` of the failing probes.
        delayed : bool, optional
            Whether the probes arrays are delayed (dask-awkward) or not.
            The default is True.

    Returns
    -------
        histograms: dict
        A dict of the form {"passing": hpass, "failing": hfail} where
        hpass : hist.Hist or hist.dask.Hist
            A 3D histogram with axes (Pt, Eta, Phi) of the passing probes.
        hfail : hist.Hist
            A 3D histogram with axes (Pt, Eta, Phi) of the failing probes.
    """

    import hist

    if delayed:
        from hist.dask import Hist
    else:
        from hist import Hist

    import egamma_tnp

    ptbins = egamma_tnp.config.get("ptbins")
    etabins = egamma_tnp.config.get("etabins")
    phibins = egamma_tnp.config.get("phibins")

    hpass = Hist(
        hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
        hist.axis.Variable(etabins, name="eta", label="eta"),
        hist.axis.Variable(phibins, name="phi", label="phi"),
        storage=hist.storage.Weight(),
    )

    hfail = Hist(
        hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
        hist.axis.Variable(etabins, name="eta", label="eta"),
        hist.axis.Variable(phibins, name="phi", label="phi"),
        storage=hist.storage.Weight(),
    )

    hpass.fill(passing_probes.pt, passing_probes.eta, passing_probes.phi)
    hfail.fill(failing_probes.pt, failing_probes.eta, failing_probes.phi)

    return {"passing": hpass, "failing": hfail}


def fill_4d_mll_histograms(
    passing_probes,
    failing_probes,
    delayed=True,
):
    """Get the 4D (Pt, Eta, Phi, mll) histogram of the passing and failing probes.

    Parameters
    ----------
        passing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, `phi` and `pair_mass` of the passing probes.
        failing_probes : awkward.Array or dask_awkward.Array
            An array with fields `pt`, `eta`, `phi` and `pair_mass` of the failing probes.
        delayed : bool, optional
            Whether the probes arrays are delayed (dask-awkward) or not.
            The default is True.

    Returns
    -------
        histograms: dict
        A dict of the form {"passing": hpass, "failing": hfail} where
        hpass : hist.Hist or hist.dask.Hist
            A 4D histogram with axes (Pt, Eta, Phi, mll) of the passing probes.
        hfail : hist.Hist
            A 4D histogram with axes (Pt, Eta, Phi, mll) of the failing probes.
    """
    import hist

    if delayed:
        from hist.dask import Hist
    else:
        from hist import Hist

    import egamma_tnp

    ptbins = egamma_tnp.config.get("ptbins")
    etabins = egamma_tnp.config.get("etabins")
    phibins = egamma_tnp.config.get("phibins")

    hpass = Hist(
        hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
        hist.axis.Variable(etabins, name="eta", label="eta"),
        hist.axis.Variable(phibins, name="phi", label="phi"),
        hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
        storage=hist.storage.Weight(),
    )

    hfail = Hist(
        hist.axis.Variable(ptbins, name="pt", label="Pt [GeV]"),
        hist.axis.Variable(etabins, name="eta", label="eta"),
        hist.axis.Variable(phibins, name="phi", label="phi"),
        hist.axis.Regular(80, 50, 130, name="mll", label="mll [GeV]"),
        storage=hist.storage.Weight(),
    )

    hpass.fill(
        passing_probes.pt,
        passing_probes.eta,
        passing_probes.phi,
        passing_probes.pair_mass,
    )
    hfail.fill(
        failing_probes.pt,
        failing_probes.eta,
        failing_probes.phi,
        failing_probes.pair_mass,
    )

    return {"passing": hpass, "failing": hfail}


# Function to convert edge value to string format for keys
def _format_edge(value):
    prefix = "m" if value < 0 else ""
    formatted = f"{prefix}{abs(value):.2f}".replace(".", "p")
    return formatted


# We could have used the function right below to convert 2D histograms to 1D histograms
# with just using axes=[the name of the x axis] but I prefer this easier approach for now.
# It makes debugging easier and it is more explicit
def _convert_2d_mll_hist_to_1d_hists(h2d):
    histograms = {}
    bin_info_list = []
    ax = h2d.axes.name[0]

    for idx in range(h2d.axes[0].size):
        min_edge = h2d.axes[0].edges[idx]
        max_edge = h2d.axes[0].edges[idx + 1]
        key = f"{ax}_{_format_edge(min_edge)}To{_format_edge(max_edge)}"
        histograms[key] = h2d[{ax: idx}]
        bin_name = f"bin{idx}_{key}"
        bin_info_list.append(
            {
                "cut": f"{ax} >= {min_edge:.6f} && {ax} < {max_edge:.6f}",
                "name": bin_name,
                "vars": {ax: {"min": min_edge, "max": max_edge}},
                "title": f"; {min_edge:.3f} < {ax} < {max_edge:.3f}",
            }
        )

    bining = {"bins": bin_info_list, "vars": [ax]}

    return histograms, bining


def _convert_4d_mll_hist_to_1d_hists(h4d, axes):
    import itertools

    import hist

    # Initialize the slicer to keep specified axes and sum over the others except 'mll'
    s = hist.tag.Slicer()
    slice_dict = {
        ax.name: s[:] if ax.name in axes or ax.name == "mll" else s[sum]
        for ax in h4d.axes
    }

    # Apply slicing to obtain the relevant projection while keeping 'mll' intact
    h = h4d[slice_dict]

    # Dictionary to hold all 1D histograms
    histograms = {}
    # List to hold all dictionaries for each bin
    bin_info_list = []

    # Generate all combinations of specified axes except 'mll'
    total_bins = np.prod([h.axes[ax].size for ax in axes])
    zfill_length = len(str(total_bins))
    counter = 0
    for idx_combination in itertools.product(*(range(h.axes[ax].size) for ax in axes)):
        bin_details = {}
        vars_details = {}
        cut_parts = []
        title_parts = []

        # Construct details using the given order
        for ax, idx in zip(axes, idx_combination):
            min_edge = h.axes[ax].edges[idx]
            max_edge = h.axes[ax].edges[idx + 1]
            vars_details[ax] = {"min": min_edge, "max": max_edge}
            cut_parts.append(f"{ax} >= {min_edge:.6f} && {ax} < {max_edge:.6f}")
            title_parts.append(f"{min_edge:.3f} < {ax} < {max_edge:.3f}")

        # Key should be constructed in the order given using the _format_edge for key consistency
        key = "_".join(
            f"{ax}_{_format_edge(vars_details[ax]['min'])}To{_format_edge(vars_details[ax]['max'])}"
            for ax in axes
        )

        # Correcting slice indices using original axis ordering
        slice_indices = {ax: idx_combination[axes.index(ax)] for ax in axes}
        slice_indices["mll"] = slice(None)
        histograms[key] = h[slice_indices]

        # Create the dictionary for this bin
        bin_name = f"bin{str(counter).zfill(zfill_length)}_{key}"
        bin_details["cut"] = " && ".join(cut_parts)
        bin_details["name"] = bin_name
        bin_details["vars"] = vars_details
        bin_details["title"] = "; " + "; ".join(title_parts)

        bin_info_list.append(bin_details)
        counter += 1

    bining = {"bins": bin_info_list, "vars": axes}

    return histograms, bining


def convert_2d_mll_hists_to_1d_hists(hist_dict):
    """Convert 2D (var, mll) histogram dict to 1D histograms.
    This will create a 1D histogram for each bin in the x-axis of the 2D histograms.

    Parameters
    ----------
        hist_dict : dict
            A dictionary of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
            where each hist.Hist is a 2D histogram with axes (var, mll).
    Returns
    -------
        histograms : dict
            A dictionary of the form {"var": {"region": {"passing": {bin_name: hist.Hist, ...}, "failing": {bin_name: hist.Hist, ...}, "bining": bining}, ...}, ...}
    """
    histograms = {}  # Create a new dictionary instead of modifying the original
    for var, region_dict in hist_dict.items():
        histograms[var] = {}  # Initialize var dictionary
        for region_name, hists in region_dict.items():
            histograms[var][region_name] = {}  # Initialize region dictionary
            for histname, h in hists.items():
                hs, bining = _convert_2d_mll_hist_to_1d_hists(h)
                histograms[var][region_name][histname] = (
                    hs  # Populate with new histograms
                )
                histograms[var][region_name]["bining"] = (
                    bining  # Set bining for this region
                )
    return histograms


def convert_4d_mll_hists_to_1d_hists(hists, axes=None):
    """Convert 4D (Pt, Eta, Phi, mll) histograms to 1D histograms.
    This will create a 1D histogram for each pair of bins in the specified axes.
    If you have `npt` bins in Pt, `neta` bins in Eta, and `nphi` bins in Phi,
    then you will get `npt x neta x nphi` 1D histograms.
    If a variable is not specified in the axes, then it will be summed over.
    For instance if you specify only `pt` and `eta`, then the 1D histograms will be
    created for each pair of Pt and Eta bins, summed over all Phi bins, so `npt x neta` histograms.

    Parameters
    ----------
        hists : dict
            A dictionary of the form {"passing": hpass, "failing": hfail}
            where hpass and hfail are 4D histograms with axes (Pt, Eta, Phi, mll).
        axes : list, optional
            A list of the axes to keep in the 1D histograms.
            The default is ["pt", "eta"].

    Returns
    -------
        histograms : dict
            A dictionary of the form {"passing": hpass, "failing": hfail}
            where hpass and hfail are dictionaries of 1D histograms.
            The keys are the bin combinations of the specified axes
            and the values are the 1D histograms for each bin combination.
        bining : dict
            A dictionary with the binning information.
    """
    if axes is None:
        axes = ["pt", "eta"]
    if len(set(axes)) != len(axes):
        raise ValueError("All axes must be unique.")

    histograms = {}
    for key, h4d in hists.items():
        hists, binning = _convert_4d_mll_hist_to_1d_hists(h4d, axes=axes)
        histograms[key] = hists

    return histograms, binning


def create_hists_root_file_for_fitter(hists, root_path, bining_path, axes=None):
    """Create a ROOT file with 1D histograms of passing and failing probes.
    To be used as input to the fitter.

    Parameters
    ----------
        hists : dict
            Either a dictionary of 2D histograms of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
            or a dictionary of 4D histograms of the form {"passing": hpass, "failing": hfail}.
            where hpass and hfail are 4D histograms with axes (Pt, Eta, Phi, mll).
        root_path : str
            The path to the ROOT file.
        bining_path : str
            The path to the pickle file with the binning information.
        axes : list, optional
            A list of the axes to keep in the 1D histograms.
            The default is ["pt", "eta"].

        Notes
        -----
            If the input is a dictionary of 2D histograms, then multiple ROOT files and binning files will be created,
            one for each variable and region present in the input.
            For each variable and region present, a trailing `_<var>_<region>.root` and `_<var>_<region>.pkl` will be added to the root_path and bining_path respectively.
    """
    import pickle

    import uproot

    if axes is None:
        axes = ["pt", "eta"]
    if len(set(axes)) != len(axes):
        raise ValueError("All axes must be unique.")

    if isinstance(hists, dict) and "passing" in hists and "failing" in hists:
        histograms, bining = convert_4d_mll_hists_to_1d_hists(hists, axes=axes)
        passing_hists = histograms["passing"]
        failing_hists = histograms["failing"]

        if passing_hists.keys() != failing_hists.keys():
            raise ValueError(
                "Passing and failing histograms must have the same binning."
            )

        names = list(passing_hists.keys())
        max_number = len(str(len(names)))

        with uproot.recreate(root_path) as f:
            counter = 0
            for name in names:
                counter_str = str(counter).zfill(max_number)
                f[f"bin{counter_str}_{name}_Pass"] = passing_hists[name]
                f[f"bin{counter_str}_{name}_Fail"] = failing_hists[name]
                counter += 1

        with open(bining_path, "wb") as f:
            pickle.dump(bining, f)

    else:
        histograms = convert_2d_mll_hists_to_1d_hists(hists)
        for var, region_dict in histograms.items():
            for region_name, hists in region_dict.items():
                new_path = root_path.replace(".root", f"_{var}_{region_name}.root")
                new_bining_path = bining_path.replace(
                    ".pkl", f"_{var}_{region_name}.pkl"
                )
                with uproot.recreate(new_path) as f:
                    passing_hists = hists["passing"]
                    failing_hists = hists["failing"]
                    names = list(passing_hists.keys())
                    max_number = len(str(len(names)))
                    counter = 0
                    for name in names:
                        counter_str = str(counter).zfill(max_number)
                        f[f"bin{counter_str}_{name}_Pass"] = passing_hists[name]
                        f[f"bin{counter_str}_{name}_Fail"] = failing_hists[name]
                        counter += 1

                with open(new_bining_path, "wb") as f:
                    pickle.dump(hists["bining"], f)


def save_hists(path, res):
    """Save histograms to a ROOT file.

    Parameters
    ----------
        path : str
            The path to the ROOT file.
        res : dict
            A histogram dictionary of the form {"var": {"region": {"passing": hist.Hist, "failing": hist.Hist}, ...}, ...}
    """
    with uproot.recreate(path) as f:
        for var, region_dict in res.items():
            for region_name, hists in region_dict.items():
                for histname, h in hists.items():
                    f[f"{var}/{region_name}/{histname}"] = h
