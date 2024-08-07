from __future__ import annotations

import dask_awkward as dak
import awkward as ak
import hist
#from hist.dask import Hist
from hist import Hist
import numpy as np
import json
import mplhep as hep
import matplotlib.pyplot as plt

from coffea.lumi_tools import LumiMask

from distributed import Client
import dask
from dask.diagnostics import ProgressBar

import urllib.request

import os
import sys

from egamma_tnp.utils.pileup import load_correction, create_correction, get_pileup_weight


class Commissioning_hists:
    def __init__(
        self,
        fileset_json,
        var_json,
        out_dir,
        tag_pt_cut=35,
        probe_pt_cut=10,
        tag_max_SC_abseta=1.4442,
        probe_max_SC_abseta=2.5,
        Z_mass_range=[80, 100],
        cutbased_id=None,
        tag_sc_abseta_range=None,
        probe_abseta_range=None,
        extra_tags_mask=None,
        extra_probes_mask=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        self.fileset_json = fileset_json
        self.var_json = var_json
        self.out_dir = out_dir
        self.tag_pt_cut = tag_pt_cut
        self.probe_pt_cut = probe_pt_cut
        self.tag_max_SC_abseta = tag_max_SC_abseta
        self.probe_max_SC_abseta = probe_max_SC_abseta
        self.Z_mass_range = Z_mass_range
        self.cutbased_id = cutbased_id
        self.tag_sc_abseta_range = tag_sc_abseta_range
        self.probe_abseta_range = probe_abseta_range
        self.extra_tags_mask = extra_tags_mask
        self.extra_probes_mask = extra_probes_mask
        self.extra_filter = extra_filter
        self.extra_filter_args = extra_filter_args

        self.data_color_list = ["black", "brown", "saddlebrown"]
        self.MC_color_list = ["deepskyblue", "limegreen", "hotpink"]

        self.MC2024_pileup = ak.Array([0.000001113213989874036, 0.0000015468091923934005, 0.0000021354450190071103, 0.0000029291338469218304, 0.000003992042749954428, 0.000005405833377829933, 0.0000072735834977346375, 0.000009724331962846662, 0.000012918274216272392, 0.0000170526149622419, 0.000022368057920838783, 0.000029155879571161017, 0.000037765494920067983, 0.000048612379683335426, 0.00006218616667282416, 0.0000790586873656787, 0.00009989168626752147, 0.00012544390044306957, 0.00015657717511638106, 0.0001942612850563351, 0.00023957715777935895, 0.0002937182560760011, 0.0003579899817243151, 0.00043380711678651587, 0.000522689529793492, 0.0006262566455876264, 0.0007462215105090346, 0.0008843856747819444, 0.0010426365496294917, 0.0012229493551649353, 0.0014273962186958265, 0.0016581653538683676, 0.0019175934660088899, 0.002208214475733204, 0.0025328271885712, 0.002894583494129698, 0.0032970968725650703, 0.003744568249914057, 0.0042419224608759715, 0.004794943739086138, 0.005410392925019933, 0.006096082865554651, 0.0068608824777247265, 0.007714615180320339, 0.00866781516074428, 0.009731306710317893, 0.010915579065169637, 0.012229942947193852, 0.013681475759704227, 0.01527378959087405, 0.017005687963005053, 0.01886981038809762, 0.020851393740706936, 0.022927300903420315, 0.02506547465186719, 0.027224963763031827, 0.029356636191662466, 0.03140464101794246, 0.033308610261160686, 0.0350065105276433, 0.03643797256026551, 0.03754785536879911, 0.038289751851088086, 0.03862912367701764, 0.038545769885154825, 0.0380353863028649, 0.037110056331124616, 0.03579761783198069, 0.03413996250306416, 0.032190428693587354, 0.030010532709785615, 0.027666337121796485, 0.02522477201906586, 0.02275020654952582, 0.0203015184549408, 0.017929837607910185, 0.01567705686218363, 0.013575121200719511, 0.011646034074885066, 0.009902465083581686, 0.00834880944787664, 0.006982537399290253, 0.005795678071150336, 0.004776303267056856, 0.003909906169865526, 0.0031806032900314855, 0.0025721201685428422, 0.002068549223952546, 0.0016548897331626839, 0.001317394620618747, 0.0010437568832815229, 0.0008231711610503282, 0.0006463045706591073, 0.0005052068974853441, 0.0003931848671938052, 0.00030465950668724277, 0.00023502024949971086, 0.00018048485172410123, 0.00013797053104661156, 0.00010497902395277662])

    def load_json(self, json_path):
        with open(json_path, 'r') as file:
            fileset_json = json.load(file)

        return fileset_json

    def load_samples(self, samples_config):
        num_target = 0
        target_sample_dict = {}
        reference_samples_dict = {}

        for key in samples_config.keys():
            if samples_config[key]["is_target"]:
                if num_target > 0:
                    print("There are more than one target samples. Exiting now...")
                    sys.exit()

                num_target += 1
                print(f"\t INFO: Loading {key} parquet files")
                target_sample_dict[key] = self.basic_preselection(samples_config[key])

            else:
                print(f"\t INFO: Loading {key} parquet files")
                reference_samples_dict[key] = self.basic_preselection(samples_config[key])

        return target_sample_dict, reference_samples_dict

    def fetch_golden_json(self, goldenjson_url):

        # Local path to save the JSON file
        split_list = goldenjson_url.split("/")
        local_file_dir = "/".join(split_list[-3:-1])
        local_file_path = "/".join(split_list[-3:])

        if not os.path.exists(local_file_path):
            print(f"\t INFO: Fetching goldenjson from {goldenjson_url}")
            os.makedirs(local_file_dir, exist_ok=True)

            try:
                # Fetch the JSON data from the URL
                with urllib.request.urlopen(goldenjson_url) as response:
                    data = response.read().decode('utf-8')

                # Parse the JSON data
                json_data = json.loads(data)

                # Save the JSON data to a local file
                with open(local_file_path, 'w') as file:
                    json.dump(json_data, file, indent=4)

                print(f"\t JSON data saved to {local_file_path} \n")

            except urllib.error.URLError as e:
                print(f"Error fetching data from {goldenjson_url}: {e}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON data: {e}")
            except IOError as e:
                print(f"Error saving JSON to file: {e}")

        return local_file_path

    def basic_preselection(self, sample_config):

        events = ak.from_parquet(sample_config["files"])

        if not sample_config["isMC"]:

            if "https" in sample_config["goldenjson"]:  # load and save golden json, if from a url
                sample_config["goldenjson"] = self.fetch_golden_json(sample_config["goldenjson"])

            lumimask = LumiMask(sample_config["goldenjson"])
            mask = lumimask(events.run, events.luminosityBlock)
            events = events[mask]

        pass_loose_cutBased = (events["cutBased >= 1"])

        pass_tag_pt_cut = (events.tag_Ele_pt > self.tag_pt_cut)
        pass_probe_pt_cut = (events.el_pt > self.probe_pt_cut)

        pass_tag_SC_abseta = (abs(events.tag_Ele_eta + events.tag_Ele_deltaEtaSC) < self.tag_max_SC_abseta) #  have to change to SC eta
        pass_probe_SC_abseta = (abs(events.el_eta + events.el_deltaEtaSC) < self.probe_max_SC_abseta) #  have to change to SC eta

        is_in_mass_range = (
            (events.pair_mass > self.Z_mass_range[0])
            & (events.pair_mass < self.Z_mass_range[1])
        )

        events = events[
            pass_loose_cutBased
            & pass_tag_pt_cut
            & pass_probe_pt_cut
            & pass_tag_SC_abseta
            & pass_probe_SC_abseta
            & is_in_mass_range
        ]

        return events

    def get_histogram_with_overflow(self, var_config, var_values, probe_region, weight):

        if var_config["custom_bin_edges"] is not None:
            bins = var_config["custom_bin_edges"]
        else:
            bins = np.linspace(var_config["hist_range"][0], var_config["hist_range"][1], var_config["n_bins"]).tolist()

        x_label = f"{var_config['xlabel']} ({probe_region})"

        h = Hist(
            hist.axis.Variable(bins, label=x_label, overflow=True, underflow=True),
            # storage=hist.storage.Weight(),
        )
        h.fill(var_values, weight=weight)

        return h

    def get_histograms_ratio(self, hist_target, hist_reference):
        return hist_target/hist_reference

    def plot_var_histogram(self, samples_config, var_config, var_values, ax, probe_region, color, is_mass_var=False, norm_val=None, pileup_weight=None):

        if samples_config["isMC"]:
            hist_type = "step"
            w2method = "sqrt"

        else:
            hist_type = "errorbar"
            w2method = "poisson"

        weight = pileup_weight if pileup_weight is not None else 1

        hist = self.get_histogram_with_overflow(var_config, var_values, probe_region=probe_region, weight=weight)

        # hist_type = "step" if samples_config["isMC"] else "errorbar"
        legend = samples_config["Legend"]

        if norm_val is not None:
            hist = hist * (norm_val / hist.sum())

        hep.histplot(hist, label=legend, histtype=hist_type, yerr=True, w2method=w2method, flow="sum", ax=ax[0], color=color)

        return hist, hist.sum()

    def plot_ratio(self, samples_config, hist_target, hist_reference, ax, color):
        hist = self.get_histograms_ratio(hist_target, hist_reference)
        #hist_type = "step" if samples_config["isMC"] else "errorbar"
        hist_type = "errorbar"

        hep.histplot(hist, histtype=hist_type, flow="sum", ax=ax[1], color=color, yerr=False)

        return 0

    def create_and_save_histograms(self):

        print("len: ", len(self.MC2024_pileup))

        # create directory to save the histograms
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        else:
            sys.exit(f"\t The directory '{self.out_dir}' already exists. Exiting now to avoid overwriting existing plots.")

        var_config = self.load_json(self.var_json)
        input_fileset = self.load_json(self.fileset_json)
        target_sample, refrence_samples = self.load_samples(input_fileset)

        plt.style.use(hep.style.CMS)

        (target_key, ) = target_sample

        for probe_region in ["EB", "EE"]:

            target_sample_ = target_sample[target_key]

            refrence_samples_ = {}
            if probe_region == "EB":
                target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) < 1.4442]
                for sample in refrence_samples:
                    reference_sample_ = refrence_samples[sample]
                    refrence_samples_[sample] = reference_sample_[abs(reference_sample_.el_eta + reference_sample_.el_deltaEtaSC) < 1.4442]
            else:
                target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) > 1.566]
                for sample in refrence_samples:
                    reference_sample_ = refrence_samples[sample]
                    refrence_samples_[sample] = reference_sample_[abs(reference_sample_.el_eta + reference_sample_.el_deltaEtaSC) > 1.566]

            for var in var_config.keys():
                print(f"\t INFO: Saving histogram for '{var}' for probes in {probe_region}")

                # create directory to save the histogram of the variable
                os.makedirs(f"{self.out_dir}/Variable_{var}_{probe_region}")

                fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True, figsize=(8, 9))
                fig.subplots_adjust(hspace=0.07)
                hep.cms.label("Preliminary", data=False, com=13.6, ax=ax[0], fontsize=15)


                # create pileup correction
                if not os.path.exists(f"{target_key}_MC2024.json"):
                    create_correction(input_fileset[target_key]["pileup_root_file"], self.MC2024_pileup, outfile=f"{target_key}_MC2024.json", normalize_pu_mc_array=True)
                pileup_corr = load_correction(f"{target_key}_MC2024.json")

                hist_target, norm_val = self.plot_var_histogram(input_fileset[target_key], var_config[var], target_sample_[var], ax, probe_region=probe_region, color="black")

                for sample in refrence_samples_:
                    refrence_sample_ = refrence_samples_[sample]

                    # get pileup weights
                    if input_fileset[sample]["isMC"]:
                        weights = get_pileup_weight(refrence_sample_.Pileup_nTrueInt, pileup_corr)
                    hist_reference_, _ = self.plot_var_histogram(input_fileset[sample], var_config[var], refrence_sample_[var], ax, probe_region=probe_region, norm_val=norm_val, color="deepskyblue")
                    _ = self.plot_ratio(input_fileset[sample], hist_target, hist_reference_, ax, color="black")
                    #hist_target.plot_ratio(hist_reference_)

                # plot reference line at y=1 for ratio plot
                ax[1].plot([40, 1500], [1, 1], ".k", linestyle="--", linewidth=1)

                ax[0].set_xlim(var_config[var]["hist_range"])
                ax[0].set_ylim([0, ax[0].set_ylim()[1] * 1.4])
                ax[0].set_xlabel("")
                ax[0].set_ylabel("Events")
                ax[0].legend()

                ax[1].set_xlim(var_config[var]["hist_range"])
                ax[1].set_ylim([0, 2])
                ax[1].set_ylabel("Data/MC")

                # save the histogram
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.pdf", dpi=300, bbox_inches="tight")

                # also save the plot in log scale
                ax[0].set_yscale("log")
                ax[0].set_ylim([0.01, ax[0].set_ylim()[1] * 100])
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.png", dpi=300, bbox_inches="tight")
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.pdf", dpi=300, bbox_inches="tight")
                plt.clf()

        return 0

    def create_and_save_histograms_with_client(self):

        #from hist.dask import Hist

        with Client(n_workers=1, threads_per_worker=24) as _:

            to_compute = self.create_and_save_histograms()

            progress_bar = ProgressBar()
            progress_bar.register()

            dask.compute(to_compute)

            progress_bar.unregister()
        return 0





out = Commissioning_hists(
        fileset_json="config/fileset_less.json",
        var_json="config/default_commissionig_binning.json",
        out_dir="test2"
)
out.create_and_save_histograms()