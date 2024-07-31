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

from distributed import Client
import dask
from dask.diagnostics import ProgressBar

import os
import sys


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
                target_sample_dict[key] = self.basic_preselection(ak.from_parquet(samples_config[key]["files"]))

            else:
                print(f"\t INFO: Loading {key} parquet files")
                reference_samples_dict[key] = self.basic_preselection(ak.from_parquet(samples_config[key]["files"]))

        return target_sample_dict, reference_samples_dict

    def basic_preselection(self, events):

        pass_tag_pt_cut = (events.tag_Ele_pt > self.tag_pt_cut)
        pass_probe_pt_cut = (events.el_pt > self.probe_pt_cut)

        pass_tag_SC_abseta = (abs(events.tag_Ele_eta + events.tag_Ele_deltaEtaSC) < self.tag_max_SC_abseta) #  have to change to SC eta
        pass_probe_SC_abseta = (abs(events.el_eta + events.el_deltaEtaSC) < self.probe_max_SC_abseta) #  have to change to SC eta

        is_in_mass_range = (
            (events.pair_mass > self.Z_mass_range[0])
            & (events.pair_mass < self.Z_mass_range[1])
        )

        events = events[
            pass_tag_pt_cut
            & pass_probe_pt_cut
            & pass_tag_SC_abseta
            & pass_probe_SC_abseta
            & is_in_mass_range
        ]

        return events

    def get_histogram_with_overflow(self, var_config, var_values, probe_region):

        #pileup weights
        #if "weight" not in events or "weight" not in failing_probes.fields:
        weight = 1

        if var_config["custom_bin_edges"] is not None:
            bins = var_config["custom_bin_edges"]
        else:
            bins = np.linspace(var_config["hist_range"][0], var_config["hist_range"][1], var_config["n_bins"]).tolist()

        x_label = f"{var_config['xlabel']} ({probe_region})"

        h = Hist(
            hist.axis.Variable(bins, label=x_label, overflow=True, underflow=True),
            storage=hist.storage.Weight(),
        )
        h.fill(var_values, weight=weight)

        return h

    def get_histograms_ratio(self):
        pass

    def plot_var_histogram(self, samples_config, var_config, var_values, probe_region, norm_val=None):
        hist = self.get_histogram_with_overflow(var_config, var_values, probe_region=probe_region)
        legend = samples_config["Legend"]
        hist_type = "step" if samples_config["isMC"] else "errorbar"

        if norm_val is not None:
            hist = hist * (norm_val / np.sum(hist.values()))

        hep.histplot(hist, label=legend, histtype=hist_type, flow="sum")

        return np.sum(hist.values())

    def create_and_save_histograms(self):

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
            for var in var_config.keys():
                print(f"\t INFO: Saving histogram for {var} for probes in {probe_region}")

                # create directory to save the histogram of the variable
                os.makedirs(f"{self.out_dir}/Variable_{var}_{probe_region}")

                hep.cms.label("Preliminary", data=False, com=13.6)
                target_sample_ = target_sample[target_key]

                if probe_region == "EB":
                    target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) < 1.4442]
                else:
                    target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) > 1.566]

                norm_val = self.plot_var_histogram(input_fileset[target_key], var_config[var], target_sample_[var], probe_region=probe_region)

                for sample in refrence_samples:
                    refrence_sample_ = refrence_samples[sample]
                    if probe_region == "EB":
                        refrence_sample_ = refrence_sample_[abs(refrence_sample_.el_eta + refrence_sample_.el_deltaEtaSC) < 1.4442]
                    else:
                        refrence_sample_ = refrence_sample_[abs(refrence_sample_.el_eta + refrence_sample_.el_deltaEtaSC) > 1.566]
                    _ = self.plot_var_histogram(input_fileset[sample], var_config[var], refrence_sample_[var], probe_region=probe_region, norm_val=norm_val)

                plt.xlim(var_config[var]["hist_range"])
                plt.legend()
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.png")
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}.pdf")
                plt.yscale("log")
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.png")
                plt.savefig(f"{self.out_dir}/Variable_{var}_{probe_region}/{var}_{probe_region}_log.pdf")
                plt.clf()

        #probe_region = "EE"
        #for var in var_config.keys():
        #    print(f"\t INFO: Saving histogram for {var} for probes in EB")
#
        #    # create directory to save the histogram of the variable
        #    os.makedirs(f"{self.out_dir}/Variable_{var}_EB")
#
        #    hep.cms.label("Preliminary", data=False, com=13.6)
        #    target_sample_ = target_sample[target_key]
        #    target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) < 1.4442]
        #    norm_val = self.plot_var_histogram(input_fileset[target_key], var_config[var], target_sample_[var], probe_region=probe_region)
#
        #    for sample in refrence_samples:
        #        refrence_sample_ = refrence_samples[sample]
        #        print(refrence_sample_)
        #        refrence_sample_ = refrence_sample_[abs(refrence_sample_.el_eta + refrence_sample_.el_deltaEtaSC) < 1.4442]
        #        print(refrence_sample_)
        #        _ = self.plot_var_histogram(input_fileset[sample], var_config[var], refrence_samples[sample][var], probe_region="EB", norm_val=norm_val)
#
        #    plt.xlim(var_config[var]["hist_range"])
        #    plt.legend()
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EB/{var}_EB.png")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EB/{var}_EB.pdf")
        #    plt.yscale("log")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EB/{var}_EB_log.png")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EB/{var}_EB_log.pdf")
        #    plt.clf()
#
        ## probes in EE
        #
        #for var in var_config.keys():
        #    
        #    print(f"\t INFO: Saving histogram for {var} for probes in EE")
        #    # create directory to save the histogram of the variable
        #    os.makedirs(f"{self.out_dir}/Variable_{var}_EE")
#
        #    hep.cms.label("Preliminary", data=False, com=13.6)
        #    target_sample_ = target_sample[target_key]
        #    target_sample_ = target_sample_[abs(target_sample_.el_eta + target_sample_.el_deltaEtaSC) > 1.566]
        #    norm_val = self.plot_var_histogram(input_fileset[target_key], var_config[var], target_sample_[var], probe_region=probe_region)
#
        #    for sample in refrence_samples:
        #        refrence_sample_ = refrence_samples[sample]
        #        refrence_sample_ = refrence_sample_[abs(refrence_sample_.el_eta + refrence_sample_.el_deltaEtaSC) > 1.566]
        #        _ = self.plot_var_histogram(input_fileset[sample], var_config[var], refrence_samples[sample][var], norm_val=norm_val, probe_region="EE")
#
        #    plt.xlim(var_config[var]["hist_range"])
        #    plt.legend()
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EE/{var}_EE.png")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EE/{var}_EE.pdf")
        #    plt.yscale("log")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EE/{var}_EE_log.png")
        #    plt.savefig(f"{self.out_dir}/Variable_{var}_EE/{var}_EE_log.pdf")
        #    plt.clf()

        return 0

    def client(self):
        with Client(n_workers=1, threads_per_worker=24) as _:

            to_compute = self.create_and_save_histograms()

            progress_bar = ProgressBar()
            progress_bar.register()

            dask.compute(to_compute)

            progress_bar.unregister()





out = Commissioning_hists(
        fileset_json="config/fileset_less.json",
        var_json="config/default_commissionig_binning.json",
        out_dir="test"
)
out.create_and_save_histograms()