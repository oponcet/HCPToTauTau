"""
Configuration of the HCPToTauTau analysis.
"""

import os
import functools

import law
import order as od
from scinum import Number
from typing import Optional

from columnflow.util import DotDict, maybe_import, dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.config_util import (
    get_root_processes_from_campaign, add_shift_aliases, get_shifts_from_sources,
    verify_config_processes,
)

from hcp.util import get_dataset_lfns

ak = maybe_import("awkward")


def add_config(
        analysis: od.Analysis,
        campaign: od.Campaign,
        config_name: Optional[str] = None,
        config_id: Optional[int] = None,
        limit_dataset_files: Optional[int] = None,
) -> od.Config:

    # get all root processes
    procs = get_root_processes_from_campaign(campaign)

    # create a config by passing the campaign, so id and name will be identical
    cfg = analysis.add_config(campaign, name=config_name, id=config_id)
    #cfg = analysis.add_config(campaign)

    # gather campaign data
    year = campaign.x.year

    # add processes we are interested in
    process_names = [
        #"data",
        "tt",
        "dy",
        "st",
        "ewk",
        "vv",
    ]
    for process_name in process_names:
        # add the process
        proc = cfg.add_process(procs.get(process_name))
        # configuration of colors, labels, etc. can happen here
        if proc.is_mc:
            proc.color1 = (244, 182, 66) if proc.name == "tt" else (244, 93, 66)

    # add datasets we need to study
    dataset_names = [
        # data
        #"data_mu_b",
        # backgrounds
        "tt_sl_powheg",
        "tt_dl_powheg",
        "dy_lep_m50_1j_madgraph",
        "dy_lep_0j_amcatnlo",
        "dy_lep_1j_amcatnlo",
        "dy_lep_2j_amcatnlo",
        "ewk_z_ll_m50_madgraph",
        "zz_pythia",
        "zz_qqll_m4_amcatnlo",
        "zz_llnunu_powheg",
        "wz_pythia",
        "wz_lllnu_amcatnlo",
        "wz_qqll_m4_amcatnlo",
        # signals
        "st_tchannel_t_powheg",
    ]
    for dataset_name in dataset_names:
        # add the dataset
        dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))
        
        # for testing purposes, limit the number of files to 2
        for info in dataset.info.values():
            info.n_files = min(info.n_files, 2)

    # verify that the root process of all datasets is part of any of the registered processes
    verify_config_processes(cfg, warn=True)

    # default objects, such as calibrator, selector, producer, ml model, inference model, etc
    cfg.x.default_calibrator = "main"
    cfg.x.default_selector = "main"
    cfg.x.default_producer = "main"
    cfg.x.default_ml_model = None
    cfg.x.default_inference_model = "main"
    cfg.x.default_categories = ("incl",)
    cfg.x.default_variables = ("n_jet", "jet1_pt")
    
    # process groups for conveniently looping over certain processs
    # (used in wrapper_factory and during plotting)
    cfg.x.process_groups = {}
    
    # dataset groups for conveniently looping over certain datasets
    # (used in wrapper_factory and during plotting)
    cfg.x.dataset_groups = {}
    
    # category groups for conveniently looping over certain categories
    # (used during plotting)
    cfg.x.category_groups = {}
    
    # variable groups for conveniently looping over certain variables
    # (used during plotting)
    cfg.x.variable_groups = {}
    
    # shift groups for conveniently looping over certain shifts
    # (used during plotting)
    cfg.x.shift_groups = {}
    
    # selector step groups for conveniently looping over certain steps
    # (used in cutflow tasks)
    cfg.x.selector_step_groups = {
        "default": ["json", "met_filter", "trigger", "lepton", "jet"],
    }

    # custom method and sandbox for determining dataset lfns
    cfg.x.get_dataset_lfns = None
    cfg.x.get_dataset_lfns_sandbox = None
    
    # whether to validate the number of obtained LFNs in GetDatasetLFNs
    # (currently set to false because the number of files per dataset is truncated to 2)
    cfg.x.validate_dataset_lfns = False
    
    # lumi values in inverse pb
    # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
    if year == 2016:
        cfg.x.luminosity = Number(36310, {
            "lumi_13TeV_2016": 0.01j,
            "lumi_13TeV_correlated": 0.006j,
        })
    elif year == 2017:
        cfg.x.luminosity = Number(41480, {
            "lumi_13TeV_2017": 0.02j,
            "lumi_13TeV_1718": 0.006j,
            "lumi_13TeV_correlated": 0.009j,
        })
    else:  # 2018
        cfg.x.luminosity = Number(59830, {
            "lumi_13TeV_2017": 0.015j,
            "lumi_13TeV_1718": 0.002j,
            "lumi_13TeV_correlated": 0.02j,
        })
        
    # names of muon correction sets and working points
    # (used in the muon producer)
    cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")
    
    # register shifts
    cfg.add_shift(name="nominal", id=0)
    
    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})
    
    # fake jet energy correction shift, with aliases flaged as "selection_dependent", i.e. the aliases
    # affect columns that might change the output of the event selection
    cfg.add_shift(name="jec_up", id=20, type="shape")
    cfg.add_shift(name="jec_down", id=21, type="shape")
    add_shift_aliases(
        cfg,
        "jec",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            "MET.pt": "MET.pt_{name}",
            "MET.phi": "MET.phi_{name}",
        },
    )
    
    # event weights due to muon scale factors
    cfg.add_shift(name="mu_up", id=10, type="shape")
    cfg.add_shift(name="mu_down", id=11, type="shape")
    add_shift_aliases(cfg, "mu", {"muon_weight": "muon_weight_{direction}"})
    
    # external files
    json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-849c6a6e"
    cfg.x.external_files = DotDict.wrap({
        # lumi files
        "lumi": {
            "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
            "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
        },
        
        # muon scale factors
        "muon_sf": (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),
    })
    
    # target file size after MergeReducedEvents in MB
    cfg.x.reduced_file_size = 512.0
    
    # columns to keep after certain steps
    cfg.x.keep_columns = DotDict.wrap({
        "cf.ReduceEvents": {
            # general event info
            "run", "luminosityBlock", "event",
            # object info
            "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.btagDeepFlavB", "Jet.hadronFlavour",
            "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.pfRelIso04_all",
            "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
            "MET.pt", "MET.phi", "MET.significance","MET.sumEt", "MET.covXX", "MET.covXY", "MET.covYY",
            "PV.npvs",
            # columns added during selection
            "deterministic_seed", "process_id", "mc_weight", "cutflow.*", "channel_id",
            "leptons_os", "leptons_ss", "single_triggered", "double_triggered",
            "m_ll", "dr_ll",
        },
        "cf.MergeSelectionMasks": {
            "normalization_weight", "process_id", "category_ids", "channel_id", "cutflow.*",
            "leptons_os", "leptons_ss", "m_ll", "dr_ll",
        },
        "cf.UniteColumns": {
            "*",
        },
    })
    
    # event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
    get_shifts = functools.partial(get_shifts_from_sources, cfg)
    cfg.x.event_weights = DotDict({
        "normalization_weight": [],
        "muon_weight": get_shifts("mu"),
    })
    
    # versions per task family, either referring to strings or to callables receving the invoking
    # task instance and parameters to be passed to the task family
    cfg.x.versions = {
        # "cf.CalibrateEvents": "prod1",
        # "cf.SelectEvents": (lambda cls, inst, params: "prod1" if params.get("selector") == "default" else "dev1"),
        # ...
    }
    
    # channels
    cfg.add_channel(name="ee", id=1)
    cfg.add_channel(name="mumu", id=2)
    
    # add categories using the "add_category" tool which adds auto-generated ids
    # the "selection" entries refer to names of selectors, e.g. in selection/example.py
    from hcp.config.categories import add_categories
    add_categories(cfg)
    
    # add variables
    # (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
    # and also correspond to the minimal set of columns that coffea's nano scheme requires)
    # add variables
    from hcp.config.variables import add_variables
    add_variables(cfg)
    
    # add triggers
    if year == 2017:
        from hcp.config.triggers import add_triggers_2017_DY
        add_triggers_2017_DY(cfg)
    else:
        raise NotImplementedError(f"triggers not implemented for {year}")
    
    # add met filters
    from hcp.config.met_filters import add_met_filters
    add_met_filters(cfg)
    
    cfg.x.get_dataset_lfns = get_dataset_lfns
    
    # define a custom sandbox
    #cfg.x.get_dataset_lfns_sandbox = dev_sandbox("bash::$CF_BASE/sandboxes/cf.sh")
    
    # define custom remote fs's to look at
    #cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: f"wlcg_fs_{cfg.campaign.x.custom['name']}"
    #cfg.x.get_dataset_lfns_remote_fs = lambda dataset_inst: f"{local_fs}"
