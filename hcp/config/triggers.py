# coding: utf-8

"""
Definition of triggers
"""

import order as od

from hcp.config.trigger_util import Trigger, TriggerLeg


def add_triggers_2017(config: od.Config) -> None:
    """
    Adds all triggers to a *config*. For the conversion from filter names to trigger bits, see
    https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py.
    """
    config.x.triggers = od.UniqueObjectIndex(Trigger, [
        #
        # single electron
        #
        Trigger(
            name="HLT_Ele32_WPTight_Gsf",
            id=201,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
                    # filter names:
                    # hltEle32WPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_mc or dataset_inst.x.era >= "D"),
            tags={"single_trigger", "single_e", "channel_e_e"},
        ),
        Trigger(
            name="HLT_Ele32_WPTight_Gsf_L1DoubleEG",
            id=202,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=35.0,
                    # filter names:
                    # hltEle32L1DoubleEGWPTightGsfTrackIsoFilter
                    # hltEGL1SingleEGOrFilter
                    trigger_bits=2 + 1024,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_e"},
        ),
        Trigger(
            name="HLT_Ele35_WPTight_Gsf",
            id=203,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=38.0,
                    # filter names:
                    # hltEle35noerWPTightGsfTrackIsoFilter
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_e", "channel_e_e"},
        ),

        #
        # single muon
        #
        Trigger(
            name="HLT_IsoMu24",
            id=101,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=26.0,
                    # filter names:
                    # hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_mu"},
        ),
        Trigger(
            name="HLT_IsoMu27",
            id=102,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=29.0,
                    # filter names:
                    # hltL3crIsoL1sMu22Or25L1f0L2f10QL3f27QL3trkIsoFiltered0p07
                    trigger_bits=2,
                ),
            ],
            tags={"single_trigger", "single_mu", "channel_mu_mu"},
        ),

        #
        # e e
        #
        #Trigger(
        #    name="HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        #    id=401,
        #    legs=[
        #        TriggerLeg(
        #            pdg_id=11,
        #            min_pt=25.0,
        #            # filter names:
        #            # hltEle24erWPTightGsfTrackIsoFilterForTau
        #            # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #            trigger_bits=2 + 64,
        #        ),
        #        TriggerLeg(
        #            pdg_id=11,
        #            min_pt=15.0,
        #            # filter names:
        #            # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
        #            # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #            trigger_bits=1024 + 256,
        #        ),
        #    ],
        #    tags={"double_trigger", "double_e_e", "channel_e_e"},
        #),
        #Trigger(
        #    name="HLT_Ele27_Ele37_CaloIdL_MW",
        #    id=402,
        #    legs=[
        #        TriggerLeg(
        #            pdg_id=11,
        #            min_pt=29.0,
        #            # filter names:
        #            # hltEle24erWPTightGsfTrackIsoFilterForTau
        #            # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #            trigger_bits=2 + 64,
        #        ),
        #        TriggerLeg(
        #            pdg_id=11,
        #            min_pt=39.0,
        #            # filter names:
        #            # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
        #            # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
        #            trigger_bits=1024 + 256,
        #        ),
        #    ],
        #    tags={"double_trigger", "double_e_e", "channel_e_e"},
        #),

        #
        # mu mu
        #
        
        #Trigger(
        #    name="HLT_Mu19_TrkIsoVVL_Mu9_TrkIsoVVL_DZ",
        #    id=301,
        #    legs=[
        #        TriggerLeg(
        #            pdg_id=13,
        #            min_pt=21.0,
        #            # filter names:
        #            # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
        #            # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
        #            trigger_bits=2 + 64,
        #        ),
        #        TriggerLeg(
        #            pdg_id=13,
        #            min_pt=12.0,
        #            # filter names:
        #            # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
        #            # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
        #            trigger_bits=1024 + 512,
        #        ),
        #    ],
        #    tags={"double_trigger", "double_mu_mu", "channel_mu_mu"},
        #),
        #Trigger(
        #    name="HLT_Mu20_Mu10_DZ",
        #    id=302,
        #    legs=[
        #        TriggerLeg(
        #            pdg_id=13,
        #            min_pt=22.0,
        #            # filter names:
        #            # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
        #            # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
        #            trigger_bits=2 + 64,
        #        ),
        #        TriggerLeg(
        #            pdg_id=13,
        #            min_pt=12.0,
        #            # filter names:
        #            # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
        #            # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
        #            trigger_bits=1024 + 512,
        #        ),
        #    ],
        #    tags={"double_trigger", "double_mu_mu", "channel_mu_mu"},
        #),

        #
        # e-tauh
        #
        Trigger(
            name="HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1",
            id=501,
            legs=[
                TriggerLeg(
                    pdg_id=11,
                    min_pt=27.0,
                    # filter names:
                    # hltEle24erWPTightGsfTrackIsoFilterForTau
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=35.0,
                    # filter names:
                    # hltSelectedPFTau30LooseChargedIsolationL1HLTMatched
                    # hltOverlapFilterIsoEle24WPTightGsfLooseIsoPFTau30
                    trigger_bits=1024 + 256,
                ),
            ],
            tags={"cross_trigger", "cross_e_tau", "channel_e_tau"},
        ),

        #
        # mu-tauh
        #
        Trigger(
            name="HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1",
            id=601,
            legs=[
                TriggerLeg(
                    pdg_id=13,
                    min_pt=22.0,
                    # filter names:
                    # hltL3crIsoL1sMu18erTau24erIorMu20erTau24erL1f0L2f10QL3f20QL3trkIsoFiltered0p07
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=2 + 64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=32.0,
                    # filter names:
                    # hltSelectedPFTau27LooseChargedIsolationAgainstMuonL1HLTMatched or
                    # hltOverlapFilterIsoMu20LooseChargedIsoPFTau27L1Seeded
                    trigger_bits=1024 + 512,
                ),
            ],
            tags={"cross_trigger", "cross_mu_tau", "channel_mu_tau"},
        ),

        #
        # tauh-tauh
        #
        Trigger(
            name="HLT_DoubleMediumChargedIsoPFTau35_Trk1_eta2p1_Reg",
            id=701,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg",
            id=702,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=40.0,
                    # filter names:
                    # hltDoublePFTau35TrackPt1TightChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg",
            id=703,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1MediumChargedIsolationAndTightOOSCPhotonsDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
        Trigger(
            name="HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg",
            id=704,
            legs=[
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
                TriggerLeg(
                    pdg_id=15,
                    min_pt=45.0,
                    # filter names:
                    # hltDoublePFTau40TrackPt1TightChargedIsolationDz02Reg
                    trigger_bits=64,
                ),
            ],
            applies_to_dataset=(lambda dataset_inst: dataset_inst.is_data),
            tags={"cross_trigger", "cross_tau_tau", "channel_tau_tau"},
        ),
    ])    
