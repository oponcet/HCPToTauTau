# coding: utf-8

"""
Objects and Event Selections.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.production.util import attach_coffea_behavior
from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import, dev_sandbox
from hcp.util import TetraVec, _invariant_mass, deltaR


from hcp.selection.dl_veto import dilep_res_veto_selection
from hcp.selection.trigger import trigger_selection
from hcp.selection.lepton import lepton_selection
from hcp.selection.jet import jet_selection
from hcp.production.main import cutflow_features

from hcp.production.prepare_objects import buildhcand

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

@selector(
    uses={
        # selectors / producers called within _this_ selector
        json_filter, met_filters, mc_weight, cutflow_features, process_ids,
        trigger_selection, dilep_res_veto_selection, lepton_selection, jet_selection,
        increment_stats, attach_coffea_behavior, 

        buildhcand,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, lepton_selection, trigger_selection, cutflow_features, process_ids,
        buildhcand, attach_coffea_behavior, 
        "hcand.*",
    },
    #sandbox=dev_sandbox("bash::$HCP_BASE/sandboxes/venv_svfit_integration_dev.sh"),
    exposed=True,
)
def main(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # met filter selection
    #########events, met_filter_results = self[met_filters](events, **kwargs)
    #########results += met_filter_results

    #event_sel_json_and_met_filter = reduce(and_, results.steps.values())
    # ensure coffea behaviors are loaded
    events = self[attach_coffea_behavior](events, **kwargs)

    # dilepton_resonance_veto
    events, dl_veto_results = self[dilep_res_veto_selection](events, **kwargs)
    results += dl_veto_results
    event_sel_json_and_met_filter_and_dlresveto = reduce(and_, results.steps.values())
    print("event_sel_json_and_met_filter_and_dlresveto")
    # trigger selection
    events, trigger_results = self[trigger_selection](events, **kwargs)
    results += trigger_results
    event_sel_json_and_met_filter_and_dlresveto_and_trigger = reduce(and_, results.steps.values())

    # lepton selection
    events, lepton_results = self[lepton_selection](events, trigger_results, **kwargs)
    results += lepton_results
    event_sel_json_and_met_filter_and_dlresveto_and_trigger_and_lepton = reduce(and_, results.steps.values())

    events = self[buildhcand](events, lepton_results, **kwargs)

    # # gentau selection 
    # gentaus = self[gentau_selection](
    #     events, 
    #     **kwargs,
    # )
    # print("gentau = ", gentaus)
    # hcand_Lzt = events.hcand * 1 # transform into LorentzVector
    # GenPart_Lzt = events.GenPart * 1 # transform into LorentzVector

    # print("hcand_Lzt fields = ", hcand_Lzt.fields)
    # print("GenPart_Ltz fields = ", GenPart_Lzt.fields)
    # print("leps 1 = ", events.hcand[:,:1]*1)
    # print("leps 2 = ", events.hcand[:,1:2]*1)
    # print("GenPart = ", events.GenPart)
    # # from IPython import embed; embed()


    # gen_mask = ak.all(hcand_Lzt.metric_table(GenPart_Lzt) < 0.5, axis=2) # 0.5 is the deltaR cut
    # gen_mask = ak.any(hcand_Lzt.metric_table(GenPart_Lzt)<0.5, axis=2) # 0.5 is the deltaR cut
    # gen_mask = ak.all(dR_hcand_GenPart < 0.5, axis=2) # 0.5 is the deltaR cut
    # gen_mask_leps1 = ak.all(events.hcand[:,:1].metric_table(events.GenPart) < 0.5 , axis=2) # 0.5 is the deltaR cut
    # gen_mask_leps2 = ak.all(events.hcand[:,1:2].metric_table(gentau) < 0.5 , axis=2) # 0.5 is the deltaR cut
    # gen_mask = ak.all((events.hcand[:,:1]*1).metric_table(events.GenPart) < 0.5 , axis=2) & ak.all((events.hcand[:,1:2]*1).metric_table(events.GenPart) < 0.5 , axis=2) # 0.5 is the deltaR cut
    # print("gen_mask to list= ", ak.to_list(gen_mask))
    #from IPython import embed; embed()

    # leps1 = events.hcand[:,:1]
    # leps2 = events.hcand[:,1:2]

    # #Test 1 

    # hcand1 = events.hcand[:,:1]
    # hcand2 = events.hcand[:,1:2]


    # dr_gentaus_hcand1 = GenPart_Lzt.metric_table(hcand1) 
    # dr_gentaus_hcand2 = GenPart_Lzt.metric_table(hcand2)


    # sorted_idx_gentaus_hcand1 = ak.argsort(dr_gentaus_hcand1, axis=1)
    # sorted_idx_gentaus_hcand2 = ak.argsort(dr_gentaus_hcand2, axis=1)

    
    # mask_dr_gentaus_hcand1 = ak.any(dr_gentaus_hcand1<0.5, axis=2)
    # mask_dr_gentaus_hcand2 = ak.any(dr_gentaus_hcand2<0.5, axis=2)

    # print("4")
    # gentaus_hcand1 = hcand1[sorted_idx_gentaus_hcand1]
    # gentaus_hcand2 = hcand2[sorted_idx_gentaus_hcand2]

    # print("5")
    # mask_dr_gentaus_hcand1 = mask_dr_gentaus_hcand1[sorted_idx_gentaus_hcand1]
    # mask_dr_gentaus_hcand2 = mask_dr_gentaus_hcand2[sorted_idx_gentaus_hcand2]

    # print("6")
    # gentaus_hcand1 = gentaus_hcand1[mask_dr_gentaus_hcand1][:,0:1] 
    # gentaus_hcand2 = gentaus_hcand2[mask_dr_gentaus_hcand2][:,0:1] 

    # from IPython import embed; embed()


    # Test: 2 Events
    # # True Info                                                                                                                                                      
    # #   g11 is close to h11, g12 to h12                                                                                                                            
    # #   g21 is close to h22, g22 to h21   

    # gentaus = lepton_results.x.GenTaus * 1      #  [ [g11, g12], [g21, g22] ]   gentau information                                                                                                 
    # hcand1  = events.hcand[:,0:1] * 1 #  [ [h11], [h21] ]     hcand1 = lepton 1                                                                                                     
    # hcand2  = events.hcand[:,1:2] * 1 #  [ [h12], [h22] ]     hacand2 = lepton 2 taus                                                                                                   

    # print("1")
    # # deltaR between gentaus and hcands                                                                                                                           
    # dr_hcand1_gentaus = hcand1.metric_table(gentaus)             #  [ [[0.3, 1.2]], [[1.3, 0.2]] ]                                                                
    # dr_hcand1_gentaus = ak.firsts(dr_hcand1_gentaus, axis=1)     #  [ [0.3, 1.2], [1.3, 0.2] ]                                                                    
    # dr_hcand2_gentaus = hcand2.metric_table(gentaus)             #  [ [[2.3, 0.2]], [[0.3, 1.2]] ]                                                                
    # dr_hcand2_gentaus = ak.firsts(dr_hcand2_gentaus, axis=1)     #  [ [2.3, 0.2], [0.3, 1.2] ]                                                                    

    # print("2")
    # # Get the indices of the gentaus closed to the hcands                                                                                                         
    # sorted_idx_hcand1_gentaus = ak.argsort(dr_hcand1_gentaus, axis=1) # [ [0, 1], [1, 0] ]                                                                        
    # sorted_idx_hcand2_gentaus = ak.argsort(dr_hcand2_gentaus, axis=1) # [ [1, 0], [0, 1] ]                                                                        

    # print("3")
    # # Sorting the gentaus by the indices: first gentau should be the closest one to the hcand                                                                     
    # gentaus_hcand1 = gentaus[sorted_idx_hcand1_gentaus] # [ [g11, g12], [g22, g21] ]                                                                              
    # gentaus_hcand2 = gentaus[sorted_idx_hcand2_gentaus] # [ [g12, g11], [g21, g22] ]                                                                              

    # print("4")
    # # Get the dr mask                                                                                                                                             
    # mask_dr_hcand1_gentaus = dr_hcand1_gentaus < 0.4    #  [ [T,F], [F,T] ]                                                                                       
    # mask_dr_hcand2_gentaus = dr_hcand2_gentaus < 0.4    #  [ [F,T], [T,F] ]                                                                                       

    # print("5")
    # # Sort the position of the mask as wellaccording to the sorted indices                                                                                        
    # mask_dr_hcand1_gentaus = mask_dr_hcand1_gentaus[sorted_idx_hcand1_gentaus] # [ [T,F], [T,F] ]                                                                 
    # mask_dr_hcand2_gentaus = mask_dr_hcand2_gentaus[sorted_idx_hcand2_gentaus] # [ [T,F], [T,F] ]                                                                 

    # print("6")
    # # Apply these sorted masks on the sorted gentaus                                                                                                              
    # # Take the 1st one as it is the closest one                                                                                                                   
    # gentaus_hcand1 = gentaus_hcand1[mask_dr_hcand1_gentaus]  # [ [g11], [g22] ]                                                                                    
    # gentaus_hcand2 = gentaus_hcand2[mask_dr_hcand2_gentaus]  # [ [g12], [g21] ]                                                                                    

    # print("7")
    # # Take the 1st or closest one
    # gentau_hcand1 = gentaus_hcand1[: , 0:1]  # [ [g11], [g22] ] 
    # gentau_hcand2 = gentaus_hcand2[: , 0:1]  # [ [g12], [g21] ] 

    # # print("gentau_hcand1 = ", ak.to_list(gentau_hcand1))
    
    # print("8")
    # # To ensure that there are matched gentaus to the hcands                                                                                                      
    # hasonegentau_hcand1 = ak.num(gentau_hcand1.pt, axis=1) == 1  # [ True, True]
    # hasonegentau_hcand2 = ak.num(gentau_hcand2.pt, axis=1) == 1  # [ True, True]  

    # print("9")
    # # Genmatching mask
    # isgenmatched = hasonegentau_hcand1 & hasonegentau_hcand2  # [ True, True ]

    # print("gentau_hcand1 fields = ", gentau_hcand1.fields)
    # print("gentau_hcand2 fields = ", gentau_hcand2.fields)



    # genleps1_id = ak.argsort(leps1.metric_table(GenPart_Lzt), axis = 1) # 0.5 is the deltaR cut
    # genleps2_id = ak.argsort(leps2.metric_table(GenPart_Lzt), axis = 1) # 0.5 is the deltaR cut

    # genleph1 = GenPart_Lzt[genleps1_id[:,:,0]]

    # hcand_results = SelectionResult(
    #     steps={
    #         "higgs_cand": ak.num(events.hcand, axis=1) == 2,
    #         # "gen_mask": ak.sum(gen_mask, axis=1) == 2, # gen match need to match both leptons
    #         # "gen_mask": isgenmatch,
    #     },
    # )


    hcand_results = SelectionResult(
        steps={
            "higgs_cand": ak.fill_none(ak.num(events.hcand, axis=1) == 2, False),
        },
    )
    results += hcand_results
    event_sel_json_and_met_filter_and_dlresveto_and_trigger_and_lepton_and_hcand = reduce(and_, results.steps.values())

    # # Add genmatch fields as new fields of hcand

    # from IPython import embed; embed()

    # dummy_hcand = convert_to_hcand_type(events.hcand, events.hcand)[:, :0]

    # events.hcand = ak.where(isgenmatched, convert_to_hcand_type(events.hcand, ak.concatenate([gentau_hcand1, gentau_hcand2], axis=1)),hcand_dummy)
   

    # # Add gentau_hcand fields as new fields of hcand
    # events.hcand[:,0:1] = ak.with_field(events.hcand[:,0:1], gentau_hcand1, "gentau_hcand1")
    # events.hcand[:,1:2] = ak.with_field(events.hcand[:,1:2], gentau_hcand2, "gentau_hcand2")
    

    print("hcand fields = ", events.hcand.fields)

    from IPython import embed; embed()


    # from IPython import embed; embed()

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results


    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    # create process ids
    events = self[process_ids](events, **kwargs)

    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)

    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        #"num_events_json_met_filter": event_sel_json_and_met_filter,
        "num_events_json_met_filter_dlresveto": event_sel_json_and_met_filter_and_dlresveto,
        "num_events_json_met_filter_dlresveto_trigger": event_sel_json_and_met_filter_and_dlresveto_and_trigger,
        "num_events_json_met_filter_dlresveto_trigger_lepton": event_sel_json_and_met_filter_and_dlresveto_and_trigger_and_lepton,
        "num_events_json_met_filter_dlresveto_trigger_lepton_hcand": event_sel_json_and_met_filter_and_dlresveto_and_trigger_and_lepton_and_hcand,
        "num_events_selected": event_sel,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.event),
        }
        group_map = {
            # per process
            "process": {
                "values": events.process_id,
                "mask_fn": (lambda v: events.process_id == v),
            },
            # per jet multiplicity
            #"njet": {
            #    "values": results.x.n_jets,
            #    "mask_fn": (lambda v: results.x.n_jets == v),
            #},
            # per channel
            "channel": {
                "values": events.channel_id,
                "mask_fn": (lambda v: events.channel_id == v),
            },
        }
    events, results = self[increment_stats](
        events,
        results,
        stats,
        weight_map=weight_map,
        group_map=group_map,
        **kwargs,
    )
    return events, results


# def convert_to_hcand_type(phobj: ak.Array, genobj: ak.Array) -> ak.Array:
#     return ak.zip(
#         {
#             "pt": phobj.pt,
#             "eta": phobj.eta,
#             "phi": phobj.phi,
#             "mass": phobj.mass,
#             "charge": phobj.charge,
#             "decayMode": phobj.decayMode,
#             "lepton": phobj.lepton,
#             "pt_gen" : genobj.pt,
#             "eta_gen": genobj.eta,
#             "phi_gen": genobj.phi,
#             "mass_gen": genobj.mass,
#         },
#         with_name = "PtEtaPhiMLorentzVector",
#         behavior = coffea.nanoevents.methods.vector.behavior,
#     )