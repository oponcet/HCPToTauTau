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
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")

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


    hcand_results = SelectionResult(
        steps={
            "higgs_cand": ak.fill_none(ak.num(events.hcand, axis=1) == 2, False),
            # "gen_mask": isgenmatched,
        },
    )
    results += hcand_results
    event_sel_json_and_met_filter_and_dlresveto_and_trigger_and_lepton_and_hcand = reduce(and_, results.steps.values())

    print("hcand fields = ", events.hcand.fields)

    # genpt_is_none = events.hcand.genpt is None

    # # Select the events where gentpt is None
    # hcand_with_none_genpt = events.hcand[genpt_is_none]

    # Print the fields of the selected events
    # print(ak.to_list(events.hcand))

    # print("hcand after where: ", ak.to_list(events.hcand))

    #from IPython import embed; embed()





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


def invariant_mass(pt, eta, phi):
    # Calculate the transverse momentum component in the z direction (assuming massless particles)
    pz = pt * np.sinh(eta)
    # Calculate the energy component
    E = np.sqrt(pt**2 + pz**2)
    # Calculate the invariant mass
    m = np.sqrt(E**2 - pt**2)  # Simplified for massless particles
    return m