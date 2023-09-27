# coding: utf-8

"""
Objects and Event Selections.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.stats import increment_stats
from columnflow.selection.cms.json_filter import json_filter
from columnflow.selection.cms.met_filters import met_filters
from columnflow.production.processes import process_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import

from hcp.selection.trigger import trigger_selection
from hcp.selection.lepton import lepton_selection
from hcp.selection.jet import jet_selection
from hcp.production.main import cutflow_features

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        # selectors / producers called within _this_ selector
        json_filter, met_filters, mc_weight, cutflow_features, process_ids, trigger_selection,
        lepton_selection, jet_selection,
        increment_stats,
    },
    produces={
        # selectors / producers whose newly created columns should be kept
        mc_weight, lepton_selection, trigger_selection, cutflow_features, process_ids,
    },
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
    print("stage-0")
    # filter bad data events according to golden lumi mask
    if self.dataset_inst.is_data:
        events, json_filter_results = self[json_filter](events, **kwargs)
        results += json_filter_results

    # met filter selection
    #########events, met_filter_results = self[met_filters](events, **kwargs)
    #########results += met_filter_results

    #event_sel_json_and_met_filter = reduce(and_, results.steps.values())

    # trigger selection
    events, trigger_results = self[trigger_selection](events, **kwargs)
    results += trigger_results
    print("stage-1")
    event_sel_json_and_met_filter_and_trigger = reduce(and_, results.steps.values())

    # lepton selection
    events, lepton_results = self[lepton_selection](events, trigger_results, **kwargs)
    results += lepton_results
    print("stage-2")
    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results
    print("stage-3")
    # combined event selection after all steps
    # results.main["event"] = results.steps.muon & results.steps.jet
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    # create process ids
    events = self[process_ids](events, **kwargs)
    print("stage-4")
    # add the mc weight
    #if self.dataset_inst.is_mc:
    #    events = self[mc_weight](events, **kwargs)
    #print("stage-5")

    # create process ids
    events = self[process_ids](events, **kwargs)
    print("stage-5")
    print(f"SelRes objects: {results.objects}")
    # add cutflow features, passing per-object masks
    events = self[cutflow_features](events, results.objects, **kwargs)
    print("stage-7")
    # increment stats
    weight_map = {
        "num_events": Ellipsis,
        #"num_events_json_met_filter": event_sel_json_and_met_filter,
        "num_events_json_met_filter_trigger": event_sel_json_and_met_filter_and_trigger,
        "num_events_selected": event_sel,
    }
    group_map = {}
    if self.dataset_inst.is_mc:
        weight_map = {
            **weight_map,
            # mc weight for all events
            "sum_mc_weight": (events.mc_weight, Ellipsis),
            "sum_mc_weight_selected": (events.mc_weight, results.main.event),
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
