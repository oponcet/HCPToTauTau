# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from hcp.production.svfit import svfit, fastMTT


from hcp.util import TetraVec, _invariant_mass, deltaR

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@producer(
    uses={
        # nano columns
        "Jet.pt",
    },
    produces={
        # new columns
        "ht", "n_jet",
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = set_ak_column_f32(events, "ht", ak.sum(events.Jet.pt, axis=1))
    events = set_ak_column_i32(events, "n_jet", ak.num(events.Jet.pt, axis=1), value_type=np.int32)
    
    return events


@producer(
    uses={
        # nano columns
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "hcand.*",
    },
    produces={
        # new columns
        "hcand_invmass", "hcand_dr",
    },
)
def ll_features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    #false_mask = ak.zeros_like(1*events.event, dtype=np.float32)
    events = ak.Array(events, behavior=coffea.nanoevents.methods.nanoaod.behavior)
    # events["hcand"] = ak.with_name(events.hcand, "PtEtaPhiMLorentzVector")
    # Hcand = 1*events.hcand
    Hcand = ak.with_name(events.hcand, "PtEtaPhiMLorentzVector")


    mass = (Hcand[:,:1] + Hcand[:,1:2]).mass
    dr = ak.firsts(Hcand[:,:1].metric_table(Hcand[:,1:2]), axis=1)
    
    #from IPython import embed; embed()
    events = set_ak_column_f32(events, "hcand_invmass", ak.firsts(mass))
    events = set_ak_column_f32(events, "hcand_dr", ak.firsts(dr))
    
    return events


@producer(
    uses={
        mc_weight, category_ids,
        # nano columns
        "Jet.pt",
    },
    produces={
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply object masks and create new collections
    reduced_events = create_collections_from_masks(events, object_masks)

    # create category ids per event and add categories back to the
    events = self[category_ids](reduced_events, target_events=events, **kwargs)

    # add cutflow columns
    events = set_ak_column(
        events,
        "cutflow.jet1_pt",
        Route("Jet.pt[:,0]").apply(events, EMPTY_FLOAT),
    )

    return events


@producer(
    uses={
        features, ll_features, category_ids, normalization_weights, muon_weights, deterministic_seeds, svfit, fastMTT
    },
    produces={
        features, ll_features, category_ids, normalization_weights, muon_weights, deterministic_seeds, svfit, fastMTT
    },
    # sandbox=dev_sandbox("bash::$HCP_BASE/sandboxes/venv_svfit_dev.sh"),
)
def main(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # features
    events = self[features](events, **kwargs)

    # hcand features
    events = self[ll_features](events, **kwargs)
    
    # category ids
    events = self[category_ids](events, **kwargs)

    # deterministic seeds
    events = self[deterministic_seeds](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

    # svfit
    events = self[svfit](events, **kwargs)

    # fastMTT
    events = self[fastMTT](events, **kwargs)

    return events
