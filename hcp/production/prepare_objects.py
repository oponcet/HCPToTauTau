# coding: utf-8

"""
Prepare h-Candidate from SelectionResult: lepton_pair & channel_id 
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from hcp.util import invariant_mass, deltaR

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)


@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.pfRelIso03_all",
        "Tau.pt", "Tau.rawDeepTau2017v2p1VSjet",
    },
    #produces={
    #    "h_cand.pt", "h_cand.eta", "h_cand.phi", "h_cand.mass", "h_cand.charge",
    #}
)
def buildhcand(self: Producer,
               events: ak.Array,
               lepton_pair: ak.Array,
               **kwargs) -> ak.Array:
    print("Started building h-cand")
    #empty_indices = ak.zeros_like(1*events.channel_id, dtype=np.uint16)[..., None][..., :0]
    #empty_indices = ak.zeros_like(1*events.Electron)
    empty_indices = ak.zeros_like(
        ak.singletons(
            ak.firsts(1*events.Electron, axis=1)
        )
    )
    #from IPython import embed; embed()
    h_cand = empty_indices[...,:0]
    print(f"h_cand | empty_indices: {h_cand}")
    channels = ["etau", "mutau", "tautau"]

    for ch_idx, ch in enumerate(channels):
        print(f"Idx: {ch_idx} || Channel: {ch} || {self.config_inst.get_channel(ch)}")

        if (ch == "etau"):
            print(f"channel: {ch}")
            #assert (ch == self.config_inst.get_channel(ch).name), "Should be etau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) >= 1)
                & (ak.num(events.Tau, axis=-1) >= 1)
                & (ak.num(events.Muon, axis=-1) == 0)
            )
            where = where & nlep_mask
            leps1 = events.Electron
            leps2 = events.Tau
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs["0"] fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs["1"].fields}""")
            iso_sort_idx_1 = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_1]
            pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_1]
            iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_2]
            pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_2]
            lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            print(f"lep1 pt: {lep1.pt}")
            print(f"lep2 pt: {lep2.pt}")
            dtrpair = ak.concatenate([lep1, lep2], axis=1) 
            print(f"dtrpair pt: {dtrpair.pt}")
            h_cand = ak.where(where, 1*dtrpair, h_cand)
        
        elif (ch == "mutau"):
            print(f"channel: {ch}")
            #assert (ch == self.config_inst.get_channel(ch).name), "Should be mutau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) == 0)
                & (ak.num(events.Tau, axis=-1) >= 1)
                & (ak.num(events.Muon, axis=-1) >= 1)
            )
            where = where & nlep_mask
            leps1 = events.Muon
            leps2 = events.Tau
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs["0"] fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs["1"].fields}""")
            iso_sort_idx_1 = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_1]
            pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_1]
            iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_2]
            pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_2]
            lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            print(f"lep1 pt: {lep1.pt}")
            print(f"lep2 pt: {lep2.pt}")
            dtrpair = ak.concatenate([lep1, lep2], axis=1) 
            print(f"dtrpair pt: {dtrpair.pt}")
            h_cand = ak.where(where, 1*dtrpair, h_cand)
            
        elif (ch == "tautau"):
            print(f"channel: {ch}")
            #assert (ch == self.config_inst.get_channel(ch).name), "Should be tautau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) == 0)
                & (ak.num(events.Tau, axis=-1) >= 2)
                & (ak.num(events.Muon, axis=-1) == 0)
            )
            where = where & nlep_mask
            leps1 = leps2 = events.Tau
            dtrpairs = ak.combinations(leps1, 2, axis=-1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs["0"] fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs["1"].fields}""")
            iso_sort_idx_1 = ak.argsort(dtrpairs["0"].rawDeepTau2017v2p1VSjet, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_1]
            pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_1]
            iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_2]
            pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_2]
            lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            print(f"lep1 pt: {lep1.pt}")
            print(f"lep2 pt: {lep2.pt}")
            dtrpair = ak.concatenate([lep1, lep2], axis=1) 
            print(f"dtrpair pt: {dtrpair.pt}")
            #from IPython import embed; embed()
            h_cand = ak.where(where, 1*dtrpair, h_cand)

    #print(f"h_cand: {h_cand.type}")
    #print(f"h_cand: {ak.max(ak.num(h_cand,axis=-1))}")

    #for i in range(1000):
    #print(h_dtr_pairs[861], h_dtr_pairs[861]["0"], h_dtr_pairs[861]["1"])
    #for i in range(10000):
    #    print(i, h_dtr_pairs[i])
    
    #print(f"hlep2: {hlep2.layout}")
    print(f"h_cand : {h_cand.fields}")
    events = set_ak_column(events, "hcand", h_cand)
    #from IPython import embed; embed()

    return events
