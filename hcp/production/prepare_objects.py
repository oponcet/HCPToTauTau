# coding: utf-8

"""
Prepare h-Candidate from SelectionResult: lepton_pair & channel_id 
"""

import functools

from typing import Optional
from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.selection import SelectionResult
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from hcp.util import invariant_mass, deltaR, transverse_mass

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

def select_pairs(dtrpairs: ak.Array,
                 met: Optional[ak.Array] = None,
                 mt_threshold: Optional[float] = None)->ak.Array:
    lep1unzip, lep2unzip = ak.unzip(dtrpairs)
    charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
    invm_mask   = (1*lep1unzip + 1*lep2unzip).mass > 40.0
    dr_mask = (1*lep1unzip).delta_r(1*lep2unzip) > 0.5
    dtr_mask = (charge_mask & invm_mask & dr_mask)
    if met is not None:
        mt_mask =  transverse_mass(lep1unzip, met) < mt_threshold
        dtr_mask = dtr_mask & mt_mask
    dtrpairs = dtrpairs[dtr_mask]
    #from IPython import embed; embed()
    return dtrpairs

def sort_and_get_pair_semilep(dtrpairs: ak.Array)->ak.Array:
    sorted_idx = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
    dtrpairs = dtrpairs[sorted_idx]
    where_many = ak.num(dtrpairs["0"], axis=1) > 1
    lep1_pfRelIso03_all = dtrpairs["0"].pfRelIso03_all
    #ak.where(where_many, (ak.firsts(lep1_pfRelIso03_all[:,:1], axis=1) == ak.firsts(lep1_pfRelIso03_all[:,1:2], axis=1)), ~where_many)
    
    where_same_iso_1 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["0"].pfRelIso03_all[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["0"].pfRelIso03_all[:,1:2], axis=1)
            ), False
        )
    )
    sorted_idx = ak.where(where_same_iso_1,
                          ak.argsort(dtrpairs["0"].pt, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]
        
    where_same_pt_1 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["0"].pt[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["0"].pt[:,1:2], axis=1)
            ), False
        )
    )
    # if so, sort the pairs with tau rawDeepTau2017v2p1VSjet
    sorted_idx = ak.where(where_same_pt_1,
                          ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]
    # check if the first two pairs have taus with same rawDeepTau2017v2p1VSjet
    where_same_iso_2 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["1"].rawDeepTau2017v2p1VSjet[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["1"].rawDeepTau2017v2p1VSjet[:,1:2], axis=1)
            ), False
        )
    )
    # if so, sort the taus with their pt
    sorted_idx = ak.where(where_same_iso_2,
                          ak.argsort(dtrpairs["1"].pt, ascending=False),
                          sorted_idx)
    # finally, the pairs are sorted
    dtrpairs = dtrpairs[sorted_idx]
    #from IPython import embed; embed() 

    lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
    lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
    print(f"lep1 pt: {lep1.pt}")
    print(f"lep2 pt: {lep2.pt}")
    dtrpair = ak.concatenate([lep1, lep2], axis=1) 

    return dtrpair




def sort_and_get_pair_fullhad(dtrpairs: ak.Array)->ak.Array:
    sorted_idx = ak.argsort(dtrpairs["0"].rawDeepTau2017v2p1VSjet, ascending=True)
    dtrpairs = dtrpairs[sorted_idx]
    where_many = ak.num(dtrpairs["0"], axis=1) > 1
    
    where_same_iso_1 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["0"].rawDeepTau2017v2p1VSjet[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["0"].rawDeepTau2017v2p1VSjet[:,1:2], axis=1)
            ), False
        )
    )
    sorted_idx = ak.where(where_same_iso_1,
                          ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]

    where_same_iso_2 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["1"].rawDeepTau2017v2p1VSjet[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["1"].rawDeepTau2017v2p1VSjet[:,1:2], axis=1)
            ), False
        )
    )
    sorted_idx = ak.where(where_same_iso_2,
                          ak.argsort(dtrpairs["0"].pt, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]
        
    # if so, sort the pairs with tau rawDeepTau2017v2p1VSjet
    # check if the first two pairs have taus with same rawDeepTau2017v2p1VSjet
    where_same_pt_1 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["0"].pt[:,:1], axis=1)
                ==
                ak.firsts(dtrpairs["0"].pt[:,1:2], axis=1)
            ), False
        )
    )
    # if so, sort the taus with their pt
    sorted_idx = ak.where(where_same_pt_1,
                          ak.argsort(dtrpairs["1"].pt, ascending=False),
                          sorted_idx)
    # finally, the pairs are sorted
    dtrpairs = dtrpairs[sorted_idx]
    #from IPython import embed; embed() 

    lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
    lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
    print(f"lep1 pt: {lep1.pt}")
    print(f"lep2 pt: {lep2.pt}")
    dtrpair = ak.concatenate([lep1, lep2], axis=1) 

    return dtrpair



@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.pfRelIso03_all",
        "Tau.pt", "Tau.rawDeepTau2017v2p1VSjet",
        "MET.pt", "MET.phi",
        #selpairs,
    },
)
def buildhcand(self: Producer,
               events: ak.Array,
               selres: SelectionResult,
               **kwargs) -> ak.Array:
    print("Started building h-cand")
    #empty_indices = ak.zeros_like(1*events.channel_id, dtype=np.uint16)[..., None][..., :0]
    #empty_indices = ak.zeros_like(1*events.Electron)
    Electrons = selres.x.Electrons
    Muons = selres.x.Muons
    Taus = selres.x.Taus
    #from IPython import embed; embed()
    empty_indices_ele = ak.zeros_like(
        ak.singletons(
            ak.firsts(Electrons, axis=1)
        )
    )
    empty_indices_mu = ak.zeros_like(
        ak.singletons(
            ak.firsts(Muons, axis=1)
        )
    )
    empty_indices_tau = ak.zeros_like(
        ak.singletons(
            ak.firsts(Taus, axis=1)
        )
    )
    #from IPython import embed; embed()
    h_cand = empty_indices_ele[...,:0]
    print(f"h_cand | empty_indices: {h_cand}")
    channels = ["etau", "mutau", "tautau"]

    for ch_idx, ch in enumerate(channels):
        print(f"Idx: {ch_idx} || Channel: {ch} || {self.config_inst.get_channel(ch)}")

        if (ch == "etau"):
            print(f"channel: {ch}")
            assert (ch == self.config_inst.get_channel(ch).name), "Should be etau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(Electrons, axis=-1) >= 1)
                & (ak.num(Taus, axis=-1) >= 1)
                & (ak.num(Muons, axis=-1) == 0)
            )
            #from IPython import embed; embed() 
            where = where & nlep_mask
            leps1 = ak.where(where, Electrons, empty_indices_ele[...,:0])
            leps2 = ak.where(where, Taus, empty_indices_tau[...,:0])
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            dtrpairs_sel = select_pairs(dtrpairs, events.MET, 50.0)
            print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")
            dtrpair = sort_and_get_pair_semilep(dtrpairs_sel)
            
            ##from IPython import embed; embed()
            #print(f"""dtrpairs["0"] fields: {dtrpairs["0"].fields}""")
            #print(f"""dtrpairs["1"] fields: {dtrpairs["1"].fields}""")
            #iso_sort_idx_1 = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_1]
            #pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_1]
            #iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_2]
            #pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_2]
            #lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            #lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            #print(f"lep1 pt: {lep1.pt}")
            #print(f"lep2 pt: {lep2.pt}")
            #dtrpair = ak.concatenate([lep1, lep2], axis=1) 
            print(f"dtrpair pt: {dtrpair.pt}")
            h_cand = ak.where(where, dtrpair, h_cand)
        
        elif (ch == "mutau"):
            print(f"channel: {ch}")
            assert (ch == self.config_inst.get_channel(ch).name), "Should be mutau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(Electrons, axis=-1) == 0)
                & (ak.num(Taus, axis=-1) >= 1)
                & (ak.num(Muons, axis=-1) == 1)
            )
            where = where & nlep_mask
            leps1 = Muons
            leps2 = Taus
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            dtrpairs_sel = select_pairs(dtrpairs, events.MET, 40.0)
            print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")
            #iso_sort_idx_1 = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_1]
            #pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_1]
            #iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_2]
            #pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_2]
            #lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            #lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            #print(f"lep1 pt: {lep1.pt}")
            #print(f"lep2 pt: {lep2.pt}")
            #dtrpair = ak.concatenate([lep1, lep2], axis=1)
            dtrpair = sort_and_get_pair_semilep(dtrpairs_sel)
            
            print(f"dtrpair pt: {dtrpair.pt}")
            h_cand = ak.where(where, dtrpair, h_cand)
            
        elif (ch == "tautau"):
            print(f"channel: {ch}")
            assert (ch == self.config_inst.get_channel(ch).name), "Should be tautau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            nlep_mask = (
                (ak.num(Electrons, axis=-1) == 0)
                & (ak.num(Taus, axis=-1) >= 2)
                & (ak.num(Muons, axis=-1) == 0)
            )
            where = where & nlep_mask
            leps1 = leps2 = Taus
            dtrpairs = ak.combinations(leps1, 2, axis=-1)
            dtrpairs_sel = select_pairs(dtrpairs)
            print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")
            #iso_sort_idx_1 = ak.argsort(dtrpairs["0"].rawDeepTau2017v2p1VSjet, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_1]
            #pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_1]
            #iso_sort_idx_2 = ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=True)
            #dtrpairs = dtrpairs[iso_sort_idx_2]
            #pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            #dtrpairs = dtrpairs[pt_sort_idx_2]
            #lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
            #lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
            #print(f"lep1 pt: {lep1.pt}")
            #print(f"lep2 pt: {lep2.pt}")
            #dtrpair = ak.concatenate([lep1, lep2], axis=1)
            dtrpair = sort_and_get_pair_fullhad(dtrpairs_sel)
            print(f"dtrpair pt: {dtrpair.pt}")
            h_cand = ak.where(where, dtrpair, h_cand)

    print(f"h_cand : {h_cand.fields}")
    events = set_ak_column(events, "hcand", h_cand)
    #from IPython import embed; embed()

    return events
