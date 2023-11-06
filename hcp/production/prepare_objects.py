# coding: utf-8

"""
Prepare h-Candidate from SelectionResult: selected lepton indices & channel_id [trigger matched] 
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

    # Unzip the dtrpairs array into two separate arrays        
    lep1unzip, lep2unzip = ak.unzip(dtrpairs)

    # Create masks to filter pairs of objects
    charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0 # opposite charge
    invm_mask   = (1*lep1unzip + 1*lep2unzip).mass > 40.0 # cut on invariant mass
    dr_mask = (1*lep1unzip).delta_r(1*lep2unzip) > 0.5 # dr cut 

    # Combine the masks using bitwise AND to create a final mask
    dtr_mask = (charge_mask & invm_mask & dr_mask)

    # If 'met' and 'mt_threshold' are provided, add an additional mask based on transverse mass
    if met is not None:
        mt_mask =  transverse_mass(lep1unzip, met) < mt_threshold
        dtr_mask = dtr_mask & mt_mask

    # Apply the final mask to filter the pairs    
    dtrpairs = dtrpairs[dtr_mask]

    #from IPython import embed; embed()

    # Return the filtered pairs
    return dtrpairs

def sort_and_get_pair_semilep(dtrpairs: ak.Array)->ak.Array:
    sorted_idx = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)

    # Sort the pairs based on pfRelIso03_all of the first object in each pair
    dtrpairs = dtrpairs[sorted_idx]
    # from IPython import embed; embed()
    #print("dtrpairs =", dtrpairs)

    # Check if there are multiple pairs for each event
    where_many = ak.num(dtrpairs["0"], axis=1) > 1

    # Extract the pfRelIso03_all values for the first object in each pair
    lep1_pfRelIso03_all = dtrpairs["0"].pfRelIso03_all
    #ak.where(where_many, (ak.firsts(lep1_pfRelIso03_all[:,:1], axis=1) == ak.firsts(lep1_pfRelIso03_all[:,1:2], axis=1)), ~where_many)
    
    # Check if the pfRelIso03_all values are the same for the first two objects in each pair
    where_same_iso_1 = (
        where_many &
        ak.fill_none(
            (
                ak.firsts(dtrpairs["0"].pfRelIso03_all[:,:1], axis=1) #
                ==
                ak.firsts(dtrpairs["0"].pfRelIso03_all[:,1:2], axis=1)
            ), False
        )
    )

    # Sort the pairs based on pt if pfRelIso03_all is the same for the first two objects
    sorted_idx = ak.where(where_same_iso_1,
                          ak.argsort(dtrpairs["0"].pt, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]
        
    # Check if the pt values are the same for the first two objects in each pair    
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
    # Sort the pairs based on pt if rawDeepTau2017v2p1VSjet is the same for the first two objects
    sorted_idx = ak.where(where_same_iso_2,
                          ak.argsort(dtrpairs["1"].pt, ascending=False),
                          sorted_idx)
    # finally, the pairs are sorted
    dtrpairs = dtrpairs[sorted_idx]
    #from IPython import embed; embed() 

    # Extract the first object in each pair (lep1) and the second object (lep2)
    lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
    lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
    # print(f"lep1 pt: {lep1.pt}")
    # print(f"lep2 pt: {lep2.pt}")

    # Concatenate lep1 and lep2 to create the final dtrpair
    dtrpair = ak.concatenate([lep1, lep2], axis=1) 

    return dtrpair




def sort_and_get_pair_fullhad(dtrpairs: ak.Array)->ak.Array:
    # redundant, because taus were sorted by the deeptau before
    sorted_idx = ak.argsort(dtrpairs["0"].rawDeepTau2017v2p1VSjet, ascending=True)
    dtrpairs = dtrpairs[sorted_idx]
    where_many = ak.num(dtrpairs["0"], axis=1) > 1

    # if the deep tau val of tau-0 is the same for the first two pair
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

    # if so, sort the pairs according to the deep tau of the 2nd tau
    sorted_idx = ak.where(where_same_iso_1,
                          ak.argsort(dtrpairs["1"].rawDeepTau2017v2p1VSjet, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]

    # if the deep tau val of tau-1 is the same for the first two pair 
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

    # sort them with the pt of the 1st tau
    sorted_idx = ak.where(where_same_iso_2,
                          ak.argsort(dtrpairs["0"].pt, ascending=False),
                          sorted_idx)
    dtrpairs = dtrpairs[sorted_idx]
        
    # check if the first two pairs have the second tau with same rawDeepTau2017v2p1VSjet
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

    lep1 = ak.singletons(ak.firsts(dtrpairs["0"], axis=1))
    lep2 = ak.singletons(ak.firsts(dtrpairs["1"], axis=1))
    # print(f"lep1 pt: {lep1.pt}")
    # print(f"lep2 pt: {lep2.pt}")
    dtrpair = ak.concatenate([lep1, lep2], axis=1) 

    return dtrpair



@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.pfRelIso03_all",
        "Tau.pt", "Tau.rawDeepTau2017v2p1VSjet",
        "MET.pt", "MET.phi",
    },
)
def buildhcand(self: Producer,
               events: ak.Array,
               selres: SelectionResult,
               **kwargs) -> ak.Array:
    print("Started building h-cand")
    #empty_indices = ak.zeros_like(1*events.channel_id, dtype=np.uint16)[..., None][..., :0]
    #empty_indices = ak.zeros_like(1*events.Electron)

    # Extract the particles (Electrons, Muons, Taus) from the selection result
    Electrons = selres.x.Electrons
    Muons = selres.x.Muons
    Taus = selres.x.Taus

    # from IPython import embed; embed()

    # Initialize empty indices for Electrons
    empty_indices_ele = ak.zeros_like(
        ak.singletons(
            ak.firsts(Electrons, axis=1)
        )
    )

    # Initialize an empty h-cand array
    h_cand = empty_indices_ele[...,:0]
    #print(f"h_cand | empty_indices: {h_cand}")

    channels = ["etau", "mutau", "tautau"]

    for ch_idx, ch in enumerate(channels):
        print(f"Idx: {ch_idx} || Channel: {ch} || {self.config_inst.get_channel(ch)}")

        if (ch == "etau"):
            print(f"channel: {ch}")
            assert (ch == self.config_inst.get_channel(ch).name), "Should be etau channel"
            where = (events.channel_id == self.config_inst.get_channel(ch).id)
            
            # Define a mask for selecting events with specific lepton counts for etau 
            nlep_mask = (
                (ak.num(Electrons, axis=-1) >= 1)
                & (ak.num(Taus, axis=-1) >= 1)
                & (ak.num(Muons, axis=-1) == 0)
            )
            #from IPython import embed; embed()

            # Apply the mask 
            where = where & nlep_mask

            leps1 = Electrons
            leps2 = Taus

            # Create pairs of leptons
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)

            # Select pairs based on certain criteria
            dtrpairs_sel = select_pairs(dtrpairs, events.MET, 50.0)
            # print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            # print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")

            dtrpair = sort_and_get_pair_semilep(dtrpairs_sel)
            #print(f"dtrpair pt: {dtrpair.pt}")
            
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
            # print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            # print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")  

            # Sort and get the selected pair
            dtrpair = sort_and_get_pair_semilep(dtrpairs_sel)
            #print(f"dtrpair pt: {dtrpair.pt}")
            
            # Update the h-cand array with the selected pair
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
            leps = Taus
            dtrpairs = ak.combinations(leps, 2, axis=-1)
            dtrpairs_sel = select_pairs(dtrpairs)
            # print(f"""dtrpairs["0"] fields: {dtrpairs_sel["0"].fields}""")
            # print(f"""dtrpairs["1"] fields: {dtrpairs_sel["1"].fields}""")

            dtrpair = sort_and_get_pair_fullhad(dtrpairs_sel)
            #print(f"dtrpair pt: {dtrpair.pt}")

            h_cand = ak.where(where, dtrpair, h_cand)

    # print(f"h_cand : {h_cand.fields}")

    # Set the "hcand" column in the events array with the h-cand information
    events = set_ak_column(events, "hcand", h_cand)
    #from IPython import embed; embed()

    return events
