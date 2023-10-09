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
)
def buildhcand(self: Producer,
               events: ak.Array,
               lepton_pair: ak.Array,
               **kwargs) -> ak.Array:
    print("Started building h-cand")
    print(f"Lepton pair: {lepton_pair}, {lepton_pair.type}")

    nleptons = ak.num(lepton_pair, axis=-1)  # [[ne, nmu, ntau], [ne, nmu, ntau], ...]
    nsumleptons = ak.sum(nleptons, axis=-1)  # [[ne+nmu+ntau], [ne+nmu+ntau], ...]
    nlepmask = nsumleptons >= 2

    #eles = ak.firsts(lepton_pair[:,:1,:], axis=1)
    #muos = ak.firsts(lepton_pair[:,1:2,:], axis=1)
    #taus = ak.firsts(lepton_pair[:,2:3,:], axis=1)
    #leps = ak.firsts(lepton_pair[:,:1,:], axis=1)

    empty_indices = ak.zeros_like(events.channel_id, dtype=np.uint16)[..., None][..., :0]
    h_dtr_pairs = empty_indices
    channels = ["etau", "mutau", "tautau"]
    #for ch_idx, ch in enumerate(channels):
    for ch_idx, ch in enumerate(channels):
        print(f"Idx: {ch_idx} || Channel: {ch} || {self.config_inst.get_channel(ch)}")
        where = (events.channel_id == self.config_inst.get_channel(ch).id)
        #print(f"where: {where}")
        #leps1 = ak.firsts(lepton_pair[:,ch_idx:(ch_idx+1),:], axis=1)
        #print(f"leps1: {leps1}")
        #leps2 = ak.firsts(lepton_pair[:,-1:,:], axis=1)
        #print(f"leps2: {leps2}")
        #print(f"Leps: {leps1.type}, {leps2.type}")
        ##print(leps1.layout)
        ##dtrpairs = ak.combinations(leps2, 2) if ch_idx == 2 else ak.cartesian([leps1, leps2], axis=-1)
        ## combinations on union is not working
        #dtrpairs = ak.cartesian([leps1, leps2], axis=1)
        #lep1unzip, lep2unzip = ak.unzip(dtrpairs)
        #charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
        #invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
        #dtr_mask = (charge_mask & invm_mask)
        #dtrpairs = dtrpairs[dtr_mask]
        ##h_dtr_pairs = ak.where(where, dtrpairs, h_dtr_pairs)        
        if (ch == "etau"):
            print(f"channel: {ch}")
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) >= 1)
                & (ak.num(events.Tau, axis=-1) >= 1)
                & (ak.num(events.Muon, axis=-1) == 0)
            )
            where = where & nlep_mask
            leps1 = events.Electron
            print(f"leps1: {leps1.fields}")
            leps2 = events.Tau
            print(f"leps2: {leps2.fields}")
            #print(f"Leps: {leps1.type}, {leps2.type}")
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs fields: {dtrpairs["1"].fields}""")
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
            print(lep1.pt, lep2.pt)
            #lep1 = ak.firsts(dtrpairs["0"], axis=1)
            #lep2 = ak.firsts(dtrpairs["1"], axis=1)
            #print(dtrpairs["0"].pt, dtrpairs["1"].pt)
            #print(lep1, lep2)
            hcand = ak.concatenate([lep1, lep2], axis=1) 
            print(hcand.pt)
            #hcand = ak.concatenate([dtrpairs["0"],dtrpairs["1"]], axis=-1)
            #print(ak.firsts(hcand, axis=1))
            #print(dtrpairs["0"].pt)
            #print(ak.firsts(dtrpairs["0"].pt, axis=1))
            h_dtr_pairs = ak.where(where, hcand, h_dtr_pairs)

        elif (ch == "mutau"):
            print(f"channel: {ch}")
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) == 0)
                & (ak.num(events.Tau, axis=-1) >= 1)
                & (ak.num(events.Muon, axis=-1) >= 1)
            )
            where = where & nlep_mask
            leps1 = events.Muon
            print(f"leps1: {leps1.fields}")
            leps2 = events.Tau
            print(f"leps2: {leps2.fields}")
            #print(f"Leps: {leps1.type}, {leps2.type}")
            dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs fields: {dtrpairs["1"].fields}""")
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
            print(lep1.pt, lep2.pt)
            #lep1 = ak.firsts(dtrpairs["0"], axis=1)
            #lep2 = ak.firsts(dtrpairs["1"], axis=1)
            #print(dtrpairs["0"].pt, dtrpairs["1"].pt)
            #print(lep1, lep2)
            hcand = ak.concatenate([lep1, lep2], axis=1) 
            print(hcand.pt)
            #hcand = ak.concatenate([dtrpairs["0"],dtrpairs["1"]], axis=-1)
            #print(ak.firsts(hcand, axis=1))
            #print(dtrpairs["0"].pt)
            #print(ak.firsts(dtrpairs["0"].pt, axis=1))
            h_dtr_pairs = ak.where(where, hcand, h_dtr_pairs)
            """
            print(f"channel: {ch}")
            iso_sort_idx_1 = ak.argsort(dtrpairs["0"].pfRelIso03_all, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_1]
            pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_1]
            iso_sort_idx_2 = ak.argsort(dtrpairs["1"].pfRelIso03_all, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_2]
            pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_2]
            h_dtr_pairs = ak.where(where, dtrpairs, h_dtr_pairs)
            """
            
        elif (ch == "tautau"):
            print(f"channel: {ch}")
            nlep_mask = (
                (ak.num(events.Electron, axis=-1) == 0)
                & (ak.num(events.Tau, axis=-1) >= 2)
                & (ak.num(events.Muon, axis=-1) == 0)
            )
            where = where & nlep_mask
            leps1 = leps2 = events.Tau
            print(f"leps1: {leps1.fields}")
            #dtrpairs = ak.cartesian([leps1, leps2], axis=1)
            dtrpairs = ak.combinations(leps1, 2, axis=-1)
            lep1unzip, lep2unzip = ak.unzip(dtrpairs)
            charge_mask = (lep1unzip.charge + lep2unzip.charge) == 0
            invm_mask = (1*lep1unzip + 1*lep2unzip).mass > 40
            dtr_mask = (charge_mask & invm_mask)
            dtrpairs = dtrpairs[dtr_mask]
            print(f"""dtrpairs fields: {dtrpairs["0"].fields}""")
            print(f"""dtrpairs fields: {dtrpairs["1"].fields}""")
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
            print(lep1.pt, lep2.pt)
            #lep1 = ak.firsts(dtrpairs["0"], axis=1)
            #lep2 = ak.firsts(dtrpairs["1"], axis=1)
            #print(dtrpairs["0"].pt, dtrpairs["1"].pt)
            #print(lep1, lep2)
            hcand = ak.concatenate([lep1, lep2], axis=1) 
            print(hcand.pt)
            #hcand = ak.concatenate([dtrpairs["0"],dtrpairs["1"]], axis=-1)
            #print(ak.firsts(hcand, axis=1))
            #print(dtrpairs["0"].pt)
            #print(ak.firsts(dtrpairs["0"].pt, axis=1))
            h_dtr_pairs = ak.where(where, hcand, h_dtr_pairs)
            """
            print(f"channel: {ch}")
            iso_sort_idx_1 = ak.argsort(dtrpairs["0"].byDeepTau2017v2p1VSjetraw, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_1]
            pt_sort_idx_1 = ak.argsort(dtrpairs["0"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_1]
            iso_sort_idx_2 = ak.argsort(dtrpairs["1"].byDeepTau2017v2p1VSjetraw, ascending=True)
            dtrpairs = dtrpairs[iso_sort_idx_2]
            pt_sort_idx_2 = ak.argsort(dtrpairs["1"].pt, ascending=False)
            dtrpairs = dtrpairs[pt_sort_idx_2]
            h_dtr_pairs = ak.where(where, dtrpairs, h_dtr_pairs)
            """
    #h_dtr_os_mask = ak.sum(h_dtr_pairs

    print(f"h_dtr: {h_dtr_pairs.type}")
    print(f"h_dtr_pairs: {ak.max(ak.num(h_dtr_pairs,axis=-1))}")
    #for i in range(1000):
    #print(h_dtr_pairs[861], h_dtr_pairs[861]["0"], h_dtr_pairs[861]["1"])
    #for i in range(10000):
    #    print(i, h_dtr_pairs[i])
    
    
    #hlep1 = h_dtr_pairs[:,:1]["0"]
    #hlep2 = h_dtr_pairs[:,:1]["1"]
    
    #print(f"hlep1: {hlep1.layout}")
    #print(f"hlep2: {hlep2.layout}")




    return events
