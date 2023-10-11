# coding: utf-8

"""
Collection of helpers
"""

import law
import order as od
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


def TetraVec(arr: ak.Array) -> ak.Array:
    TetraVec = ak.zip({"pt": arr.pt, "eta": arr.eta, "phi": arr.phi, "mass": arr.mass},
                      with_name="PtEtaPhiMLorentzVector",
                      behavior=coffea.nanoevents.methods.vector.behavior)
    return TetraVec


def _invariant_mass(events: ak.Array) -> ak.Array:
    empty_events = ak.zeros_like(events, dtype=np.uint16)[:, 0:0]
    where = ak.num(events, axis=1) == 2
    events_with_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((TetraVec(events_with_2[:, :1]) + TetraVec(events_with_2[:, 1:2])).mass), 0)
    return mass


def deltaR(objects1: ak.Array,
           objects2: ak.Array) -> ak.Array:
    dr = objects1.metric_table(objects2)
    #print(dr)
    return ak.fill_none(ak.firsts(ak.firsts(dr, axis=-1)),0)
    

def invariant_mass(events: ak.Array):
    empty_events = ak.zeros_like(events, dtype=np.uint16)[:, 0:0]
    where = ak.num(events, axis=1) == 2
    events_2 = ak.where(where, events, empty_events)
    mass = ak.fill_none(ak.firsts((1 * events_2[:, :1] + 1 * events_2[:, 1:2]).mass), 0)
    return mass


def new_invariant_mass(objs1: ak.Array, objs2: ak.Array):
    lep_lep = ak.concatenate([objs1, objs2], axis=1)
    empty_events = ak.zeros_like(lep_lep, dtype=np.float32)[:, 0:0]
    where = ak.num(lep_lep, axis=1) == 2
    events = ak.where(where, lep_lep, empty_events)
    mass = ak.fill_none(ak.firsts((1 * events[:, :1] + 1 * events[:, 1:2]).mass), 0)
    return mass


def get_dataset_lfns(
        dataset_inst: od.Dataset,
        shift_inst: od.Shift,
        dataset_key: str,
) -> list[str]:
    # destructure dataset_key into parts and create the lfn base directory

    dataset_id, full_campaign, tier = dataset_key.split("/")[1:]
    main_campaign, sub_campaign = full_campaign.split("-", 1)
    print(dataset_id, full_campaign, tier)
    lfn_base = law.wlcg.WLCGDirectoryTarget(
        #f"/store/{dataset_inst.data_source}/{main_campaign}/{dataset_id}/{tier}/{sub_campaign}/0",
        f"/eos/cms/store/group/phys_tau/TauFW/nano/UL2017/{dataset_id}/{full_campaign}/{tier}",
        #f"/eos/user/g/gsaha3/Exotic/HCP_Test/UL2017/{dataset_id}/{full_campaign}/{tier}",
        fs=f"local",
    )
    # loop though files and interpret paths as lfns
    paths = [lfn_base.child(basename, type="f").path for basename in lfn_base.listdir(pattern="*.root")]
    print(paths)
    return paths
    
