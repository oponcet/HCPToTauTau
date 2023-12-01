"""
Dilepton veto Selection
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import, DotDict

from hcp.selection.muon import muon_dl_veto_selection
from hcp.selection.electron import electron_dl_veto_selection

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        muon_dl_veto_selection, electron_dl_veto_selection,
        # nano columns
        "event",
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass", "Electron.charge",
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass", "Muon.charge",
    },
    produces={
        muon_dl_veto_selection, electron_dl_veto_selection,
        # new columns
    },
)
def dilep_res_veto_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    print("find low mass and Z mass resonances")
    dl_electron_veto_indices = self[electron_dl_veto_selection](events)
    #print("bla-0")
    dl_muon_veto_indices = self[muon_dl_veto_selection](events)
    #print("bla-1")
    #dummy = ak.zeros_like(1*events.event, dtype=np.float32)
    eles = events.Electron[dl_electron_veto_indices]
    muos = events.Muon[dl_muon_veto_indices]
    #print("bla-2")
    eles_comb = ak.combinations(eles, 2)
    muos_comb = ak.combinations(muos, 2)
    #print("bla-3")
    leps_comb = ak.concatenate([eles_comb,muos_comb], axis=1)
    #print("bla-4")
    leps1, leps2 = ak.unzip(leps_comb)
    #print("bla-5")
    charge_pair = (leps1.charge + leps2.charge)
    invm_pair = (1*leps1 + 1*leps2).mass 
    dr_pair = (1*leps1).delta_r(1*leps2)
    #from IPython import embed; embed()

    #print("bla-6")
    base_mask = (dr_pair > 0.15)
    
    low_mass_res_mask = (base_mask
                         & (invm_pair < 12)) 
    z_mass_res_mask = (base_mask
                       & ((invm_pair > 70) | (invm_pair < 110))
                       & (charge_pair == 0))
    #print("bla-7")
    res_mask = (low_mass_res_mask | z_mass_res_mask)

    invm = invm_pair[res_mask]
    dr = dr_pair[res_mask]
    #print("bla-8")
    no_res_mask = ak.num(invm, axis=1) == 0
    #print("bla-9")
    
    return events, SelectionResult(
        steps={
            "dl_veto": no_res_mask,
        },
        aux={
            "res_leps_1": leps1[res_mask],
            "res_leps_2": leps2[res_mask],
        },
    )
