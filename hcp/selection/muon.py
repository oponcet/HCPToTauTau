# coding: utf-8

"""
Muon selection
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, selector
from columnflow.util import maybe_import

from hcp.config.trigger_util import Trigger
from hcp.util import trigger_object_matching

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        # nano columns
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mediumId",
        "Muon.pfRelIso04_all", "Muon.dxy", "Muon.dz",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
    },
    exposed=False,
)
def muon_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Muon selection returning two sets of indidces for default and veto muons.

    References:

    - Isolation working point: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2?rev=59
    - ID und ISO : https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017?rev=15
    """
    is_single = trigger.has_tag("single_mu")
    is_cross = trigger.has_tag("cross_mu_tau")
    # is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-muon mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])
    elif is_cross:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)

    default_mask = None
    default_indices = None

    # veto muon mask
    veto_mask = (
        (events.Muon.pt > 10)
        & (abs(events.Muon.eta) < 2.4)
        & (abs(events.Muon.dz) < 0.2)
        & (abs(events.Muon.dxy) < 0.045)
        & (events.Muon.mediumId == 1)
        & (events.Muon.pfRelIso04_all < 0.3)
    )

    if is_single or is_cross:
        minpt = trigger.legs[0].min_pt
        default_mask = (
            matches_leg0
            & (events.Muon.pt > minpt)
            & (abs(events.Muon.eta) < 2.4)
            & (abs(events.Muon.dxy) < 0.045)
            & (abs(events.Muon.dz) < 0.2)
            & (events.Muon.mediumId == 1)
            & (events.Muon.pfRelIso04_all < 0.15)
        )
        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)

        # muons passing the veto mask but not the default one
        veto_mask = veto_mask & ~default_mask
        
    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    return default_indices, veto_indices




@selector(
    uses={
        # nano columns
        "Muon.pt", "Muon.eta", "Muon.pfRelIso04_all", "Muon.dxy", "Muon.dz",
        "Muon.isGlobal", "Muon.isPFcand", "Muon.isTracker",
    },
    exposed=False,
)
def muon_dl_veto_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)

    # DL veto mask
    dl_veto_mask = (
        (events.Muon.pt > 15.0)
        & (abs(events.Muon.eta) < 2.4)
        & (events.Muon.isGlobal == True)
        & (events.Muon.isPFcand == True)
        & (events.Muon.isTracker ==True)
        & (abs(events.Muon.dz) < 0.2)
        & (abs(events.Muon.dxy) < 0.045)
        & (events.Muon.pfRelIso04_all < 0.3)
    )

    # convert to sorted indices
    dl_veto_indices = sorted_indices[dl_veto_mask[sorted_indices]]
    dl_veto_indices = ak.values_astype(dl_veto_indices, np.int32)

    return dl_veto_indices
