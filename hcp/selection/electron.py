# coding: utf-8

"""
Electron Selection
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, selector
from columnflow.util import maybe_import, DotDict

from hcp.config.trigger_util import Trigger
from hcp.util import trigger_object_matching, IF_NANO_V9, IF_NANO_V11

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.dxy", "Electron.dz",
        "Electron.pfRelIso03_all", "Electron.convVeto", "Electron.lostHits",
        IF_NANO_V9("Electron.mvaFall17V2Iso_WP80", "Electron.mvaFall17V2Iso_WP90", "Electron.mvaFall17V2noIso_WP90"),
        IF_NANO_V11("Electron.mvaIso_WP80", "Electron.mvaIso_WP90", "Electron.mvaNoIso_WP90"),
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
    },
    exposed=False,
)
def electron_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Electron selection returning two sets of indidces for default and veto electrons.
    See https://twiki.cern.ch/twiki/bin/view/CMS/EgammaNanoAOD?rev=4
    """
    is_single = trigger.has_tag("single_e")
    is_cross = trigger.has_tag("cross_e_tau")
    #is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-electron mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])
    elif is_cross:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)

    # obtain mva flags, which might be located at different routes, depending on the nano version
    if "mvaIso_WP80" in events.Electron.fields:
        # >= nano v10
        mva_iso_wp80 = events.Electron.mvaIso_WP80
        mva_iso_wp90 = events.Electron.mvaIso_WP90
        mva_noniso_wp90 = events.Electron.mvaNoIso_WP90
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        mva_noniso_wp90 = events.Electron.mvaFall17V2noIso_WP90

    # default electron mask, only required for single and cross triggers with electron leg
    default_mask = None
    default_indices = None
    
    # veto electron mask
    veto_mask = (
        (events.Electron.pt > 10.0)
        & (abs(events.Electron.eta) < 2.5)
        & (abs(events.Electron.dz) < 0.2)
        & (abs(events.Electron.dxy) < 0.045)
        & (mva_noniso_wp90 == 1)
        & (events.Electron.convVeto == 1)
        & (events.Electron.lostHits <= 1)
        & (events.Electron.pfRelIso03_all < 0.3)
    )

    if is_single or is_cross:
        minpt = trigger.legs[0].min_pt
        #min_pt = 26.0 if is_2016 else (33.0 if is_single else 25.0)
        default_mask = (
            matches_leg0
            & (events.Electron.pt > minpt)
            & (abs(events.Electron.eta) < 2.1)
            & (abs(events.Electron.dxy) < 0.045)
            & (abs(events.Electron.dz) < 0.2)
            & (mva_iso_wp80 == 1)
        )
        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)
        veto_mask = veto_mask & ~default_mask

    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    return default_indices, veto_indices



@selector(
    uses={
        # nano columns
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.dxy", "Electron.dz", "Electron.pfRelIso03_all", "Electron.cutBased",
    },
    exposed=False,
)
def electron_dl_veto_selection(
    self: Selector,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Electron.pt, axis=-1, ascending=False)

    # DL veto mask
    dl_veto_mask = (
        (events.Electron.pt > 15.0)
        & (abs(events.Electron.eta) < 2.5)
        & (events.Electron.cutBased == 1)
        & (abs(events.Electron.dz) < 0.2)
        & (abs(events.Electron.dxy) < 0.045)
        & (events.Electron.pfRelIso03_all < 0.3)
    )
    # convert to sorted indices
    dl_veto_indices = sorted_indices[dl_veto_mask[sorted_indices]]
    dl_veto_indices = ak.values_astype(dl_veto_indices, np.int32)
    
    return dl_veto_indices
