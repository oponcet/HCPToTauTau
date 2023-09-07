# coding: utf-8

"""
Lepton Selection
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import

from hcp.selection.util import trigger_object_matching, IF_NANO_V9, IF_NANO_V11
from hcp.config.trigger_util import Trigger
from hcp.util import invariant_mass, deltaR

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        # nano columns
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mediumId", "Muon.tightId",
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
    is_double = trigger.has_tag("double_mu_mu")
    is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-muon mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])
    elif is_double:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[0].pdg_id) == 13
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Muon, events.TrigObj[leg_masks[0]])

    # pt sorted indices for converting masks to indices
    sorted_indices = ak.argsort(events.Muon.pt, axis=-1, ascending=False)

    # default muon mask, only required for single and cross triggers with muon leg
    default_mask = None
    default_indices = None
    if is_single or is_double:
        if is_2016:
            min_pt = 23.0 if is_single else 20.0
        else:
            min_pt = 33.0 if is_single else 25.0
        default_mask = (
            (events.Muon.tightId == 1) &
            (abs(events.Muon.eta) < 2.1) &
            (abs(events.Muon.dxy) < 0.045) &
            (abs(events.Muon.dz) < 0.2) &
            (events.Muon.pfRelIso04_all < 0.15) &
            (events.Muon.pt > min_pt) &
            matches_leg0
        )
        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)

    # veto muon mask
    veto_mask = (
        (events.Muon.mediumId == 1) &
        (abs(events.Muon.eta) < 2.4) &
        (abs(events.Muon.dxy) < 0.045) &
        (abs(events.Muon.dz) < 0.2) &
        (events.Muon.pfRelIso04_all < 0.3) &
        (events.Muon.pt > 10)
    )
    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    return default_indices, veto_indices


@selector(
    uses={
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.dxy", "Electron.dz",
        "Electron.pfRelIso03_all",
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
    is_double = trigger.has_tag("double_e_e")
    is_2016 = self.config_inst.campaign.x.year == 2016

    # start per-electron mask with trigger object matching
    if is_single:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 1
        assert abs(trigger.legs[0].pdg_id) == 11
        # match leg 0
        matches_leg0 = trigger_object_matching(events.Electron, events.TrigObj[leg_masks[0]])
    elif is_double:
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
        # mva_noniso_wp90 = events.Electron.mvaNoIso_WP90
    else:
        # <= nano v9
        mva_iso_wp80 = events.Electron.mvaFall17V2Iso_WP80
        mva_iso_wp90 = events.Electron.mvaFall17V2Iso_WP90
        # mva_noniso_wp90 = events.Electron.mvaFall17V2noIso_WP90

    # default electron mask, only required for single and cross triggers with electron leg
    default_mask = None
    default_indices = None
    if is_single or is_double:
        min_pt = 26.0 if is_2016 else (33.0 if is_single else 25.0)
        default_mask = (
            (mva_iso_wp80 == 1) &
            (abs(events.Electron.eta) < 2.1) &
            (abs(events.Electron.dxy) < 0.045) &
            (abs(events.Electron.dz) < 0.2) &
            (events.Electron.pt > min_pt) &
            matches_leg0
        )
        # convert to sorted indices
        default_indices = sorted_indices[default_mask[sorted_indices]]
        default_indices = ak.values_astype(default_indices, np.int32)

    # veto electron mask
    veto_mask = (
        (
            (mva_iso_wp90 == 1) |
            False
            # disabled as part of the resonant synchronization effort
            # ((mva_noniso_wp90 == 1) & (events.Electron.pfRelIso03_all < 0.3))
        ) &
        (abs(events.Electron.eta) < 2.5) &
        (abs(events.Electron.dxy) < 0.045) &
        (abs(events.Electron.dz) < 0.2) &
        (events.Electron.pt > 10.0)
    )
    # convert to sorted indices
    veto_indices = sorted_indices[veto_mask[sorted_indices]]
    veto_indices = ak.values_astype(veto_indices, np.int32)

    return default_indices, veto_indices


@selector(
    uses={
        electron_selection, muon_selection,
        # nano columns
        "event", "Electron.charge", "Muon.charge", "Electron.mass", "Muon.mass",
    },
    produces={
        electron_selection, muon_selection,
        # new columns
        "channel_id", "leptons_os", "leptons_ss", "single_triggered", "double_triggered",
        #"m_ll", "dr_ll",
    },
)
def lepton_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Combined lepton selection.
    """
    # get channels from the config
    ch_ee = self.config_inst.get_channel("ee")
    ch_mumu = self.config_inst.get_channel("mumu")

    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask
    leptons_os = false_mask
    leptons_ss = false_mask
    single_triggered = false_mask
    double_triggered = false_mask
    empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    sel_electron_indices = empty_indices
    sel_muon_indices = empty_indices
    #m_ll = false_mask
    #dr_ll = false_mask
    
    # perform each lepton election step separately per trigger
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        is_single = trigger.has_tag("single_trigger")
        is_double = trigger.has_tag("double_trigger")

        # electron selection
        electron_indices, electron_veto_indices = self[electron_selection](
            events,
            trigger,
            leg_masks,
            call_force=True,
            **kwargs,
        )

        # muon selection
        muon_indices, muon_veto_indices = self[muon_selection](
            events,
            trigger,
            leg_masks,
            call_force=True,
            **kwargs,
        )
        #print(f"muon_indices: {muon_indices}")
        
        # lepton pair selecton per trigger via lepton counting
        if trigger.has_tag({"single_e", "double_e_e"}):
            # expect 1 electron, 1 veto electron (the same one), 0 veto muons
            is_ee = (
                trigger_fired &
                (ak.num(electron_indices, axis=1) == 2) &
                (ak.num(electron_veto_indices, axis=1) == 2) &
                (ak.num(muon_veto_indices, axis=1) == 0) &
                (invariant_mass(events.Electron[electron_indices]) > 10) &
                (invariant_mass(events.Electron[electron_indices]) < 160)
            )
            # determine the os/ss charge sign relation
            is_os = ak.sum(events.Electron[electron_indices].charge, axis=-1) == 0
            #print(f"is_os: {is_os}")
            is_ss = ak.sum(events.Electron[electron_indices].charge, axis=-1) != 0
            # store global variables
            where = (channel_id == 0) & is_ee
            channel_id = ak.where(where, ch_ee.id, channel_id)
            leptons_os = ak.where(where, is_os, leptons_os)
            leptons_ss = ak.where(where, is_ss, leptons_ss)
            single_triggered = ak.where(where & is_single, True, single_triggered)
            double_triggered = ak.where(where & is_double, True, double_triggered)
            sel_electron_indices = ak.where(where, electron_indices, sel_electron_indices)
            #print(f"Electrons: {events.Electron[sel_electron_indices]}")
            #_m_ll = invariant_mass(events.Electron[electron_indices])
            #m_ll = ak.where(where, _m_ll, m_ll)
            #_dr_ll = deltaR(events.Electron[electron_indices][:,:1], events.Electron[electron_indices][:,1:2]) 
            #dr_ll = ak.where(where, _dr_ll, dr_ll)
            
        elif trigger.has_tag({"single_mu", "double_mu_mu"}):
            # expect 1 muon, 1 veto muon (the same one), 0 veto electrons
            is_mumu = (
                trigger_fired &
                (ak.num(muon_indices, axis=1) == 2) &
                (ak.num(muon_veto_indices, axis=1) == 2) &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (invariant_mass(events.Muon[muon_indices]) > 10) &
                (invariant_mass(events.Muon[muon_indices]) < 160)
            )
            # determine the os/ss charge sign relation
            is_os = ak.sum(events.Muon[muon_indices].charge, axis=-1) == 0
            is_ss = ak.sum(events.Muon[muon_indices].charge, axis=-1) != 0
            # store global variables
            where = (channel_id == 0) & is_mumu
            channel_id = ak.where(where, ch_mumu.id, channel_id)
            leptons_os = ak.where(where, is_os, leptons_os)
            leptons_ss = ak.where(where, is_ss, leptons_ss)
            single_triggered = ak.where(where & is_single, True, single_triggered)
            double_triggered = ak.where(where & is_double, True, double_triggered)
            sel_muon_indices = ak.where(where, muon_indices, sel_muon_indices)
            #_m_ll = invariant_mass(events.Muon[muon_indices])
            #m_ll = ak.where(where, _m_ll, m_ll)
            #_dr_ll = deltaR(events.Muon[muon_indices][:,:1], events.Muon[muon_indices][:,1:2])
            #dr_ll = ak.where(where, _dr_ll, dr_ll)
            
    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    leptons_os = ak.fill_none(leptons_os, False)
    leptons_ss = ak.fill_none(leptons_ss, False)
    sel_electron_indices = ak.values_astype(sel_electron_indices, np.int32)
    sel_muon_indices = ak.values_astype(sel_muon_indices, np.int32)
    #m_ll = ak.values_astype(m_ll, np.float32)
    #print(f"m_ll: {m_ll}")
    #dr_ll = ak.values_astype(dr_ll, np.float32)
    #print(f"dr_ll: {dr_ll}")

    #print(f"channelId: {channel_id}")
    #print(f"leptons_os: {leptons_os}")
    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    events = set_ak_column(events, "leptons_os", leptons_os)
    events = set_ak_column(events, "leptons_ss", leptons_ss)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "double_triggered", double_triggered)
    #events = set_ak_column(events, "m_ll", m_ll)
    #events = set_ak_column(events, "dr_ll", dr_ll)
    
    #print(f"Column m_ll: {events.m_ll}")
    
    return events, SelectionResult(
        steps={
            "lepton": channel_id != 0,
        },
        objects={
            "Electron": {
                "Electron": sel_electron_indices,
            },
            "Muon": {
                "Muon": sel_muon_indices,
            },
        },
        aux={
            # save the selected lepton pair for the duration of the selection
            # multiplication of a coffea particle with 1 yields the lorentz vector
            "lepton_pair": ak.concatenate(
                [
                    events.Electron[sel_electron_indices] * 1,
                    events.Muon[sel_muon_indices] * 1,
                ],
                axis=1,
            ),
        },
    )
