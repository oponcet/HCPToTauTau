# coding: utf-8

"""
Lepton Selection
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.columnar_util import set_ak_column
from columnflow.util import maybe_import, DotDict

from hcp.selection.muon import muon_selection, muon_dl_veto_selection
from hcp.selection.electron import electron_selection, electron_dl_veto_selection
from hcp.selection.tau import tau_selection

from hcp.config.trigger_util import Trigger
from hcp.util import invariant_mass, deltaR, new_invariant_mass, trigger_object_matching, IF_NANO_V9, IF_NANO_V11 #, get_nleps_dl_veto

np = maybe_import("numpy")
ak = maybe_import("awkward")



@selector(
    uses={
        electron_selection, muon_selection, tau_selection,
        #muon_dl_veto_selection, electron_dl_veto_selection,
        # nano columns
        "event", "Electron.charge", "Muon.charge", "Tau.charge", "Electron.mass", "Muon.mass",
        "Tau.mass",
    },
    produces={
        electron_selection, muon_selection, tau_selection,
        #muon_dl_veto_selection, electron_dl_veto_selection,
        # new columns
        "channel_id", "single_triggered", "cross_triggered",
        "m_ll", "dr_ll", #"tau2_isolated", "leptons_os", 
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
    ch_etau = self.config_inst.get_channel("etau")
    ch_mutau = self.config_inst.get_channel("mutau")
    ch_tautau = self.config_inst.get_channel("tautau")

    print(f"channels: {ch_etau}, {ch_mutau}, {ch_tautau}")
    
    # prepare vectors for output vectors
    false_mask = (abs(events.event) < 0)
    channel_id = np.uint8(1) * false_mask
    single_triggered = false_mask
    cross_triggered = false_mask
    empty_indices = ak.zeros_like(1 * events.event, dtype=np.uint16)[..., None][..., :0]
    sel_electron_indices = empty_indices
    sel_muon_indices = empty_indices
    sel_tau_indices = empty_indices
    m_ll = false_mask
    dr_ll = false_mask

    #dl_electron_veto_indices = self[electron_dl_veto_selection](events)
    #dl_muon_veto_indices = self[muon_dl_veto_selection](events)

    #n_dl_veto_leptons = get_nleps_dl_veto(events, dl_electron_veto_indices, dl_muon_veto_indices)
    #where = (
    #    (channel_id == 0)
    #    & (n_dl_veto_leptons == 0)
    #)
    
    # perform each lepton election step separately per trigger
    for trigger, trigger_fired, leg_masks in trigger_results.x.trigger_data:
        print(f"trigger: {trigger}")
        print(f"trigger_fired: {trigger_fired}")
        is_single = trigger.has_tag("single_trigger")
        is_cross = trigger.has_tag("cross_trigger")

        print(f"Triggered? is_single: {is_single} :: is_cross: {is_cross} ")
        
        # electron selection
        electron_indices, electron_veto_indices = self[electron_selection](
            events,
            trigger,
            leg_masks,
            call_force=True,
            **kwargs,
        )
        print("ele idx done")
        
        # muon selection
        muon_indices, muon_veto_indices = self[muon_selection](
            events,
            trigger,
            leg_masks,
            call_force=True,
            **kwargs,
        )
        print("muon idx done")

        # tau selection
        tau_indices = self[tau_selection](
            events,
            trigger,
            leg_masks,
            electron_indices,
            muon_indices,
            call_force=True,
            **kwargs,
        )
        print("tau idx done")

        # lepton pair selecton per trigger via lepton counting
        if trigger.has_tag({"single_e", "cross_e_tau"}):
            
            # expect at least 1 electron,
            # 0 veto electron (loose ele, but not the one selected already as trigger matched, otherwise loose ele only),
            # 0 veto muons (loose muons),
            # and at least one tau
            is_etau = (
                trigger_fired
                & (ak.num(electron_indices, axis=1) >= 1)
                & (ak.num(electron_veto_indices, axis=1) == 0)
                & (ak.num(muon_veto_indices, axis=1) == 0)
                & (ak.num(tau_indices, axis=1) >= 1)
            )
            
            #dr_etau = events.Electron[electron_indices].metric_table(events.Tau[tau_indices])
            chr_etau = events.Electron[electron_indices].metric_table(events.Tau[tau_indices], metric=lambda a, b: a.charge + b.charge)
            #dr_etau_mask = ak.any(dr_etau > 0.5, axis=2)
            #dr_etau_mask = ak.fill_none(ak.firsts(dr_etau_mask), False)
            chr_etau_mask = ak.any(chr_etau == 0, axis=2)
            chr_etau_mask = ak.fill_none(ak.firsts(chr_etau_mask), False)
            #where_etau = (channel_id == 0) & is_etau & dr_etau_mask & chr_etau_mask
            where_etau = (channel_id == 0) & is_etau & chr_etau_mask #& dr_etau_mask 
            #from IPython import embed; embed()
            #print(f"e-charge: {e_charge} | {e_charge.type}")
            #print(f"tau-charge: {tau_charge} | {tau_charge.type}")
            #print(f"WHERE: {where}")
            channel_id = ak.where(where_etau, ch_etau.id, channel_id)
            #tau2_isolated = ak.where(where, is_iso, tau2_isolated)
            #leptons_os = ak.where(where, is_os, leptons_os)
            single_triggered = ak.where(where_etau & is_single, True, single_triggered)
            cross_triggered = ak.where(where_etau & is_cross, True, cross_triggered)
            sel_electron_indices = ak.where(where_etau, electron_indices, sel_electron_indices)
            sel_tau_indices = ak.where(where_etau, tau_indices, sel_tau_indices)
            #print(f"ele idxs : {electron_indices}")
            #print(f"tau idxs : {tau_indices}")
            _m_ll = new_invariant_mass(events.Electron[electron_indices][:,:1], events.Tau[tau_indices][:,:1])
            #print(_m_ll)
            m_ll = ak.where(where_etau, _m_ll, m_ll)
            #print(m_ll)
            _dr_ll = deltaR(events.Electron[electron_indices][:,:1], events.Tau[tau_indices][:,:1]) 
            dr_ll = ak.where(where_etau, _dr_ll, dr_ll)

        elif trigger.has_tag({"single_mu", "cross_mu_tau"}):
            # expect 1 muon, 1 veto muon (the same one), 0 veto electrons, and at least one tau
            is_mutau = (
                trigger_fired &
                (ak.num(muon_indices, axis=1) >= 1) &
                (ak.num(muon_veto_indices, axis=1) == 0) &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 1)
            )
            #is_iso = ak.sum(tau_iso_mask, axis=1) >= 1
            #dr_mutau = events.Muon[muon_indices].metric_table(events.Tau[tau_indices])
            chr_mutau = events.Muon[muon_indices].metric_table(events.Tau[tau_indices], metric=lambda a, b: a.charge + b.charge)
            #dr_mutau_mask = ak.any(dr_mutau > 0.5, axis=2)
            #dr_mutau_mask = ak.fill_none(ak.firsts(dr_mutau_mask), False)
            chr_mutau_mask = ak.any(chr_mutau == 0, axis=2)
            chr_mutau_mask = ak.fill_none(ak.firsts(chr_mutau_mask), False)
            #where_mutau = (channel_id == 0) & is_mutau & dr_mutau_mask & chr_mutau_mask
            where_mutau = (channel_id == 0) & is_mutau & chr_mutau_mask #& dr_mutau_mask 
            channel_id = ak.where(where_mutau, ch_mutau.id, channel_id)
            #tau2_isolated = ak.where(where, is_iso, tau2_isolated)
            #leptons_os = ak.where(where, is_os, leptons_os)
            single_triggered = ak.where(where_mutau & is_single, True, single_triggered)
            cross_triggered = ak.where(where_mutau & is_cross, True, cross_triggered)
            sel_muon_indices = ak.where(where_mutau, muon_indices, sel_muon_indices)
            sel_tau_indices = ak.where(where_mutau, tau_indices, sel_tau_indices)
            _m_ll = new_invariant_mass(events.Muon[muon_indices][:,:1], events.Tau[tau_indices][:,:1])
            m_ll = ak.where(where_mutau, _m_ll, m_ll)
            _dr_ll = deltaR(events.Muon[muon_indices][:,:1], events.Tau[tau_indices][:,:1]) 
            dr_ll = ak.where(where_mutau, _dr_ll, dr_ll)

        elif trigger.has_tag("cross_tau_tau"):
            # expect 0 veto electrons, 0 veto muons and at least two taus of which one is isolated
            is_tautau = (
                trigger_fired &
                (ak.num(electron_veto_indices, axis=1) == 0) &
                (ak.num(muon_veto_indices, axis=1) == 0) &
                (ak.num(tau_indices, axis=1) >= 2)
                #(ak.sum(tau_iso_mask, axis=1) >= 1)
            )
            #from IPython import embed; embed()
            tau_pairs = ak.combinations(events.Tau[tau_indices], 2)
            #dr_tautau = (tau_pairs["0"]).metric_table(tau_pairs["1"])
            chr_tautau = tau_pairs["0"].charge + tau_pairs["1"].charge
            #dr_tautau_mask = ak.any(dr_tautau > 0.5, axis=2)
            #dr_tautau_mask = ak.fill_none(ak.firsts(dr_tautau_mask), False)
            chr_tautau_mask = ak.any(chr_tautau == 0, axis=1)
            #where_tautau = (channel_id == 0) & is_tautau & dr_tautau_mask & chr_tautau_mask
            where_tautau = (channel_id == 0) & is_tautau & chr_tautau_mask #& dr_tautau_mask
            channel_id = ak.where(where_tautau, ch_tautau.id, channel_id)
            #tau2_isolated = ak.where(where, is_iso, tau2_isolated)
            #leptons_os = ak.where(where, is_os, leptons_os)
            single_triggered = ak.where(where_tautau & is_single, True, single_triggered)
            cross_triggered = ak.where(where_tautau & is_cross, True, cross_triggered)
            sel_tau_indices = ak.where(where_tautau, tau_indices, sel_tau_indices)
            _m_ll = new_invariant_mass(events.Tau[tau_indices][:,:1], events.Tau[tau_indices][:,1:2])
            m_ll = ak.where(where_tautau, _m_ll, m_ll)
            _dr_ll = deltaR(events.Tau[tau_indices][:,:1], events.Tau[tau_indices][:,1:2]) 
            dr_ll = ak.where(where_tautau, _dr_ll, dr_ll)

    # some final type conversions
    channel_id = ak.values_astype(channel_id, np.uint8)
    #leptons_os = ak.fill_none(leptons_os, False)
    sel_electron_indices = ak.values_astype(sel_electron_indices, np.int32)
    sel_muon_indices = ak.values_astype(sel_muon_indices, np.int32)
    sel_tau_indices = ak.values_astype(sel_tau_indices, np.int32)
    m_ll = ak.values_astype(m_ll, np.float32)
    dr_ll = ak.values_astype(dr_ll, np.float32)

    
    # save new columns
    events = set_ak_column(events, "channel_id", channel_id)
    #events = set_ak_column(events, "leptons_os", leptons_os)
    #events = set_ak_column(events, "tau2_isolated", tau2_isolated)
    events = set_ak_column(events, "single_triggered", single_triggered)
    events = set_ak_column(events, "cross_triggered", cross_triggered)
    events = set_ak_column(events, "m_ll", m_ll)
    events = set_ak_column(events, "dr_ll", dr_ll)

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
            "Tau": {
                "Tau": sel_tau_indices,
            },
        },
        aux={
            "Electrons": events.Electron[sel_electron_indices],
            "Muons": events.Muon[sel_muon_indices],
            "Taus": events.Tau[sel_tau_indices],
        },
    )
