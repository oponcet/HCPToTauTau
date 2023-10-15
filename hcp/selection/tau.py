# coding: utf-8

"""
Tau Selection
https://cms.cern.ch/iCMS/analysisadmin/cadilines?id=2325&ancode=HIG-20-006&tp=an&line=HIG-20-006
http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_192_v15.pdf
"""

from columnflow.selection import Selector, selector
from columnflow.util import maybe_import, DotDict

from hcp.config.trigger_util import Trigger
from hcp.util import invariant_mass, deltaR, new_invariant_mass, trigger_object_matching, IF_NANO_V9, IF_NANO_V11, get_nleps_dl_veto

np = maybe_import("numpy")
ak = maybe_import("awkward")



@selector(
    uses={
        # nano columns
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.dz", "Tau.idDeepTau2017v2p1VSe",
        "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet",
        "TrigObj.pt", "TrigObj.eta", "TrigObj.phi",
        "Electron.pt", "Electron.eta", "Electron.phi",
        "Muon.pt", "Muon.eta", "Muon.phi",
    },
    # shifts are declared dynamically below in tau_selection_init
    exposed=False,
)
def tau_selection(
    self: Selector,
    events: ak.Array,
    trigger: Trigger,
    leg_masks: list[ak.Array],
    electron_indices: ak.Array,
    muon_indices: ak.Array,
    **kwargs,
) -> tuple[ak.Array, ak.Array]:
    """
    Tau selection returning a set of indices for taus that are at least VVLoose isolated (vs jet)
    and a second mask to select the action Medium isolated ones, eventually to separate normal and
    iso inverted taus for QCD estimations.

    TODO: there is no decay mode selection yet, but this should be revisited!
    """
    is_single_e = trigger.has_tag("single_e")
    is_single_mu = trigger.has_tag("single_mu")
    is_cross_e = trigger.has_tag("cross_e_tau")
    is_cross_mu = trigger.has_tag("cross_mu_tau")
    is_cross_tau = trigger.has_tag("cross_tau_tau")
    is_any_cross_tau = is_cross_tau
    is_2016 = self.config_inst.campaign.x.year == 2016
    # tau id v2.1 working points (binary to int transition after nano v10)
    if self.config_inst.campaign.x.version < 10:
        # https://cms-nanoaod-integration.web.cern.ch/integration/master/mc94X_doc.html
        tau_vs_e = DotDict(vvloose=2, vloose=4)
        tau_vs_mu = DotDict(vloose=1, tight=8)
        tau_vs_jet = DotDict(vvloose=2, loose=8, medium=16)
    else:
        # https://cms-nanoaod-integration.web.cern.ch/integration/cms-swmaster/data106Xul17v2_v10_doc.html#Tau
        tau_vs_e = DotDict(vvloose=2, vloose=3)
        tau_vs_mu = DotDict(vloose=1, tight=4)
        tau_vs_jet = DotDict(vvloose=2, loose=4, medium=5)

    # start per-tau mask with trigger object matching per leg
    if is_cross_e or is_cross_mu:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) == 2
        assert abs(trigger.legs[1].pdg_id) == 15
        # match leg 1
        matches_leg1 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[1]])
    elif is_cross_tau:
        # catch config errors
        assert trigger.n_legs == len(leg_masks) >= 2
        assert abs(trigger.legs[0].pdg_id) == 15
        assert abs(trigger.legs[1].pdg_id) == 15
        # match both legs
        matches_leg0 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[0]])
        matches_leg1 = trigger_object_matching(events.Tau, events.TrigObj[leg_masks[1]])

    # determine minimum pt and maximum eta
    if is_single_e or is_single_mu:
        #min_pt = 20.0
        #max_eta = 2.3
        min_pt = 40.0
        max_eta = 2.3
    elif is_cross_e:
        # only existing after 2016, so force a failure in case of misconfiguration
        # min_pt = None if is_2016 else 35.0
        min_pt = trigger.legs[1].min_pt
        max_eta = 2.1
    elif is_cross_mu:
        # min_pt = 25.0 if is_2016 else 32.0
        min_pt = trigger.legs[1].min_pt
        max_eta = 2.1
    elif is_cross_tau:
        min_pt = 40.0
        max_eta = 2.1

    # base tau mask for default and qcd sideband tau
    base_mask = (
        (events.Tau.pt > min_pt)
        & (abs(events.Tau.eta) < max_eta)
        & (abs(events.Tau.dz) < 0.2)
        #(events.Tau.decayModeFindingNewDMs >= 0.5) &
        & (events.Tau.idDeepTau2017v2p1VSe >= (tau_vs_e.vvloose if is_any_cross_tau else tau_vs_e.vloose))
        & (events.Tau.idDeepTau2017v2p1VSmu >= (tau_vs_mu.vloose if is_any_cross_tau else tau_vs_mu.tight))
        & (events.Tau.idDeepTau2017v2p1VSjet >= tau_vs_jet.loose)
    )

    # remove taus with too close spatial separation to previously selected leptons
    if electron_indices is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Electron[electron_indices]) > 0.5, axis=2)
    if muon_indices is not None:
        base_mask = base_mask & ak.all(events.Tau.metric_table(events.Muon[muon_indices]) > 0.5, axis=2)

    # add trigger object masks
    if is_cross_e or is_cross_mu:
        base_mask = base_mask & matches_leg1
    elif is_cross_tau:
        # taus need to be matched to at least one leg, but as a side condition
        # each leg has to have at least one match to a tau
        base_mask = base_mask & (
            (matches_leg0 | matches_leg1) &
            ak.any(matches_leg0, axis=1) &
            ak.any(matches_leg1, axis=1)
        )

    # indices for sorting first by isolation, then by pt
    # for this, combine iso and pt values, e.g. iso 255 and pt 32.3 -> 2550032.3
    #f = 10 ** (np.ceil(np.log10(ak.max(events.Tau.pt))) + 1)
    #sort_key = events.Tau.idDeepTau2017v2p1VSjet * f + events.Tau.pt
    sort_key = events.Tau.rawDeepTau2017v2p1VSjet
    sorted_indices = ak.argsort(sort_key, axis=-1, ascending=False)

    # convert to sorted indices
    base_indices = sorted_indices[base_mask[sorted_indices]]
    base_indices = ak.values_astype(base_indices, np.int32)

    # additional mask to select final, Medium isolated taus
    # iso_mask = events.Tau[base_indices].idDeepTau2017v2p1VSjet >= tau_vs_jet.medium

    return base_indices#, iso_mask


@tau_selection.init
def tau_selection_init(self: Selector) -> None:
    # register tec shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag("tec")
    }
