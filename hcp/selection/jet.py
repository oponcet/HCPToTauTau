# coding: utf-8

"""
Jet selection
"""


from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.util import sorted_indices_from_mask
from columnflow.util import maybe_import

np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.jetId", "Jet.puId", "Jet.btagDeepFlavB",
        "hcand.pt", "hcand.eta", "hcand.phi", "hcand.mass",
    },
)
def jet_selection(
        self: Selector,
        events: ak.Array,
        **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    is_2016 = self.config_inst.campaign.x.year == 2016

    # nominal selection
    default_mask = (
        (events.Jet.pt > 30.0) &
        (abs(events.Jet.eta) < 2.4) &
        (events.Jet.jetId == 6) &
        (  # tight plus lepton veto
            (events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4))
        ) &
        ak.all(events.Jet.metric_table(events.hcand) > 0.5, axis=2)
    )
    # pt sorted indices to convert mask
    indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    jet_indices = indices[default_mask]

    # b-tagged jets, tight working point
    wp_tight = self.config_inst.x.btag_working_points.deepjet.tight
    bjet_mask = (default_mask) & (events.Jet.btagDeepFlavB >= wp_tight)
    bjet_indices = indices[bjet_mask]

    bjet_sel = ak.num(bjet_indices, axis=1) == 0

    # build and return selection results
    # "objects" maps source columns to new columns and selections to be applied on the old columns
    # to create them, e.g. {"Jet": {"MyCustomJetCollection": indices_applied_to_Jet}}
    return events, SelectionResult(
        steps={
            "bjet": bjet_sel,
        },
        objects={
            "Jet": {
                "Jet": jet_indices,
                "BJet": bjet_indices,
            },
        },
        aux={
            "n_jets": ak.num(jet_indices, axis=1),
        },
    )
