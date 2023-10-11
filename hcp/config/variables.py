# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT

def add_common_features(cfg: od.config) -> None:
    """
    Adds common features
    """
    cfg.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    cfg.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    

def add_lepton_features(cfg: od.Config) -> None:
    """
    Adds lepton features only
    """
    for obj in ["Electron", "Muon", "Tau"]:
        for i in range(2):
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_pt",
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 200.),
                unit="GeV",
                x_title=obj + r" $p_{T}$",
            )
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_phi",
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(32, -3.2, 3.2),
                x_title=obj + r" $\phi$",
            )
            cfg.add_variable(
                name=f"{obj.lower()}_{i+1}_eta",
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(25, -2.5, 2.5),
                x_title=obj + r" $\eta$",
            )

    cfg.add_variable(
        name="m_ll",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 200),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    cfg.add_variable(
        name="m_ll_sel",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 200),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    cfg.add_variable(
        name="m_ele_ele",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 200),
        unit="GeV",
        x_title=r"$m_{ee}$",
    )
    cfg.add_variable(
        name="m_mu_mu",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 200),
        unit="GeV",
        x_title=r"$m_{mumu}$",
    )
    cfg.add_variable(
        name="dr_ll",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 5),
        x_title=r"$\Delta R(l,l)$",
    )
    cfg.add_variable(
        name="dr_ll_sel",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 5),
        x_title=r"$\Delta R(l,l)$",
    )
    cfg.add_variable(
        name="leptons_os",
        binning=(2, 0, 2),
        x_title=r"$\Delta R(l,l)$",
    )
    

def add_jet_features(cfg: od.Config) -> None:
    """
    Adds jet features only
    """
    cfg.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    cfg.add_variable(
        name="jets_pt",
        expression="Jet.pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$p_{T} of all jets$",
    )
    for i in range(2):
        cfg.add_variable(
            name=f"jet_{i+1}_pt",
            expression=f"Jet.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0.0, 400.0),
            unit="GeV",
            x_title=r"Jet $p_{T}$",
        )
        cfg.add_variable(
            name=f"jet_{i+1}_eta",
            expression=f"Jet.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(30, -3.0, 3.0),
            x_title=r"Jet $\eta$",
        )
    cfg.add_variable(
        name="ht",
        # expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        expression="ht",
        binning=(40, 0.0, 800.0),
        unit="GeV",
        x_title="HT",
    )
    

def add_highlevel_features(cfg: od.Config) -> None:    
    """
    Adds MET and other high-level features
    """
    cfg.add_variable(
        name="met",
        expression="MET.pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 100.0),
        x_title=r"MET",
    )
    

def add_weight_features(cfg: od.Config) -> None:
    """
    Adds weights
    """
    cfg.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    

def add_cutflow_features(cfg: od.Config) -> None:
    """
    Adds cf features
    """
    cfg.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )


def add_hcand_features(cfg: od.Config) -> None:
    """
    Adds h lepton features only
    """
    for i in range(2):
        cfg.add_variable(
            name=f"hlepton_{i+1}_pt",
            expression=f"hcand.pt[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 200.),
            unit="GeV",
            x_title=f"lepton_{i+1}" + r" $p_{T}$",
        )
        cfg.add_variable(
            name=f"hlepton_{i+1}_eta",
            expression=f"hcand.eta[:,{i}]",
            null_value=EMPTY_FLOAT,
            binning=(25, -2.5, 2.5),
            unit="GeV",
            x_title=f"lepton_{i+1}" + r" $\eta$",
        )
    cfg.add_variable(
        name="hcand_invmass",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 400),
        unit="GeV",
        x_title=r"$m_{ll}$",
    )
    cfg.add_variable(
        name="hcand_dr",
        null_value=EMPTY_FLOAT,
        binning=(40, 0, 5),
        x_title=r"$\Delta R(l,l)$",
    )

    
def add_variables(cfg: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    add_common_features(cfg)
    add_lepton_features(cfg)
    add_jet_features(cfg)
    add_highlevel_features(cfg)
    add_hcand_features(cfg)
    add_weight_features(cfg)
    add_cutflow_features(cfg)
