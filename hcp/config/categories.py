# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    add_category(
        config,
        name="incl",
        id=1,
        selection="cat_incl",
        label="inclusive",
    )
    add_category(
        config,
        name="2j",
        id=100,
        selection="cat_2j",
        label="2 jets",
    )
    add_category(
        config,
        name="ele_tau",
        id=101,
        selection="sel_etau",
        label="etau_channel",
    )
    add_category(
        config,
        name="mu_tau",
        id=102,
        selection="sel_mutau",
        label="mutau_channel",
    )
    add_category(
        config,
        name="tau_tau",
        id=103,
        selection="sel_tautau",
        label="tautau_channel",
    )
    add_category(
        config,
        name="os",
        id=201,
        selection="sel_os",
        label="os_channel",
    )
    #add_category(
    #    config,
    #    name="ss",
    #    id=202,
    #    selection="sel_ss",
    #    label="ss_channel",
    #)
