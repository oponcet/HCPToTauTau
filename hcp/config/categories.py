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
        name="ele_ele",
        id=101,
        selection="sel_ee",
        label="ee_channel",
    )
    add_category(
        config,
        name="mu_mu",
        id=102,
        selection="sel_mm",
        label="mm_channel",
    )
    add_category(
        config,
        name="os",
        id=201,
        selection="sel_os",
        label="os_channel",
    )
    add_category(
        config,
        name="ss",
        id=202,
        selection="sel_ss",
        label="ss_channel",
    )
