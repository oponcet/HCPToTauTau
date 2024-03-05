# coding: utf-8

"""
Configuration of the HCPToTauTau analysis.
"""

import os

import law
import order as od


#
# the main analysis object
#

analysis_hcp = ana = od.Analysis(
    name="analysis_hcp",
    id=1,
)

# analysis-global versions
# (see cfg.x.versions below for more info)
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$HCP_BASE/sandboxes/venv_columnar_tf.sh",
    law.config.get("analysis", "default_columnar_sandbox"),
    "$CF_BASE/sandboxes/venv_cmssw_sand.sh",
    "$HCP_BASE/sandboxes/venv_cmssw_sand.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    "$HCP_BASE/sandboxes/venv_cmssw_sand.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("HCP_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

from hcp.config.configs_run2ul_DY import add_config as add_config_run2ul_DY
from hcp.config.configs_run2ul_SR import add_config as add_config_run2ul_SR
#from cmsdb.campaigns.run2_2017_nano_local_v9 import campaign_run2_2017_nano_v9
from cmsdb.campaigns.run2_2017_nano_local_v10 import campaign_run2_2017_nano_local_v10

"""
add_config_run2ul_DY(
    analysis_hcp,
    campaign_run2_2017_nano_v9.copy(),
    config_name=campaign_run2_2017_nano_v9.name,
    config_id=2,
)
"""
add_config_run2ul_SR(
    analysis_hcp,
    campaign_run2_2017_nano_local_v10.copy(),
    config_name=campaign_run2_2017_nano_local_v10.name,
    config_id=2,
)
