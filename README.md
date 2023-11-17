# HCPToTauTau Analysis

Quickstart:
- Clone/Fork+Clone the [repository](https://github.com/gsaha009/HCPToTauTau.git)
- `git checkout TauTauDev`
- `source setup.sh hcpenv`: It will install all the dependencies for the 1st time only. Later, it will just activate the environment.
- copy `/afs/cern.ch/work/g/gsaha/public/IPHC/Work/ColumnFlowAnalyses/HCPToTauTau/modules/cmsdb/cmsdb/campaigns/run2_2017_nano_local_v9/` to `<your path>/HCPToTauTau/modules/cmsdb/cmsdb/campaigns/`: I'm sure that its a stupid way!
- To run `SelectEvents` task: `law run cf.SelectEvents --version v1 --dataset dy_lep_m50_2j_madgraph --branch 0`
- To run `Plotting` task: `law run cf.PlotVariables1D --version v1 --datasets dy_lep_m50_2j_madgraph --variables hlepton_1_pt,hlepton_2_pt,hcand_invmass,hcand_dr --categories tau_tau,ele_tau,mu_tau`


### Resources

- [columnflow](https://github.com/columnflow/columnflow)
- [law](https://github.com/riga/law)
- [order](https://github.com/riga/order)
- [luigi](https://github.com/spotify/luigi)
