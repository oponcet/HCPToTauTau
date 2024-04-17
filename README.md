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



### Tips and trick :hear_no_evil:
Welcome to the Tips and Tricks section! Here, you'll find recipes to make your journey with ColumnFlow smoother and more efficient. Let's unlock the secrets to mastering ColumnFlow with ease! :rocket:


#### How to use ROOT in columnflow :deciduous_tree:

Before each session and before configuring your ColumnFlow setup, ensure that you import the necessary library. For ROOT, if you're utilizing lxplus, you can execute the following command:

```
LD_LIBRARY_PATH=/usr/local/lib
CPPYY_BACKEND_LIBRARY=/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/lib/libcppyy_backend3_9.so
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/bin/thisroot.sh 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/lib/
source setup.sh enveos 
CPPYY_BACKEND_LIBRARY=/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/lib/libcppyy_backend3_9.so
source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/bin/thisroot.sh 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/lib/
```

If you want to use ROOT in one of your producer/selector/calibrator, at the beginning of your file, import the ROOT module :

```
root = maybe_import("ROOT")
```
You can check if the module has succesfully been imported by just printing :
```
print("root module : ", root)
``` 
Then you can simply use it to plot some ak.array like : 

```
# Convert your akward array to numpy array : 
leps1_mass = ak.to_numpy(leps1["mass"])
# Create a TH1D histogram for metx
hist1 = root.TH1D("hist_mass_leps1", "mass", 50, 0, 5)

# Fill the histogram with metx values
for mass in leps1_mass:
            hist1.Fill(mass)

# Draw the histogram
hist1.Draw()

# Save the histogram in a root file
root_file = root.TFile("hist_mass1.root", "RECREATE")
hist1.Write()
root_file.Close()
```

See example of producer using ROOT : [svfit.py](https://github.com/oponcet/HCPToTauTau/blob/SVFIT_dev/hcp/production/svfit.py)

#### How to use ClassicSVFit in columnflow? :star2:
The [ClassicSVFit](https://github.com/SVfit/ClassicSVfit/tree/fastMTT_19_02_2019) package is used for di-tau mass reconstruction. 
Version used : `fastMTT_19_02_2019`
Installation path in hcp analysis: [HCPToTauTau/module/extern](https://github.com/oponcet/HCPToTauTau/tree/SVFIT_dev/modules/extern/TauAnalysis)

##### Installation instructions:
The installation is done without CMSSW because of incompatibility of columnflow (using gfal) and CMSSW. It is possible to build the software without CMSSW framework if the following prerequisites are satisfied (oldest software version the instructions were tested with):

- ROOT (6.10/3 or newer)
- GCC (6.3 or newer)

In order to install the code, execute:
```
git clone https://github.com/SVfit/ClassicSVfit TauAnalysis/ClassicSVfit -b fastMTT_19_02_2019
export LIBRARY_PATH=$LIBRARY_PATH:$PWD/TauAnalysis/ClassicSVfit/lib
make -f TauAnalysis/ClassicSVfit/Makefile -j4
```
If installation is not succeding because of compilation error, you need to  add copy constructor : in [`TauAnalysis/ClassicSVfit/interface/MeasuredTauLepton.h`](https://github.com/SVfit/ClassicSVfit/blob/fastMTT_19_02_2019/interface/MeasuredTauLepton.h):

```
// Copy assignment operator
    MeasuredTauLepton& operator=(const MeasuredTauLepton& other)
    {
      if (this != &other) // protect against invalid self-assignment
      {
        // 1: deallocate old memory
        // No dynamic memory allocation in this class, so no deallocation needed

        // 2: copy the elements (deep copy)
        type_ = other.type_;
        pt_ = other.pt_;
        eta_ = other.eta_;
        phi_ = other.phi_;
        mass_ = other.mass_;
        energy_ = other.energy_;
        px_ = other.px_;
        py_ = other.py_;
        pz_ = other.pz_;
        p_ = other.p_;
        decayMode_ = other.decayMode_;
        p4_ = other.p4_;
        p3_ = other.p3_;

        // 3: assign the new memory to the object
      }
      return *this;
    }
```
You can try running SVFit with :
```
./TauAnalysis/ClassicSVfit/exec/testClassicSVfit
```
This should produce a root file.

##### ClassicSVFit and columnflow 
In order to use ClassicSVFit in columnflow, the [pybind11](https://pybind11.readthedocs.io/en/stable/basics.html) wrapper has been used. The wrapper for the different class can be found here : [pybind_wrapper.cpp](https://github.com/oponcet/ClassicSVfit/blob/fastMTT_19_02_2019/wrapper/pybind_wrapper.cpp)

This wrapper need to be compile with : 
```
export LIBRARY_PATH=$LIBRARY_PATH:$PWD/TauAnalysis/ClassicSVfit/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cern.ch/user/o/oponcet/private/analysis/HCPToTauTau/modules/extern/TauAnalysis/ClassicSVfit/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.30.04/x86_64-centosstream9-gcc113-opt/lib/
cd modules/extern/
cmake -S TauAnalysis/ClassicSVfit/wrapper/ -B TauAnalysis/ClassicSVfit/wrapper/
make -C TauAnalysis/ClassicSVfit/wrapper/
```

It should produce a `.so` file which can be used as a module in columnflow. For example you can import it like :

```
from modules.extern.TauAnalysis.ClassicSVfit.wrapper.pybind_wrapper import *
```

#### Plot quickly ak.array :bar_chart:	
Sometimess it can be nice to quickly have a look a the distribtuion of your favorite varaible. For this you can include the following modules: 
```
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
```

And save your distribtuion in few lines:

```
# Make the plot
fig, ax = plt.subplots()
plt.hist(ak.to_numpy(leps1["mass"]).flatten(), bins=20, range=(0, 2), alpha = 0.5)
plt.savefig("mass.pdf") 
```