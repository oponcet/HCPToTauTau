"""
SVFit mH production methods.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.categories import category_ids
from columnflow.production.normalization import normalization_weights
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.selection.util import create_collections_from_masks
from columnflow.util import maybe_import
from columnflow.selection import SelectionResult
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from hcp.util import invariant_mass, deltaR, transverse_mass
from hcp.util import TetraVec, _invariant_mass, deltaR


from modules.extern.TauAnalysis.ClassicSVfit.wrapper.pybind_wrapper import *
import time

np = maybe_import("numpy")
ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")
root = maybe_import("ROOT")
hist = maybe_import("hist")
mpl = maybe_import("matplotlib")
plt = maybe_import("matplotlib.pyplot")
mplhep = maybe_import("mplhep")


# Print the version of the libraries 
# print_if_verbose(verb,"hist module : ", hist)
# print_if_verbose(verb,"root module : ", root)

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

# Function to print the output if the verbosity is greater than 0
def print_if_verbose(verbosity, *args):
    if verbosity > 0:
        print(*args)


@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.pfRelIso03_all",
        "Tau.*",
        "MET.*",
        "hcand.*",
    },
    # sandbox=dev_sandbox("bash::$HCP_BASE/sandboxes/venv_columnar_tf.sh"),
)
def svfit(self: Producer,
               events: ak.Array,
               **kwargs) -> ak.Array:

    print("\033[96m >>> Started SVFit production\033[0m")
    
    verb = 0

    print(breakhere)

    # Get the channel ID
    channel_id = events.channel_id

    print_if_verbose(verb, f"channel_id : {channel_id}")

    # decayType is used for SVFit to specify the decay type of the tau.
    decayType = channel_id

    # Assigns a value to decayType based on the value of channel_id:
    # If channel_id equals 3 (tautau), decayType is assigned 1 tauhdecayType,
    # otherwise, if channel_id equals 1 (mutau), decayType is assigned 3 mudecayType,
    # otherwise, if channel_id equals 2 (etau), decayType is assigned 2 edecayType,
    # if none of the conditions are met, decayType retains the value of channel_id.
    decayType = ak.where(channel_id == 3, 1, ak.where(channel_id == 2, 2, ak.where(channel_id == 1, 3, channel_id)))

    # kappa is used in SVFit to specify the number of parameters in the fit which depends on the  channels (etau, mutau, tautau). Also refered as k factor.
    kappa = channel_id

    # #  From SVFit 
    #// Define the k factor
    # double kappa; // use 3 for emu, 4 for etau and mutau, 5 for tautau channel
    kappa = ak.where(channel_id == 3, 5., ak.where(channel_id == 2, 4., ak.where(channel_id == 1, 4., 3)))

    kappa_np = ak.to_numpy(kappa)

    print_if_verbose(verb, f"decayType : {decayType}")
    print_if_verbose(verb, f"kappa : {kappa}")


    # Get MET
    print_if_verbose(verb, f"""Met fields: {events.MET.fields}""")

    # Get the metx and mety values
    metx = events.MET.pt * np.cos(events.MET.phi)
    mety = events.MET.pt * np.sin(events.MET.phi)

    print_if_verbose(verb,"event.MET.covXY:", events.MET.covXY)
    print_if_verbose(verb,"event.MET.covXX:", events.MET.covXX)
    print_if_verbose(verb,"event.MET.covYY:", events.MET.covYY)

    covXX = ak.to_numpy(events.MET.covXX)
    covXY = ak.to_numpy(events.MET.covXY)
    covYY = ak.to_numpy(events.MET.covYY)

    # Create a 2x2 matrix for each covariance matrix
    covMET = []
    covMET = np.zeros((len(covXX), 2, 2))
    # covMET is a diagonale matrix 
    covMET[:, 0, 0] = covXX
    covMET[:, 1, 0] = covXY
    covMET[:, 0, 1] = covXY 
    covMET[:, 1, 1] = covYY

    # Convert covMet to a list of TMatrixD objects for the SVFit algorithm
    rootMETMatrices = []

    for cov in covMET:
        matrix = root.TMatrixD(2, 2)
        matrix[0][0] = cov[0][0]
        matrix[1][0] = cov[1][0]
        matrix[0][1] = cov[0][1]
        matrix[1][1] = cov[1][1]
        rootMETMatrices.append(matrix)
    

    # Get the hcand
    h_cand = events.hcand

    print_if_verbose(verb, f"h_cand fields : {h_cand.fields}")


    # Get the leptons pairs from the hcand
    leps1 = events.hcand[:,:1] # e muon or tau
    leps2 = events.hcand[:,1:2] # always tau

    print_if_verbose(verb, f"leps1 fields : {leps1.fields}")
    print_if_verbose(verb, f"leps2 fields: {leps2.fields}")

    # Convert the Awkward Array to numpy array for leps1 
    leps1_pt = ak.to_numpy(leps1["pt"])
    leps1_eta = ak.to_numpy(leps1["eta"])
    leps1_phi = ak.to_numpy(leps1["phi"])
    leps1_mass = ak.to_numpy(leps1["mass"])
    leps1_dm = ak.to_numpy(leps1["decayMode"])
    leps1_flavor = ak.to_numpy(leps1["lepton"])
    
    print_if_verbose(verb, "leps1 flavour = ", leps1_flavor ) # 13 = muon, 11 = electron, 15 = tau
    print_if_verbose(verb,"leps1 pt = ", leps1_pt )
    print_if_verbose(verb,"leps1 eta = ", leps1_eta )
    print_if_verbose(verb,"leps1 phi = ", leps1_phi )
    print_if_verbose(verb,"leps1 mass = ", leps1_mass )
    print_if_verbose(verb,"leps1 decayMode = ", leps1_dm)
    

    # Make the plot
    # fig, ax = plt.subplots()
    # plt.style.use(mplhep.style.CMS)
    # plt.hist(variable.flatten(), bins=20, range=(0, 2), alpha = 0.5)
    # plt.hist(ak.to_numpy(leps1["mass"]).flatten(), bins=20, range=(0, 2), alpha = 0.5)
    # plt.savefig("mass.pdf")    

    # # Convert the Awkward Array to numpy array for leps1 
    # leps2_pt = ak.to_numpy(leps2["pt"])
    # leps2_eta = ak.to_numpy(leps2["eta"])
    # leps2_phi = ak.to_numpy(leps2["phi"])
    # leps2_mass = ak.to_numpy(leps2["mass"])
    # leps2_dm = ak.to_numpy(leps2["decayMode"])
    # leps2_flavor = ak.to_numpy(leps2["lepton"])

    # print_if_verbose(verb,"leps2 pt = ", leps2_pt )
    # print_if_verbose(verb,"leps2 eta = ", leps2_eta )
    # print_if_verbose(verb,"leps2 phi = ", leps2_phi )
    # print_if_verbose(verb,"leps2 mass = ", leps2_mass )
    # print_if_verbose(verb,"leps2 decayMode = ", leps2_dm)


    # # # # Create a TH1D histogram for metx
    # # hist1 = root.TH1D("hist_mass_leps1", "mass", 50, 0, 5)
    # # hist2 = root.TH1D("hist_mass_leps2", "mass", 50, 0, 5)

    # # # Fill the histogram with metx values
    # # for mass in leps1_mass:
    # #     hist1.Fill(mass)

    # # for mass in leps2_mass:
    # #     hist2.Fill(mass)

    # # # Draw the histogram
    # # hist1.Draw()

    # # # Save the histogram in a root file
    # # root_file = root.TFile("hist_mass1.root", "RECREATE")
    # # hist1.Write()
    # # root_file.Close()
    
    # # # Draw the histogram
    # # hist2.Draw()

    # # # Save the histogram in a root file
    # # root_file = root.TFile("hist_mass2.root", "RECREATE")
    # # hist2.Write()
    # # root_file.Close()

    # # Define the MeasuredTauLepton objects for the SVFit algorithm
    # leps1_MeasuredTauLepton = np.array([MeasuredTauLepton(*args) for args in zip(decayType, leps1_pt, leps1_eta, leps1_phi ,leps1_mass, leps1_dm)])
    # leps2_MeasuredTauLepton = np.array([MeasuredTauLepton(1,*args) for args in zip(leps2_pt, leps2_eta, leps2_phi ,leps2_mass, leps2_dm)])

    # print_if_verbose(verb,f"leps1_MeasuredTauLepton : {leps1_MeasuredTauLepton}")
    # print_if_verbose(verb,f"leps2_MeasuredTauLepton : {leps2_MeasuredTauLepton}")

    # svfit_verbosity = 0

    # print_if_verbose(verb,"svfit_verbosity : ", svfit_verbosity)
    # svFitAlgo = ClassicSVfit(svfit_verbosity)

    # massContraint = 91 # Z mass GeV
    # #massContraint = 125.06 # Higgs mass GeV

    # # Set the mass constraint
    # svFitAlgo.setDiTauMassConstraint(massContraint)

    # mass_array = []
    # err_mass_array = []
    
    # start_time = time.time()

    # # Loop over each event 
    # print(f"len(leps1_MeasuredTauLepton) : {len(leps1_MeasuredTauLepton)}")
    # for i in range(len(leps1_MeasuredTauLepton)):
    # #for i in range(1000):

    #     # Set the likelihood file name
    #     # svfit_filename = "testSVfit_hacand" + str(i) + ".root"
    #     svfit_filename = ""
    #     svFitAlgo.setLikelihoodFileName(svfit_filename)

    #     svFitAlgo.addLogM_fixed(True, kappa_np[i])
    #     # svFitAlgo.addLogM_dynamic(False)

    #     leps_MeasuredTauLepton = [leps1_MeasuredTauLepton[i], leps2_MeasuredTauLepton[i]]

    #     # svFitAlgo.prepareLeptonInput(leps_MeasuredTauLepton)
    #     # svFitAlgo.addMETEstimate(metx[i], mety[i], rootMETMatrices[i]) 
    #     # print_if_verbose(verb,"svFitAlgo.prepareLeptonInput= ", svFitAlgo.prepareLeptonInput(leps_MeasuredTauLepton))

    #     # Perform integration for each element
    #     print_if_verbose(verb,f"leps_MeasuredTauLepton[i] : {leps_MeasuredTauLepton}")
    #     svFitAlgo.integrate(leps_MeasuredTauLepton, metx[i], mety[i], rootMETMatrices[i])

    #     isValidSolution = svFitAlgo.isValidSolution()
    #     print_if_verbose(verb,f"isValidSolution : {isValidSolution}")
    #     mass = svFitAlgo.getHistogramAdapter().getMass()
    #     errmass = svFitAlgo.getHistogramAdapter().getMassErr()
    #     print( f"mass : {mass}")
    #     print( f"errmass : {errmass}")
    #     mass_array.append(mass)
    #     err_mass_array.append(errmass)
        
    # end_time = time.time()

    # execution_time_ = end_time - start_time
    
    # print(f"Execution time SVFit: {execution_time_} seconds")
        
    # print_if_verbose(verb,f"mass_array : {mass_array}")
    # print_if_verbose(verb,f"err_mass_array : {err_mass_array}")

    # # Make Mass plot
    # fig, ax = plt.subplots()
    # plt.style.use(mplhep.style.CMS)
    # plt.hist(mass_array, bins=130, range=(1, 131), alpha = 0.5)
    # plt.xlabel("Mass in GeV")
    # plt.ylabel("Events")
    # plt.savefig("mass_SVFit.pdf") 

    # # # Create a TH1D histogram for metx
    # hist1 = root.TH1D("mass_SVFit", "mass", 130, 0, 130)

    # # Fill the histogram with metx values
    # for mass in mass_array:
    #     hist1.Fill(mass)

    # # Draw the histogram
    # hist1.Draw()

    # # Save the histogram in a root file
    # root_file = root.TFile("mass_SVFit.root", "RECREATE")
    # hist1.Write()
    # root_file.Close()


    # print(f"h_cand : {breakhere}")
    return events



@producer(
    uses={
        "channel_id",
        "Electron.pt", "Electron.pfRelIso03_all",
        "Muon.pt", "Muon.pfRelIso03_all",
        "Tau.*",
        "MET.*",
        "hcand.*",
    },
    # sandbox=dev_sandbox("bash::$HCP_BASE/sandboxes/venv_columnar_tf.sh"),
)
def fastMTT(self: Producer,
               events: ak.Array,
               **kwargs) -> ak.Array:

    print("\033[96m >>> Started FastMTT production\033[0m")

    verb = 0

    # Get the channel ID
    channel_id = events.channel_id

    print_if_verbose(verb, f"channel_id : {channel_id}")

    # decayType is used for SVFit to specify the decay type of the tau.
    decayType = channel_id

    # Assigns a value to decayType based on the value of channel_id:
    # If channel_id equals 3 (tautau), decayType is assigned 1 tauhdecayType,
    # otherwise, if channel_id equals 1 (mutau), decayType is assigned 3 mudecayType,
    # otherwise, if channel_id equals 2 (etau), decayType is assigned 2 edecayType,
    # if none of the conditions are met, decayType retains the value of channel_id.
    decayType = ak.where(channel_id == 3, 1, ak.where(channel_id == 2, 2, ak.where(channel_id == 1, 3, channel_id)))

    # kappa is used in SVFit to specify the number of parameters in the fit which depends on the  channels (etau, mutau, tautau). Also refered as k factor.
    kappa = channel_id

    # #  From SVFit 
    #// Define the k factor
    # double kappa; // use 3 for emu, 4 for etau and mutau, 5 for tautau channel
    kappa = ak.where(channel_id == 3, 5., ak.where(channel_id == 2, 4., ak.where(channel_id == 1, 4., 3)))

    kappa_np = ak.to_numpy(kappa)

    print_if_verbose(verb, f"decayType : {decayType}")
    print_if_verbose(verb, f"kappa : {kappa}")


    # Get MET
    print_if_verbose(verb, f"""Met fields: {events.MET.fields}""")

    # Get the metx and mety values
    metx = events.MET.pt * np.cos(events.MET.phi)
    mety = events.MET.pt * np.sin(events.MET.phi)

    print_if_verbose(verb,"event.MET.covXY:", events.MET.covXY)
    print_if_verbose(verb,"event.MET.covXX:", events.MET.covXX)
    print_if_verbose(verb,"event.MET.covYY:", events.MET.covYY)

    covXX = ak.to_numpy(events.MET.covXX)
    covXY = ak.to_numpy(events.MET.covXY)
    covYY = ak.to_numpy(events.MET.covYY)

    # Create a 2x2 matrix for each covariance matrix
    covMET = []
    covMET = np.zeros((len(covXX), 2, 2))
    # covMET is a diagonale matrix 
    covMET[:, 0, 0] = covXX
    covMET[:, 1, 0] = covXY
    covMET[:, 0, 1] = covXY 
    covMET[:, 1, 1] = covYY

    # Convert covMet to a list of TMatrixD objects for the SVFit algorithm
    rootMETMatrices = []

    for cov in covMET:
        matrix = root.TMatrixD(2, 2)
        matrix[0][0] = cov[0][0]
        matrix[1][0] = cov[1][0]
        matrix[0][1] = cov[0][1]
        matrix[1][1] = cov[1][1]
        rootMETMatrices.append(matrix)
    
    # Get the hcand
    h_cand = events.hcand

    print_if_verbose(verb, f"h_cand fields : {h_cand.fields}")


    # Get the leptons pairs from the hcand
    leps1 = events.hcand[:,:1] # e muon or tau
    leps2 = events.hcand[:,1:2] # always tau

    print_if_verbose(verb, f"leps1 fields : {leps1.fields}")
    print_if_verbose(verb, f"leps2 fields: {leps2.fields}")

    # Convert the Awkward Array to numpy array for leps1 
    leps1_pt = ak.to_numpy(leps1["pt"])
    leps1_eta = ak.to_numpy(leps1["eta"])
    leps1_phi = ak.to_numpy(leps1["phi"])
    leps1_mass = ak.to_numpy(leps1["mass"])
    leps1_dm = ak.to_numpy(leps1["decayMode"])
    leps1_flavor = ak.to_numpy(leps1["lepton"])
    
    print_if_verbose(verb, "leps1 flavour = ", leps1_flavor ) # 13 = muon, 11 = electron, 15 = tau
    print_if_verbose(verb,"leps1 pt = ", leps1_pt )
    print_if_verbose(verb,"leps1 eta = ", leps1_eta )
    print_if_verbose(verb,"leps1 phi = ", leps1_phi )
    print_if_verbose(verb,"leps1 mass = ", leps1_mass )
    print_if_verbose(verb,"leps1 decayMode = ", leps1_dm)
    

    # Make the plot
    # fig, ax = plt.subplots()
    # plt.style.use(mplhep.style.CMS)
    # plt.hist(variable.flatten(), bins=20, range=(0, 2), alpha = 0.5)
    # plt.hist(ak.to_numpy(leps1["mass"]).flatten(), bins=20, range=(0, 2), alpha = 0.5)
    # plt.savefig("mass.pdf")    

    # Convert the Awkward Array to numpy array for leps1 
    leps2_pt = ak.to_numpy(leps2["pt"])
    leps2_eta = ak.to_numpy(leps2["eta"])
    leps2_phi = ak.to_numpy(leps2["phi"])
    leps2_mass = ak.to_numpy(leps2["mass"])
    leps2_dm = ak.to_numpy(leps2["decayMode"])
    leps2_flavor = ak.to_numpy(leps2["lepton"])

    print_if_verbose(verb,"leps2 pt = ", leps2_pt )
    print_if_verbose(verb,"leps2 eta = ", leps2_eta )
    print_if_verbose(verb,"leps2 phi = ", leps2_phi )
    print_if_verbose(verb,"leps2 mass = ", leps2_mass )
    print_if_verbose(verb,"leps2 decayMode = ", leps2_dm)

    # Define the MeasuredTauLepton objects for the SVFit algorithm
    leps1_MeasuredTauLepton = np.array([MeasuredTauLepton(*args) for args in zip(decayType, leps1_pt, leps1_eta, leps1_phi ,leps1_mass, leps1_dm)])
    leps2_MeasuredTauLepton = np.array([MeasuredTauLepton(1,*args) for args in zip(leps2_pt, leps2_eta, leps2_phi ,leps2_mass, leps2_dm)])

    print_if_verbose(verb,f"leps1_MeasuredTauLepton : {leps1_MeasuredTauLepton}")
    print_if_verbose(verb,f"leps2_MeasuredTauLepton : {leps2_MeasuredTauLepton}")

    # FastMTT Algorithm
    FastMTTAlgo = FastMTT()

    # massContraint = 91 # Z mass GeV
    # massContraint = 125.06 # Higgs mass GeV

    # svFitAlgo.setDiTauMassConstraint(massContraint)

    FastMTTAlgo.massLikelihood(massContraint)

    ditaumass_array = []


    # Loop over each element in the arrays
    start_time = time.time()

    for i in range(len(leps1_MeasuredTauLepton)):

        leps_MeasuredTauLepton = [leps1_MeasuredTauLepton[i], leps2_MeasuredTauLepton[i]]

        # Your code inside the for loop
        FastMTTAlgo.run(leps_MeasuredTauLepton, metx[i],  mety[i], rootMETMatrices[i])
   
        # Get the tau1 and tau2 P4 from FastMTT as LorentzVector
        tau1P4mtt = FastMTTAlgo.getTau1P4()
        tau2P4mtt = FastMTTAlgo.getTau2P4()

        print_if_verbose(verb,f"tau1P4mtt : {tau1P4mtt}")
        print_if_verbose(verb,f"tau2P4mtt : {tau2P4mtt}")

        # Get the ditau mass
        ditauMass = (tau1P4mtt + tau2P4mtt).M()

        print(f"ditauMass : {ditauMass}")
        ditaumass_array.append(ditauMass)

    end_time = time.time()

    execution_time_ = end_time - start_time

    print(f"Execution time FastMTT: {execution_time_} seconds")

    # Make Mass plot
    fig, ax = plt.subplots()
    plt.style.use(mplhep.style.CMS)
    plt.hist(ditaumass_array, bins=200, range=(0, 200), alpha = 0.5)
    plt.xlabel("Mass in GeV")
    plt.ylabel("Events")
    plt.savefig("mass_FastMTT.pdf") 

    # # Create a TH1D histogram for metx
    hist1 = root.TH1D("mass_FastMTT", "mass", 130, 0, 130)

    # Fill the histogram with metx values
    for mass in ditaumass_array:
        hist1.Fill(mass)

    # Draw the histogram
    hist1.Draw()

    # Save the histogram in a root file
    root_file = root.TFile("mass_FastMTT.root", "RECREATE")
    hist1.Write()
    root_file.Close()

    print(f"h_cand : {breakhere}") # breakhere

    return events