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
plothist = maybe_import("plothist")

print("plot_hist : ", plothist)
print("plt: ", plt)

# Matplotlib paramter for the plots

# params = {"ytick.color" : "black",
#           "xtick.color" : "black",
#           "axes.labelcolor" : "black",
#           "axes.edgecolor" : "black",
#           "text.usetex" : True,
#           "font.family" : "serif",
#           "font.serif" : ["Computer Modern Serif"]}
# plt.rcParams.update(params)


def create_TH1_histogram(array, name, title, bins, min_val, max_val):
    """
    Create a TH1D histogram from an array.

    Args:
    - array (list): The array containing the data.
    - name (str): The name of the histogram.
    - title (str): The title of the histogram.
    - bins (int): The number of bins.
    - min_val (float): The minimum value for the x-axis.
    - max_val (float): The maximum value for the x-axis.

    Returns:
    - hist (ROOT.TH1D): The TH1D histogram.
    """    
    hist = root.TH1D(name, title, bins, min_val, max_val)
    for value in array:
        # print(value)
        hist.Fill(value)

    # Save the histogram in a root file
    root_file = root.TFile("SVFitplots/"+name+".root", "RECREATE")
    hist.Write()
    root_file.Close()

    return hist

def create_histogram_matplotlib(array, name, title, bins, range_min, range_max):
    """
    Create a histogram from an array using Matplotlib.

    Args:
    - array (array-like): The array containing the data.
    - name (str): The name of the histogram.
    - title (str): The title of the histogram.
    - bins (int or array-like): The number of bins or bin edges.
    - range_min (float): The minimum value for the x-axis.
    - range_max (float): The maximum value for the x-axis.

    Returns:
    - hist (array): The histogram values.
    - bin_edges (array): The bin edges.
    """

    # Create histogram using Matplotlib
    hist, bin_edges, _ = plt.hist(array, bins=bins, range=(range_min, range_max), histtype='step', label=name, alpha = 0.5)

    # Set plot labels and title
    plt.xlabel(title)
    plt.ylabel('a.u.')
    plt.title(title)


    # Add smaller ticks on x-axis
    # plt.gca().xaxis.set_minor_locator(plt.MultipleLocator((range_max-range_min)/20.))

    # # Add smaller ticks on y-axis
    # plt.gca().yaxis.set_minor_locator(plt.MultipleLocator((range_max-range_min)/20.))

    # Save the histogram plot as an image
    plt.savefig("SVFitplots/"+name+".pdf")

    return hist, bin_edges

def create_histogram_plothist(array, name, title, bins, range_min, range_max, xaxis):
    """
    Create a histogram from an array using plothist.

    Args:
    - array (array-like): The array containing the data.
    - name (str): The name of the histogram.
    - title (str): The title of the histogram.
    - bins (int or array-like): The number of bins or bin edges.
    - range_min (float): The minimum value for the x-axis.
    - range_max (float): The maximum value for the x-axis.

    Returns:
    - hist (array): The histogram values.
    - bin_edges (array): The bin edges.
    """

    # Make the histogram
    h = plothist.make_hist(array, bins=bins, range=(range_min, range_max))

    # Plot the histogram
    fig, ax = plt.subplots()
    plothist.plot_hist(h, ax=ax)
    ax.set_xlabel(xaxis)
    ax.set_ylabel("Entries")
    ax.set_title(title)

    # Save the histogram plot as an image
    fig.savefig("SVFitplots/"+name+"plthist.pdf", bbox_inches="tight")

    return h


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

    # print(breakhere)

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
    leps1_genpt = ak.to_numpy(leps1["genpt"])
    leps1_geneta = ak.to_numpy(leps1["geneta"])
    leps1_genphi = ak.to_numpy(leps1["genphi"])
    leps1_genmass = ak.to_numpy(leps1["genmass"])

    
    print_if_verbose(verb, "leps1 flavour = ", leps1_flavor ) # 13 = muon, 11 = electron, 15 = tau
    print_if_verbose(verb,"leps1 pt = ", leps1_pt )
    print_if_verbose(verb,"leps1 eta = ", leps1_eta )
    print_if_verbose(verb,"leps1 phi = ", leps1_phi )
    print_if_verbose(verb,"leps1 mass = ", leps1_mass )
    print_if_verbose(verb,"leps1 decayMode = ", leps1_dm)
    print_if_verbose(verb,"leps1 genpt = ", leps1_genpt )
    print_if_verbose(verb,"leps1 geneta = ", leps1_geneta )
    print_if_verbose(verb,"leps1 genphi = ", leps1_genphi )
    print_if_verbose(verb,"leps1 genmass = ", leps1_genmass )

    # Convert the Awkward Array to numpy array for leps1 
    leps2_pt = ak.to_numpy(leps2["pt"])
    leps2_eta = ak.to_numpy(leps2["eta"])
    leps2_phi = ak.to_numpy(leps2["phi"])
    leps2_mass = ak.to_numpy(leps2["mass"])
    leps2_dm = ak.to_numpy(leps2["decayMode"])
    leps2_flavor = ak.to_numpy(leps2["lepton"])
    leps2_genpt = ak.to_numpy(leps2["genpt"])
    leps2_geneta = ak.to_numpy(leps2["geneta"])
    leps2_genphi = ak.to_numpy(leps2["genphi"])
    leps2_genmass = ak.to_numpy(leps2["genmass"])


    print_if_verbose(verb,"leps2 pt = ", leps2_pt )
    print_if_verbose(verb,"leps2 eta = ", leps2_eta )
    print_if_verbose(verb,"leps2 phi = ", leps2_phi )
    print_if_verbose(verb,"leps2 mass = ", leps2_mass )
    print_if_verbose(verb,"leps2 decayMode = ", leps2_dm)
    print_if_verbose(verb,"leps2 genpt = ", leps2_genpt )
    print_if_verbose(verb,"leps2 geneta = ", leps2_geneta )
    print_if_verbose(verb,"leps2 genphi = ", leps2_genphi )
    print_if_verbose(verb,"leps2 genmass = ", leps2_genmass )

    
    leps1_genp4 = ak.zip({"pt": leps1_genpt, "eta": leps1_geneta, "phi": leps1_genphi, "mass": leps1_genmass},
                      with_name="PtEtaPhiMLorentzVector",
                      behavior=coffea.nanoevents.methods.vector.behavior)

    leps2_genp4 = ak.zip({"pt": leps2_genpt, "eta": leps2_geneta, "phi": leps2_genphi, "mass": leps2_genmass},
                        with_name="PtEtaPhiMLorentzVector",
                        behavior=coffea.nanoevents.methods.vector.behavior)



    gen_ditaumass = ak.to_numpy((leps1_genp4 + leps2_genp4).mass).flatten()
    # gen_ditaupt = ak.to_numpy((leps1_genp4 + leps2_genp4).pt).flatten()
    # gen_ditaueta = ak.to_numpy((leps1_genp4 + leps2_genp4).eta).flatten()
    # gen_ditauphi = ak.to_numpy((leps1_genp4 + leps2_genp4).phi).flatten()
    gen_taus_pt = ak.to_numpy((leps1_genp4.pt, leps2_genp4.pt)).flatten()
    gen_taus_eta = ak.to_numpy((leps1_genp4.eta, leps2_genp4.eta)).flatten()
    gen_taus_phi = ak.to_numpy((leps1_genp4.phi, leps2_genp4.phi)).flatten()
    gen_tau1_pt = ak.to_numpy(leps1_genp4.pt).flatten()
    gen_tau1_eta = ak.to_numpy(leps1_genp4.eta).flatten()
    gen_tau1_phi = ak.to_numpy(leps1_genp4.phi).flatten()
    gen_tau2_pt = ak.to_numpy(leps2_genp4.pt).flatten()
    gen_tau2_eta = ak.to_numpy(leps2_genp4.eta).flatten()
    gen_tau2_phi = ak.to_numpy(leps2_genp4.phi).flatten()


    gen_mass_hist = create_TH1_histogram(gen_ditaumass, "gen_mass_dy", "Generated di-tau Mass Distribution", 200, 0, 200)
    gen_mass_hist_plthist = create_histogram_plothist(gen_ditaumass, "gen_mass_dy", "Generated di-tau Mass Distribution", 200, 0, 200, "Generated di-tau Mass")

    # breakhere
    # Define the MeasuredTauLepton objects for the SVFit algorithm
    leps1_MeasuredTauLepton = np.array([MeasuredTauLepton(*args) for args in zip(decayType, leps1_pt, leps1_eta, leps1_phi ,leps1_mass, leps1_dm)])
    leps2_MeasuredTauLepton = np.array([MeasuredTauLepton(1,*args) for args in zip(leps2_pt, leps2_eta, leps2_phi ,leps2_mass, leps2_dm)])

    print_if_verbose(verb,f"leps1_MeasuredTauLepton : {leps1_MeasuredTauLepton}")
    print_if_verbose(verb,f"leps2_MeasuredTauLepton : {leps2_MeasuredTauLepton}")

    svfit_verbosity = 0

    print_if_verbose(verb,"svfit_verbosity : ", svfit_verbosity)
    svFitAlgo = ClassicSVfit(svfit_verbosity)

    massContraint = 91 # Z mass GeV
    # # massContraint = 125.06 # Higgs mass GeV

    # #Set the mass constraint
    svFitAlgo.setDiTauMassConstraint(massContraint)

    ditau_mass_array = []

    # Tau 4-vectors
    tau1_pt_array = []
    tau1_eta_array = []
    tau1_phi_array = []
    tau2_pt_array = []
    tau2_eta_array = []
    tau2_phi_array = []
    taus_pt_array = [] # tau1 and tau2
    taus_eta_array = []
    taus_phi_array = []

    start_time = time.time()

    # Loop over each event 
    print(f"len(leps1_MeasuredTauLepton) : {len(leps1_MeasuredTauLepton)}")

    h = TauTauHistogramAdapter()
    # print(type(h))
    # h.getNQuantities()
    # print(h.getNQuantities())
    # for i in range (h.getNQuantities()):
    #     print(h.getQuantity(i))

    svFitAlgo.setHistogramAdapter(TauTauHistogramAdapter())

    # print(type(svFitAlgo.getHistogramAdapter()))

    for i in range(len(leps1_MeasuredTauLepton)):
    # for i in range(10):

        # Set the likelihood file name
        # svfit_filename = "testSVfit_hacand" + str(i) + ".root"
        svfit_filename = ""
        svFitAlgo.setLikelihoodFileName(svfit_filename)

        svFitAlgo.addLogM_fixed(True, kappa_np[i])
        # svFitAlgo.addLogM_dynamic(False)

        leps_MeasuredTauLepton = [leps1_MeasuredTauLepton[i], leps2_MeasuredTauLepton[i]]


        # svFitAlgo.prepareLeptonInput(leps_MeasuredTauLepton)
        # svFitAlgo.addMETEstimate(metx[i], mety[i], rootMETMatrices[i]) 
        # print_if_verbose(verb,"svFitAlgo.prepareLeptonInput= ", svFitAlgo.prepareLeptonInput(leps_MeasuredTauLepton))

        # Perform integration for each element
        print_if_verbose(verb,f"leps_MeasuredTauLepton[i] : {leps_MeasuredTauLepton}")
        svFitAlgo.integrate(leps_MeasuredTauLepton, metx[i], mety[i], rootMETMatrices[i])

        isValidSolution = svFitAlgo.isValidSolution()
        print_if_verbose(verb,f"isValidSolution : {isValidSolution}")

        # print(type(svFitAlgo.getHistogramAdapter()))

        # mass = svFitAlgo.getHistogramAdapter().getMass()


       # Get tau 4-vectors as LorentzVectors
        tau1P4 = svFitAlgo.getHistogramAdapter().GetFittedTau1LV()
        tau2P4 = svFitAlgo.getHistogramAdapter().GetFittedTau2LV()

        print_if_verbose(verb,f"tau1P4 : {tau1P4}")
        print_if_verbose(verb,f"tau1P4 : {tau1P4}")

        # Get the ditau mass
        tau1_pt_array.append(tau1P4.Pt())
        tau1_eta_array.append(tau1P4.Eta())
        tau1_phi_array.append(tau1P4.Phi())
        tau2_pt_array.append(tau2P4.Pt())
        tau2_eta_array.append(tau2P4.Eta())
        tau2_phi_array.append(tau2P4.Phi())

       

        # print( f"mass : {mass}")
        # print( f"errmass : {errmass}")
        # ditau_mass_array.append(mass)
        
    end_time = time.time()

    execution_time_ = end_time - start_time
    
    print(f"Execution time SVFit: {execution_time_} seconds")

     # print(f"ditauMass : {ditauMass}")
    taus_pt_array.append(tau1_pt_array)
    taus_pt_array.append(tau2_pt_array)
    taus_eta_array.append(tau1_eta_array)
    taus_eta_array.append(tau2_eta_array)
    taus_phi_array.append(tau1_phi_array)
    taus_phi_array.append(tau2_phi_array)

    print("taus_pt_array : ", taus_pt_array)
        
    # print_if_verbose(verb,f"mass_array : {ditau_mass_array}")
    # print_if_verbose(verb,f"err_mass_array : {err_mass_array}")

    # Make DI tau Mass plot
    # mass_hist = create_TH1_histogram(ditau_mass_array, "mass_SVFit_massconstrain_dy", "Mass Distribution", 200, 0, 200)

    # Make Tau 4-vectors plots
    # pt_hist = create_TH1_histogram(flatten(taus_pt_array), "pt_hist_SVFit_massconstrain_dy", "Pt Distribution", 100, 0, 200)
    # eta_hist = create_TH1_histogram(flatten(taus_eta_array), "eta_hist_SVFit_massconstrain_dy", "Eta Distribution", 60, -3, 3)
    # phi_hist = create_TH1_histogram(flatten(taus_phi_array), "phi_hist_SVFit_massconstrain_dy", "Phi Distribution", 60, -3.2, 3.2)
  
    genpt_hist = create_TH1_histogram(gen_taus_pt, "gen_pt_hist_SVFit_dy", "Pt Distribution", 100, 0, 200)
    geneta_hist = create_TH1_histogram(gen_taus_eta, "gen_eta_hist_SVFit_dy", "Eta Distribution", 60, -3, 3)
    genphi_hist = create_TH1_histogram(gen_taus_phi, "gen_phi_hist_SVFit_dy", "Phi Distribution", 60, -3.2, 3.2)
    genpt1_hist = create_TH1_histogram(gen_tau1_pt, "gen_pt1_hist_SVFit_dy", "Pt Distribution", 100, 0, 200)
    geneta1_hist = create_TH1_histogram(gen_tau1_eta, "gen_eta1_hist_SVFit_dy", "Eta Distribution", 60, -3, 3)
    genphi1_hist = create_TH1_histogram(gen_tau1_phi, "gen_phi1_hist_SVFit_dy", "Phi Distribution", 60, -3.2, 3.2)
    genpt2_hist = create_TH1_histogram(gen_tau2_pt, "gen_pt2_hist_SVFit_dy", "Pt Distribution", 100, 0, 200)
    geneta2_hist = create_TH1_histogram(gen_tau2_eta, "gen_eta2_hist_SVFit_dy", "Eta Distribution", 60, -3, 3)
    genphi2_hist = create_TH1_histogram(gen_tau2_phi, "gen_phi2_hist_SVFit_dy", "Phi Distribution", 60, -3.2, 3.2)

    


    # Resolution : 
    # mass_resolution = (np.array(gen_ditaumass) - np.array(mass_array)) / np.array(gen_ditaumass)    
    # pt_resolution = (np.array(gen_taus_pt) - np.array(taus_pt_array)) / np.array(gen_taus_pt)
    # eta_resolution = (np.array(gen_taus_eta)- np.array(taus_eta_array)) 
    # phi_resolution = (np.array(gen_taus_phi) - np.array(taus_phi_array)) 
    pt1_resolution = (np.array(gen_tau1_pt) - np.array(tau1_pt_array)) / np.array(gen_tau1_pt)
    eta1_resolution = (np.array(gen_tau1_eta)- np.array(tau1_eta_array))
    phi1_resolution = (np.array(gen_tau1_phi) - np.array(tau1_phi_array))
    pt2_resolution = (np.array(gen_tau2_pt) - np.array(tau2_pt_array)) / np.array(gen_tau2_pt)
    eta2_resolution = (np.array(gen_tau2_eta)- np.array(tau2_eta_array))
    phi2_resolution = (np.array(gen_tau2_phi) - np.array(tau2_phi_array))

    # print("eta_resolution = " , eta_resolution)
    # print("gen_taus_eta = ", gen_taus_eta)
    # print("taus_eta_array = ", taus_eta_array)

    # mass_resolution_hist = create_TH1_histogram(mass_resolution, "mass_resolution_SVFit_massconstrain_dy", "Mass resolution", 100, -1, 1)
    # pt_resolution_hist = create_TH1_histogram(pt_resolution, "pt_resolution_SVFit_massconstrain_dy", "Pt resolution", 50, -1, 1)
    # eta_resolution_hist = create_TH1_histogram(eta_resolution, "eta_resolution_SVFit_massconstrain_dy", "Eta resolution",  50, -1, 1)
    # phi_resolution_hist = create_TH1_histogram(phi_resolution, "phi_resolution_SVFit_massconstrain_dy", "Phi resolution", 50, -1, 1)
    pt1_resolution_hist = create_TH1_histogram(pt1_resolution, "pt1_resolution_SVFit_massconstrain_dy", "Pt resolution", 100, -1, 1)
    eta1_resolution_hist = create_TH1_histogram(eta1_resolution, "eta1_resolution_SVFit_massconstrain_dy", "Eta resolution",  100, -1, 1)
    phi1_resolution_hist = create_TH1_histogram(phi1_resolution, "phi1_resolution_SVFit_massconstrain_dy", "Phi resolution", 100, -1, 1)
    pt2_resolution_hist = create_TH1_histogram(pt2_resolution, "pt2_resolution_SVFit_massconstrain_dy", "Pt resolution", 100, -1, 1)
    eta2_resolution_hist = create_TH1_histogram(eta2_resolution, "eta2_resolution_SVFit_massconstrain_dy", "Eta resolution",  100, -1, 1)
    phi2_resolution_hist = create_TH1_histogram(phi2_resolution, "phi2_resolution_SVFit_massconstrain_dy", "Phi resolution", 100, -1, 1)

    
    # mass_resolution_hist_plt = create_histogram_matplotlib(mass_resolution, "mass_resolution_SVFit_massconstrain_dy", "Mass resolution", 100, -1, 1)
    # pt_resolution_hist_plt = create_histogram_matplotlib(pt_resolution, "pt_resolution_SVFit_massconstrain_dy", r"$p_{T}$ resolution", 100, -1, 1)
    # eta_resolution_hist_plt = create_histogram_matplotlib(eta_resolution, "eta_resolution_SVFit_massconstrain_dy", r"$|\eta| resolution", 100, -1, 1)
    # phi_resolution_hist_plt = create_histogram_matplotlib(phi_resolution, "phi_resolution_SVFit_massconstrain_dy", r"$\phi$ resolution", 100, -1, 1)
   

    # # mass_resolution_hist_plthist = create_histogram_plothist(mass_resolution, "mass_resolution_SVFit_massconstrain_dy", "Mass Resolution Distribution", 100, -1, 1, r"$\frac{m_{H,gen}-m_{H,reco}}{H,m_{gen}}$")
    # pt_resolution_hist_plthist = create_histogram_plothist(pt_resolution, "pt_resolution_SVFit_massconstrain_dy", r"$p_{T,\tau}$ resolution Distribution", 50, -1, 1, r"$\frac{p_{T,\tau,gen}-p_{T,\tau,reco}}{p_{T,\tau,gen}}$")
    # eta_resolution_hist_plthist = create_histogram_plothist(eta_resolution, "eta_resolution_SVFit_massconstrain_dy", r"$|\eta|_{\tau}$ resolution Distribution", 50, -0.2, 0.2, r"$|\eta|_{\tau,gen}-|\eta|_{\tau,reco}$")
    # phi_resolution_hist_plthist = create_histogram_plothist(phi_resolution, "phi_resolution_SVFit_massconstrain_dy", r"$\phi_{\tau,gen}$ resolution Distribution", 50, -0.2, 0.2, r"$\phi_{\tau,gen}-\phi_{\tau,reco}$")
    pt1_resolution_hist_plthist = create_histogram_plothist(pt1_resolution, "pt1_resolution_SVFit_massconstrain_dy", r"$p_{T,\tau1}$ resolution Distribution", 100, -1, 1, r"$\frac{p_{T,\tau1,gen}-p_{T,\tau1,reco}}{p_{T,\tau1,gen}}$")
    eta1_resolution_hist_plthist = create_histogram_plothist(eta1_resolution, "eta1_resolution_SVFit_massconstrain_dy", r"$|\eta|_{\tau1}$ resolution Distribution", 100, -0.2, 0.2, r"$|\eta|_{\tau1,gen}-|\eta|_{\tau1,reco}$")
    phi1_resolution_hist_plthist = create_histogram_plothist(phi1_resolution, "phi1_resolution_SVFit_massconstrain_dy", r"$\phi_{\tau1}$ resolution Distribution", 100, -0.2, 0.2, r"$\phi_{\tau1,gen}-\phi_{\tau1,reco}$")
    pt2_resolution_hist_plthist = create_histogram_plothist(pt2_resolution, "pt2_resolution_SVFit_massconstrain_dy", r"$p_{T,\tau2}$ resolution Distribution", 100, -1, 1, r"$\frac{p_{T,\tau2,gen}-p_{T,\tau2,reco}}{p_{T,\tau2,gen}}$")
    eta2_resolution_hist_plthist = create_histogram_plothist(eta2_resolution, "eta2_resolution_SVFit_massconstrain_dy", r"$|\eta|_{\tau2}$ resolution Distribution", 100, -0.2, 0.2, r"$|\eta|_{\tau2,gen}-|\eta|_{\tau2,reco}$")
    phi2_resolution_hist_plthist = create_histogram_plothist(phi2_resolution, "phi2_resolution_SVFit_massconstrain_dy", r"$\phi_{\tau2}$ resolution Distribution", 100, -0.2, 0.2, r"$\phi_{\tau2,gen}-\phi_{\tau2,reco}$")


    # mass_resolution_hist_plthist = create_histogram_plothist(mass_resolution, "mass_resolution_SVFit_massconstrain_dy", "Mass resolution", 100, -1, 1)

    print(f"h_cand : {breakhere}")
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
    leps1_genpt = ak.to_numpy(leps1["genpt"])
    leps1_geneta = ak.to_numpy(leps1["geneta"])
    leps1_genphi = ak.to_numpy(leps1["genphi"])
    leps1_genmass = ak.to_numpy(leps1["genmass"])

    
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
    leps2_genpt = ak.to_numpy(leps2["genpt"])
    leps2_geneta = ak.to_numpy(leps2["geneta"])
    leps2_genphi = ak.to_numpy(leps2["genphi"])
    leps2_genmass = ak.to_numpy(leps2["genmass"])


    print_if_verbose(verb,"leps2 pt = ", leps2_pt )
    print_if_verbose(verb,"leps2 eta = ", leps2_eta )
    print_if_verbose(verb,"leps2 phi = ", leps2_phi )
    print_if_verbose(verb,"leps2 mass = ", leps2_mass )
    print_if_verbose(verb,"leps2 decayMode = ", leps2_dm)
    print_if_verbose(verb,"leps2 genpt = ", leps2_genpt )
    print_if_verbose(verb,"leps2 geneta = ", leps2_geneta )
    print_if_verbose(verb,"leps2 genphi = ", leps2_genphi )
    print_if_verbose(verb,"leps2 genmass = ", leps2_genmass )

    
    leps1_genp4 = ak.zip({"pt": leps1_genpt, "eta": leps1_geneta, "phi": leps1_genphi, "mass": leps1_genmass},
                      with_name="PtEtaPhiMLorentzVector",
                      behavior=coffea.nanoevents.methods.vector.behavior)

    leps2_genp4 = ak.zip({"pt": leps2_genpt, "eta": leps2_geneta, "phi": leps2_genphi, "mass": leps2_genmass},
                        with_name="PtEtaPhiMLorentzVector",
                        behavior=coffea.nanoevents.methods.vector.behavior)



    gen_ditaumass = ak.to_numpy((leps1_genp4 + leps2_genp4).mass).flatten()
    # gen_ditaupt = ak.to_numpy((leps1_genp4 + leps2_genp4).pt).flatten()
    # gen_ditaueta = ak.to_numpy((leps1_genp4 + leps2_genp4).eta).flatten()
    # gen_ditauphi = ak.to_numpy((leps1_genp4 + leps2_genp4).phi).flatten()
    gen_taus_pt = ak.to_numpy((leps1_genp4.pt, leps2_genp4.pt)).flatten()
    gen_taus_eta = ak.to_numpy((leps1_genp4.eta, leps2_genp4.eta)).flatten()
    gen_taus_phi = ak.to_numpy((leps1_genp4.phi, leps2_genp4.phi)).flatten()
    gen_tau1_pt = ak.to_numpy(leps1_genp4.pt).flatten()
    gen_tau1_eta = ak.to_numpy(leps1_genp4.eta).flatten()
    gen_tau1_phi = ak.to_numpy(leps1_genp4.phi).flatten()
    gen_tau2_pt = ak.to_numpy(leps2_genp4.pt).flatten()
    gen_tau2_eta = ak.to_numpy(leps2_genp4.eta).flatten()
    gen_tau2_phi = ak.to_numpy(leps2_genp4.phi).flatten()

    # Root plot
    gen_mass_hist = create_TH1_histogram(gen_ditaumass, "gen_mass_dy", "Gen Mass Distribution", 200, 0, 200)
    # gen_pt_hist = create_TH1_histogram(gen_ditaupt, "gen_pt_dy", "Gen Pt Distribution", 200, 0, 200)
    # gen_eta_hist = create_TH1_histogram(gen_ditaueta, "gen_eta_dy", "Gen Eta Distribution", 100, -3, 3)
    # gen_phi_hist = create_TH1_histogram(gen_ditauphi, "gen_phi_dy", "Gen Phi Distribution", 100, -3.2, 3.2)

    # Plot with plothist
    gen_mass_hist_plthist = create_histogram_plothist(gen_ditaumass, "gen_mass_dy", "Gen Mass Distribution", 100, 0, 200, "Gen Mass")
    gen_pt_hist_plthist = create_histogram_plothist(gen_taus_pt, "gen_pt_dy", r"Gen $p_{T}$ Distribution", 100, 0, 200, r"Gen $p_{T}$")
    gen_eta_hist_plthist = create_histogram_plothist(gen_taus_eta, "gen_eta_dy", r"Gen $|\eta|$ Distribution", 60, -3, 3, r"Gen $|\eta|$")
    gen_phi_hist_plthist = create_histogram_plothist(gen_taus_phi, "gen_phi_dy", r"Gen $\phi$ Distribution", 60, -3.2, 3.2, r"Gen $\phi$")



    # Define the MeasuredTauLepton objects for the SVFit algorithm
    leps1_MeasuredTauLepton = np.array([MeasuredTauLepton(*args) for args in zip(decayType, leps1_pt, leps1_eta, leps1_phi ,leps1_mass, leps1_dm)])
    leps2_MeasuredTauLepton = np.array([MeasuredTauLepton(1,*args) for args in zip(leps2_pt, leps2_eta, leps2_phi ,leps2_mass, leps2_dm)])

    print_if_verbose(verb,f"leps1_MeasuredTauLepton : {leps1_MeasuredTauLepton}")
    print_if_verbose(verb,f"leps2_MeasuredTauLepton : {leps2_MeasuredTauLepton}")

    # FastMTT Algorithm
    FastMTTAlgo = FastMTT()


    ditaumass_array = []
    # ditaueta_array = []
    # ditauphi_array = []
    # ditauPt_array = []
    
    # Tau 4-vectors
    tau1_pt_array = []
    tau1_eta_array = []
    tau1_phi_array = []
    tau2_pt_array = []
    tau2_eta_array = []
    tau2_phi_array = []
    taus_pt_array = [] # tau1 and tau2
    taus_eta_array = []
    taus_phi_array = []



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

        # print(f"ditauMass : {ditauMass}")
        ditaumass_array.append(ditauMass)
        # Get the tau 4-vectors
        tau1_pt_array.append(tau1P4mtt.Pt())
        tau1_eta_array.append(tau1P4mtt.Eta())
        tau1_phi_array.append(tau1P4mtt.Phi())
        tau2_pt_array.append(tau2P4mtt.Pt())
        tau2_eta_array.append(tau2P4mtt.Eta())
        tau2_phi_array.append(tau2P4mtt.Phi())
        taus_pt_array.append(tau1P4mtt.Pt())
        taus_eta_array.append(tau1P4mtt.Eta())
        taus_phi_array.append(tau1P4mtt.Phi())
        taus_pt_array.append(tau2P4mtt.Pt())
        taus_eta_array.append(tau2P4mtt.Eta())
        taus_phi_array.append(tau2P4mtt.Phi())


    end_time = time.time()

    execution_time_ = end_time - start_time

    print(f"Execution time FastMTT: {execution_time_} seconds")

    # # Make Mass plot
    # fig, ax = plt.subplots()
    # plt.style.use(mplhep.style.CMS)
    # plt.hist(ditaumass_array, bins=200, range=(0, 200), alpha = 0.5)
    # plt.xlabel("Mass in GeV")
    # plt.ylabel("Events")
    # plt.savefig("mass_FastMTT.pdf") 

    

    mass_hist = create_TH1_histogram(ditaumass_array, "mass_FastMtt_dy", "Mass Distribution", 200, 0, 200)
    # taus_hist = create_TH1_histogram(taus_pt_array, "pt_hist_FastMtt_dy", "Pt Distribution", 100, 0, 200)
    # eta_hist = create_TH1_histogram(taus_eta_array, "eta_hist_FastMtt_dy", "Eta Distribution", 60, -3, 3)
    # phi_hist = create_TH1_histogram(taus_phi_array, "phi_hist_FastMtt_dy", "Phi Distribution", 60, -3.2, 3.2)

    # Resolution Higs Mass : 
    mass_resolution = (np.array(gen_ditaumass) - np.array(ditaumass_array)) / np.array(gen_ditaumass) 

    # Resolution Tau 4-vectors
    # pt_resolution = (np.array(gen_taus_pt) - np.array(taus_pt_array)) / np.array(gen_taus_pt)
    # eta_resolution = (np.array(gen_taus_eta)- np.array(taus_eta_array)) 
    # phi_resolution = (np.array(gen_taus_phi) - np.array(taus_phi_array)) 
    pt1_resolution = (np.array(gen_tau1_pt) - np.array(tau1_pt_array)) / np.array(gen_tau1_pt)
    eta1_resolution = (np.array(gen_tau1_eta)- np.array(tau1_eta_array))
    phi1_resolution = (np.array(gen_tau1_phi) - np.array(tau1_phi_array))
    pt2_resolution = (np.array(gen_tau2_pt) - np.array(tau2_pt_array)) / np.array(gen_tau2_pt)
    eta2_resolution = (np.array(gen_tau2_eta)- np.array(tau2_eta_array))
    phi2_resolution = (np.array(gen_tau2_phi) - np.array(tau2_phi_array))


    mass_resolution_hist = create_TH1_histogram(mass_resolution, "mass_resolution_FastMtt_dy", "Mass resolution", 100, -1, 1)
    # pt_resolution_hist = create_TH1_histogram(pt_resolution, "pt_resolution_FastMtt_dy", "Pt resolution", 100, -1, 1)
    # eta_resolution_hist = create_TH1_histogram(eta_resolution, "eta_resolution_FastMtt_dy", "Eta resolution",  100, -1, 1)
    # phi_resolution_hist = create_TH1_histogram(phi_resolution, "phi_resolution_FastMtt_dy", "Phi resolution", 100, -1, 1)
    pt1_resolution_hist = create_TH1_histogram(pt1_resolution, "pt1_resolution_FastMtt_dy", "Pt resolution", 100, -1, 1)
    eta1_resolution_hist = create_TH1_histogram(eta1_resolution, "eta1_resolution_FastMtt_dy", "Eta resolution",  100, -1, 1)
    phi1_resolution_hist = create_TH1_histogram(phi1_resolution, "phi1_resolution_FastMtt_dy", "Phi resolution", 100, -1, 1)
    pt2_resolution_hist = create_TH1_histogram(pt2_resolution, "pt2_resolution_FastMtt_dy", "Pt resolution", 100, -1, 1)
    eta2_resolution_hist = create_TH1_histogram(eta2_resolution, "eta2_resolution_FastMtt_dy", "Eta resolution",  100, -1, 1)
    phi2_resolution_hist = create_TH1_histogram(phi2_resolution, "phi2_resolution_FastMtt_dy", "Phi resolution", 100, -1, 1)


    # mass_resolution_hist_plt = create_histogram_matplotlib(mass_resolution, "mass_resolution_SVFit_massconstrain_dy", "Mass resolution", 100, -1, 1)
    # pt_resolution_hist_plt = create_histogram_matplotlib(pt_resolution, "pt_resolution_SVFit_massconstrain_dy", r"$p_{T}$ resolution", 100, -1, 1)
    # eta_resolution_hist_plt = create_histogram_matplotlib(eta_resolution, "eta_resolution_SVFit_massconstrain_dy", r"$|\eta| resolution", 100, -1, 1)
    # phi_resolution_hist_plt = create_histogram_matplotlib(phi_resolution, "phi_resolution_SVFit_massconstrain_dy", r"$\phi$ resolution", 100, -1, 1)
   

    mass_resolution_hist_plthist = create_histogram_plothist(mass_resolution, "mass_resolution_FastMtt_dy", "Mass Resolution Distribution", 100, -1, 1, r"$\frac{m_{H,gen}-m_{H,reco}}{m_{H,gen}}$")
    # pt_resolution_hist_plthist = create_histogram_plothist(pt_resolution, "pt_resolution_FastMtt_dy", r"$p_{T}$ resolution Distribution", 100, -1, 1, r"$\frac{p_{T,\tau,gen}-p_{T,\tau,reco}}{p_{T,\tau,gen}}$")
    # eta_resolution_hist_plthist = create_histogram_plothist(eta_resolution, "eta_resolution_FastMtt_dy", r"$|\eta|$ resolution Distribution", 100, -0.2, 0.2, r"$|\eta|_{\tau,gen}-|\eta|_{\tau,reco}$")
    # phi_resolution_hist_plthist = create_histogram_plothist(phi_resolution, "phi_resolution_FastMtt_dy", r"$\phi$ resolution Distribution", 100, -0.2, 0.2, r"$\phi_{\tau,gen}-\phi_{\tau,reco}$")
    pt1_resolution_hist_plthist = create_histogram_plothist(pt1_resolution, "pt1_resolution_FastMtt_dy", r"$p_{T}$ resolution Distribution", 100, -1, 1, r"$\frac{p_{T,\tau1,gen}-p_{T,\tau1,reco}}{p_{T,\tau1,gen}}$")
    eta1_resolution_hist_plthist = create_histogram_plothist(eta1_resolution, "eta1_resolution_FastMtt_dy", r"$|\eta|$ resolution Distribution", 100, -0.2, 0.2, r"$|\eta|_{\tau1,gen}-|\eta|_{\tau1,reco}$")
    phi1_resolution_hist_plthist = create_histogram_plothist(phi1_resolution, "phi1_resolution_FastMtt_dy", r"$\phi$ resolution Distribution", 100, -0.2, 0.2, r"$\phi_{\tau1,gen}-\phi_{\tau1,reco}$")
    pt2_resolution_hist_plthist = create_histogram_plothist(pt2_resolution, "pt2_resolution_FastMtt_dy", r"$p_{T}$ resolution Distribution", 100, -1, 1, r"$\frac{p_{T,\tau2,gen}-p_{T,\tau2,reco}}{p_{T,\tau2,gen}}$")
    eta2_resolution_hist_plthist = create_histogram_plothist(eta2_resolution, "eta2_resolution_FastMtt_dy", r"$|\eta|$ resolution Distribution", 100, -0.2, 0.2, r"$|\eta|_{\tau2,gen}-|\eta|_{\tau2,reco}$")
    phi2_resolution_hist_plthist = create_histogram_plothist(phi2_resolution, "phi2_resolution_FastMtt_dy", r"$\phi$ resolution Distribution", 100, -0.2, 0.2, r"$\phi_{\tau2,gen}-\phi_{\tau2,reco}$")



    print(f"h_cand : {breakhere}") # breakhere

    return events


