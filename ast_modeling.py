import time
import numpy as np
from utilss import *
from numba import njit
from pathlib import Path
import matplotlib.pyplot as plt

class Astrocyte: 
    def __init__(self, dt, N, Ra, Cm1, astrocyte_data, stim_start_nt, stim_end_nt, glut_stim, pot_stim, stim_comp_glut, stim_comp_pot, stim_dur=0.0, n_stim=0, n_stimulated=0):
        self.dt = dt
        self.N = N
        self.Ra = Ra # intracellular resistivity
        self.Cm1 = Cm1
        self.stim_start_nt = stim_start_nt
        self.stim_end_nt = stim_end_nt
        self.stim_dur = stim_dur
        self.stim_comp_glut = stim_comp_glut
        self.stim_comp_pot = stim_comp_pot
        self.glut_stim = glut_stim
        self.pot_stim = pot_stim
        self.n_stim = n_stim
        self.n_stimulated = n_stimulated

        # Load astrocyte data
        self.astrocyte = np.loadtxt(astrocyte_data)        
        self.parents = self.astrocyte[:, 6].astype(int)
        self.parents[0] = 0  # Root of tree
        self.xaxis, self.yaxis, self.zaxis, self.radius, self.identifier = self.astrocyte[:, 2], self.astrocyte[:, 3], self.astrocyte[:, 4], self.astrocyte[:, 5], self.astrocyte[:, 1]
        
        # Initialize variables
        self.init_variables()
    
    
    def init_variables(self):        
        # Initialize geometry-related variables
        self.distance_from_soma = np.zeros(self.N)
        self.length = np.zeros(self.N)
        self.area = np.zeros(self.N)
        self.volume = np.zeros(self.N)
        self.rho = np.zeros(self.N)
        self.Rm = np.zeros(self.N)
        self.Cm = np.zeros(self.N)
        self.g_l = np.zeros(self.N)
 
        self.DiffK = np.zeros(self.N)
        self.DiffNa = np.zeros(self.N)
        self.DiffCa = np.zeros(self.N)
        self.gamma = np.zeros(self.N)
        self.rho2 = np.zeros(self.N)
        
        # Calculate geometric properties and initialize electrical parameters
        for i in range(self.N):
            self.calculate_geo_res(i)
        for i in range(self.N):
            self.calculate_gamma(i)

    def calculate_geo_res(self, i): 

        self.distance_from_soma[i] = np.sqrt((self.xaxis[i] - self.xaxis[0])**2 + (self.yaxis[i] - self.yaxis[0])**2 + (self.zaxis[i] - self.zaxis[0])**2)
        j = self.parents[i]
        if i == 0:
            self.length[i] = self.radius[i]
            self.area[i] = 4.0 * np.pi * self.radius[i]**2
            self.volume[i] = 4.0/3.0 * np.pi * self.radius[i]**3
        else:
            self.length[i] = np.sqrt((self.xaxis[i] - self.xaxis[j-1])**2 + (self.yaxis[i] - self.yaxis[j-1])**2 + (self.zaxis[i] - self.zaxis[j-1])**2)
            self.area[i] = 2.0 * np.pi * self.radius[i] * self.length[i] 
            self.volume[i] = np.pi * self.radius[i]**2 * self.length[i]
        
        self.rho[i] = self.Ra * 1.0e4 * self.length[i] / (2.0 * np.pi * self.radius[i]**2)   #ohms
        if self.identifier[i] not in [1, 2]:
            self.Rm[i] = 50000.0
            self.Cm[i] = self.Cm1 * 2.0
        else:
            self.Rm[i] = 50000.0
            self.Cm[i] = self.Cm1
        self.g_l[i] = 1000.0 / self.Rm[i]

    def calculate_gamma(self, i):
        for i in range(self.N):
            j = self.parents[i]
            if i == 0:
                if self.N == 1:
                    self.rho2[i] = 0
                    self.gamma[i] = 0
                else:
                    self.rho2[i] = 1.0 / (self.rho[i] + self.rho[i]) # 1/ohms i+1 last to run multicompartment, currently set to i to avoid error in single comp simulation
                    self.gamma[i] = 1.0e11 * self.rho2[i] / (2.0 * np.pi * self.radius[i] * self.length[i])* 1.0e-5  #mS/cm^2 i+1 both to run multicompartment, currently set to i to avoid error in single comp simulation
                
            else:
                self.rho2[i] = 1.0 / (self.rho[i] + self.rho[j-1])
                self.gamma[i] = 1.0e11 * self.rho2[i] / (2.0 * np.pi * self.radius[i] * self.length[i]) * 1.0e-5  #mS/cm^2
                
            self.DiffK[i] = 250.0 / (self.length[i]**2) # /s. 250 is the diffusion coefficent in um^2/s
            self.DiffNa[i] = 600.0 / (self.length[i]**2) # /s
            self.DiffCa[i] = 5.0/ (self.length[i]**2) # /s


    def astrocyte_dynamics_diffusion(self, iskip, last):
        v,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2 = update_astrocyte_dynamics(self.dt, self.identifier, self.gamma, self.parents, self.DiffNa, self.DiffK, self.DiffCa, self.g_l, self.distance_from_soma, self.radius, self.area, self.volume, self.Cm, self.stim_start_nt, self.stim_end_nt, self.stim_dur, self.glut_stim, self.pot_stim, self.stim_comp_glut, self.stim_comp_pot, self.n_stim, self.n_stimulated, self.N, last, iskip)
        return v,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2

  
if __name__ == "__main__":
    start_time = time.time()
    select_cell = "Astrocyte"

    if select_cell == "Astrocyte":        
        astrocyte_data_file = Path("NMO_73320.txt")   # edit path. SHould be good as long as the file is in the same dir as this script
        if not astrocyte_data_file.is_file():
            raise FileNotFoundError(
                f"[ERROR] File not found: {astrocyte_data_file.resolve()}\n"
                "Please make sure a valid astrocytic morphology .txt file path is provided."
    )
        else:
            df = np.loadtxt(astrocyte_data_file) 
            xaxis, yaxis, zaxis = df[:, 2], df[:, 3], df[:, 4]    
            N = 1 # number of compartment is 1 since we are modeling a single cell with whole cell attributes
            N_1 = 0
            N_2 = 1
            stim_start_nt = 220000000   #220000000 #22nd second 1000000000
            stim_end_nt = 230000000  #230000000 #23rd second     
            stim_type = "Potassium"

            if stim_type == "Potassium":
                pot_stim_freq = "Single"
                if pot_stim_freq == "Single":
                    pot_stim_comps = {}
                    pot_stims = {}
                    pot_stim_comps["stim_comp_0"] = 0
                    pot_stims["pot_stim_0"] = 7000
                 
                pot_stim_comps_array = np.array(list(pot_stim_comps.values()), dtype=np.int64)
                pot_stims_array = np.array(list(pot_stims.values()), dtype=np.float64)

                astrocyte = Astrocyte(dt=0.00000010, N=N, Ra=400.0, Cm1=1.0, astrocyte_data=astrocyte_data_file, stim_start_nt=stim_start_nt, stim_end_nt=stim_end_nt, glut_stim=None, pot_stim=pot_stims_array, stim_comp_glut=None, stim_comp_pot=pot_stim_comps_array) 
                
                iskip = 1000 #for testing/identifying only the base and the peak if needed, use 1000; else 16000
                last = 300000000  # for testing/identifying only the base and the peak if needed, use 300000000; else 10000000000 for 1000s

                vs,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2 = astrocyte.astrocyte_dynamics_diffusion(iskip, last) 
                
                comb_df = concat_ast_param(vs, Naks, Kos, Kks, Caks, phos, phis, IdiffKs, JNaks, JNaKks, Iks, time_array, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, N_1, N_2)
                
                output_path = Path(__file__).with_name("comb_astro_data.csv")                
                comb_df.to_csv(output_path, index=False)  
                                          
                end_time = time.time()
                elapsed_time = end_time - start_time 
                print(f"Simulation Completed! Total time elapsed: {elapsed_time} seconds")
                 