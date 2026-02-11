import numpy as np
import time
from utils import *
from numba import njit
from numba.typed import Dict
from numba.core import types
from pathlib import Path

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
        self.ids = self.astrocyte[:, 0].astype(int)
        
        # Initialize variables
        self.init_variables()
    
    
    def init_variables(self):
        #self.tau1, self.tau2, self.tau3, self.tau4, self.tau5 = 30.0, 35.0, 40.0, 45.0, 50.0  
        # Initialize geometry-related variables
        self.distance_from_soma = np.zeros(self.N)
        self.length = np.zeros(self.N)
        self.area = np.zeros(self.N)
        self.area_cross = np.zeros(self.N)
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

        self.distance_from_soma[:] = 0.0
        for i in range(self.N):
            j = self.parents[i]        # parent ID
            if j > 0:                             # parent row index
                self.distance_from_soma[i] = self.distance_from_soma[j-1] + self.length[i]
            else:
                # roots / soma samples start at 0
                self.distance_from_soma[i] = 0.0

        for i in range(self.N):
                self.calculate_gamma(i)

        self.branch_to_comps , self.branch_max_dict =  self.compute_branch_groups()
        self.nb_branch_to_comps, self.nb_branch_max = self.convert_to_numba_dict()

    def compute_branch_groups(self):
    # """
    # Returns:
    #   - branch_to_indices: {"branch_1": [row_idx0, row_idx1, ...], ...}
    #   - branch_to_ids    : {"branch_1": [id0, id1, ...], ...}          (SWC IDs)
    #   - branch_heads_id  : {"branch_1": head_id, ...}                   (SWC ID of head)
    #   - branch_max_dict  : {"branch_1": max_distance, ...}
    #   - branch_label     : np.ndarray (N,) with the head row-index label for each node (-1 for soma/root)
    # Notes:
    #   * A "primary branch" is the subtree whose root is any node whose parent <= 0.
    #   * Uses parent-row = parent_id - 1 (j-1).
    # """  

    # 1) Label each node by its primary branch head (child of soma)
        labels = np.full(self.N, -1, dtype=int)
        for i in range(self.N):
            if self.parents[i] <= 0:
                continue  # soma/root nodes are not inside any branch
            k = i
            while self.parents[k] > 0 and self.parents[self.parents[k] - 1] > 0:
                k = self.parents[k] - 1
            labels[i] = k  # k is the row index of the branch head

        # 2) Keep heads in order of first appearance (stable branch_1, branch_2, ...)
        heads, seen = [], set()
        for k in labels:
            if k >= 0 and k not in seen:
                heads.append(k); seen.add(k)

        # 3) Build groups and compute max per branch (include the head itself)
        branch_to_indices = {}
        branch_to_comps     = {}
        branch_heads_id   = {}
        branch_max_dict   = {}

        for idx, h in enumerate(heads, start=1):
            name = f"branch_{idx}"
            mask = (labels == h) | (np.arange(self.N) == h)     # include head
            idxs = np.nonzero(mask)[0]                          # all row indices in this branch
            branch_to_indices[name] = idxs.tolist()
            branch_to_comps [name]     = [int(self.ids[j])-1 for j in idxs]
            branch_heads_id[name]   = int(self.ids[h])
            branch_max_dict[name]   = float(np.max(self.distance_from_soma[idxs]))

        return branch_to_comps, branch_max_dict

    # Convert to Numba typed dictionary
    def convert_to_numba_dict(self): 
        # Create typed dicts
        nb_branch_to_comps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
        nb_branch_max = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        
        # Map branch names to integers and fill dicts
        for idx, (bname, comps) in enumerate(self.branch_to_comps.items()): 
            nb_branch_to_comps[idx] = np.array(comps, dtype=np.int64)
            nb_branch_max[idx] = self.branch_max_dict[bname] 
        
        return nb_branch_to_comps, nb_branch_max

    def calculate_geo_res(self, i): 

        # self.distance_from_soma[i] = np.sqrt((self.xaxis[i] - self.xaxis[0])**2 + (self.yaxis[i] - self.yaxis[0])**2 + (self.zaxis[i] - self.zaxis[0])**2)
        # self.distance_from_soma, _, _ = self.cumulative_distance_from_soma(self.ids, self.parents, self.x, self.y, self.z)
        j = self.parents[i]
        if i == 0:
            self.length[i] = self.radius[i]
            self.area[i] = 4.0 * np.pi * self.radius[i]**2            
            self.volume[i] = 4.0/3.0 * np.pi * self.radius[i]**3
        else:
            self.length[i] = np.sqrt((self.xaxis[i] - self.xaxis[j-1])**2 + (self.yaxis[i] - self.yaxis[j-1])**2 + (self.zaxis[i] - self.zaxis[j-1])**2)
            self.area[i] = 2.0 * np.pi * self.radius[i] * self.length[i]             
            self.volume[i] = np.pi * self.radius[i]**2 * self.length[i]

        self.area_cross[i] = np.pi * self.radius[i]**2
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
                    self.rho2[i] = 1.0 / (self.rho[i] + self.rho[i+1]) # 1/ohms i+1 last
                    self.gamma[i] = 1.0e11 * self.rho2[i] / (2.0 * np.pi * self.radius[i+1] * self.length[i+1])* 1.0e-5  #mS/cm^2 i+1 both
                
            else:
                self.rho2[i] = 1.0 / (self.rho[i] + self.rho[j-1])
                self.gamma[i] = 1.0e11 * self.rho2[i] / (2.0 * np.pi * self.radius[i] * self.length[i]) * 1.0e-5  #mS/cm^2
               
            self.DiffK[i] = 250.0 * self.area_cross[i]/ (self.length[i] * self.volume[i]) # /s. 250 is the diffusion coefficent in um^2/s
            self.DiffNa[i] = 600.0 * self.area_cross[i]/ (self.length[i] * self.volume[i]) # /s #600
            # self.DiffCa[i] = 5.0 * self.area_cross[i]/ (self.length[i] * self.volume[i])


    def astrocyte_dynamics_diffusion(self, iskip, last):
        v,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, JGluTs = update_astrocyte_dynamics(self.dt, self.identifier, self.gamma, self.parents, self.DiffNa, self.DiffK, self.DiffCa, self.g_l, self.distance_from_soma, self.radius, self.area, self.area_cross, self.volume, self.Cm, self.stim_start_nt, self.stim_end_nt, self.stim_dur, self.glut_stim, self.pot_stim, self.stim_comp_glut, self.stim_comp_pot, self.n_stim, self.n_stimulated, self.N, last, iskip, self.nb_branch_max, self.nb_branch_to_comps)
        return v,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, JGluTs


if __name__ == "__main__":
    start_time = time.time()
    select_cell = "Astrocyte"

    if select_cell == "Astrocyte":        
        astrocyte_data_file = Path("NMO_282188.txt")   # edit path. SHould be good as long as the file is in the same dir as this script
        if not astrocyte_data_file.is_file():
            raise FileNotFoundError(
                f"[ERROR] File not found: {astrocyte_data_file.resolve()}\n"
                "Please make sure a valid astrocytic morphology .txt file path is provided."
    )
        else:
            df = np.loadtxt(astrocyte_data_file) 
            xaxis, yaxis, zaxis, parents, identifier = df[:, 2], df[:, 3], df[:, 4] , df[:,6].astype(int), df[:, 1] 
            parents[0] = 0

            ids = df[:, 0].astype(int)   

            #  Uncomment for parent-child visualization purpose
            # # Build segments (parent -> child, then None to break) 
            # id2idx = {nid: i for i, nid in enumerate(ids)}
            # Xs, Ys, Zs = [], [], []
            # for i, pid in enumerate(parents):
            #     if pid > 0 and pid in id2idx:
            #         p = id2idx[pid]
            #         Xs += [xaxis[p], xaxis[i], None]
            #         Ys += [yaxis[p], yaxis[i], None]
            #         Zs += [zaxis[p], zaxis[i], None] 

            data_size = df.shape[0]        
            
            N = 3     # number of compartment to simulate. 1 if modeling a single cell with whole cell attributes else "data_size" for all compartments           
            N_1 = 0
            N_2 = 3

            # save mutiple csv's for distance and branches
            astrocyte_init = Astrocyte(dt=0.00000010, N=N, Ra=400.0, Cm1=1.0, astrocyte_data=astrocyte_data_file, stim_start_nt=None, stim_end_nt=None, glut_stim=None, pot_stim=None, stim_comp_glut=None, stim_comp_pot=None)
           
            df_dist_soma = pd.DataFrame({
            'Compartment_Index': range(len(astrocyte_init.distance_from_soma)),
            'Distance_from_Soma_um': astrocyte_init.distance_from_soma
            })
            df_dist_soma.to_csv("distance_from_soma.csv", index=False)

            df_branch_to_comps = pd.DataFrame(list(astrocyte_init.branch_to_comps.items()), columns=['branch', 'compartments'])
            df_branch_to_comps.to_csv("branch_to_comps.csv", index=False)

            df_branch_max_dict = pd.DataFrame.from_dict(astrocyte_init.branch_max_dict, orient='index', columns=['max_value'])
            df_branch_max_dict.to_csv("branch_to_max_distance.csv")


            ################################################ Adjust for stimulation timing modifications ##################################################################

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

                vs,  Kos, Naks, Clks, Kks, Nass, time_array, JNaKks, phis, phos, Icoups, IdiffKs, IdiffNas, IdiffCas, Caks, JNaks, JTRPVks, Iks, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, JGluTs = astrocyte.astrocyte_dynamics_diffusion(iskip, last)
                
                comb_df = concat_ast_param(vs, Naks, Kos, Kks, Caks, phos, phis, IdiffNas, JNaks, JNaKks, Iks, time_array, JNBCks, JNHEs, JNKCC1ks, JNCXs, JNaKks_alpha2beta1, JNaKks_alpha2beta2, JGluTs, N_1, N_2)
                
                output_path = Path(__file__).with_name("comb_astro_data.csv")                
                comb_df.to_csv(output_path, index=False)  
                                          
                end_time = time.time()
                elapsed_time = end_time - start_time 
                print(f"Simulation Completed! Total time elapsed: {elapsed_time} seconds")
                 