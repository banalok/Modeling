# %%
######################################### Test for Glutamate and Ex. K+ trends #############################################################################################

import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def run():
    N = 1
    nt= 300000000  # Total number of timesteps
    dt = 0.0000001  # Time step size
    t_in = 20.0  # Start time for recording data
    iskip = 16000  # Store every 1000th point

    stim_start_nt = 220000000   #220000000 #22nd second 1000000000
    stim_end_nt = 220500000  # End of stimulation

    stim_comp_glut = np.arange(0, N)  # Compartments affected by stimulation
    glut_stim = np.ones(N) * 1300.00  # Initial glutamate concentration

    # Allocate space for data storage
    num_records = (nt - int(t_in / dt)) // iskip
    Glus = np.zeros((N, num_records))  # Store glutamate concentrations
    time_array = np.zeros(num_records)  # Store corresponding time points
    kss = np.zeros((N, num_records))
    # Glu_all = np.zeros((N, nt)) 
    time_t = 0.0  # Current simulation time
    j_10 = 0  # Index for storing records
    
    for ii in range(1, nt + 1):
        Glu = np.zeros(N)
        # Update simulation time
        time_t += dt

        # Stimulation phase
        if stim_start_nt*dt <= ii <= stim_end_nt*dt:
            for i, comp in enumerate(stim_comp_glut):
                stim_value = glut_stim[i]
                Glu[comp] = stim_value

        # Exponential decay phase
        elif ii*dt > stim_end_nt*dt:
            for i, comp in enumerate(stim_comp_glut):
                stim_value = glut_stim[i]
                Glu[comp] = stim_value * np.exp(-(ii*dt - stim_end_nt*dt) / 0.5)
        # Glu_all[:, ii] = Glu 
        # Store data every 'iskip' steps after time > t_in
        if time_t > t_in and ii % iskip == 0:
            time_array[j_10] = time_t
            Glus[:, j_10] = Glu
            j_10 += 1  # Increment storage index
    # Glu_all_s = Glu_all.shape
    return time_array, Glus

# Run the simulation and plot the results
time_array, Glus = run()

# Plot the results
plt.plot(time_array, Glus[0])
plt.xlabel('Time (s)')
plt.ylabel('Glu Concentration (µM)')
plt.title('Glutamate Concentration Decay')
plt.show()
#     Ks = np.ones(N) * 2900.0 
#     stim_comp_pot = np.arange(0, N)  # Compartments affected by stimulation
#     pot_stim = np.ones(N) * 1100.00   # -2300 4100.00# to have global, decaying stimulation at each compartment. THis replaces the one that comes from the UI
#     pot_stim_amp = 4000  # 7000 600  # 7000    
#     alpha = 0.01
#     beta = 0.01
#     for ii in range(1, nt+1):
#         # Glu = np.zeros(N)
#         # Pot = np.zeros(N)
#         time_t = time_t + dt 

#         # Icoup = np.zeros(N)
#         # IdiffK = np.zeros(N)
#         # IdiffNa = np.zeros(N)
#         # IdiffCa = np.zeros(N)
#         # Na_stim = np.zeros(N)         
#         for k in range(N):
#             if pot_stim is not None:
#                 if stim_start_nt*dt <= ii*dt <= stim_end_nt*dt:
#                     # corresponding stimulation values
#                     for i, comp in enumerate(stim_comp_pot):                        
#                         if k == comp:
#                             stim_value = pot_stim[i]
#                             Ks[k] = 2900 +  stim_value *(1 - np.exp(-alpha * (ii * dt - stim_start_nt * dt)))

#                         if k+1 == N:
#                             break
#                         else:
#                             pass
#                         # print("No diffusion")

#                 elif ii*dt > stim_end_nt*dt:
#                     # with exponential decay after stimulation ends
#                     for i, comp in enumerate(stim_comp_pot):
#                         if k == comp:                            
#                             stim_value = pot_stim[i]                            
#                             Ks[k] = pot_stim_amp * np.exp(-beta * (ii * dt - stim_end_nt  * dt)) + 2900 * (1 - np.exp(-beta * (ii * dt - stim_end_nt * dt))) #2900 + 4100 * np.exp(-(ii * dt - stim_start_nt * dt) / 0.9) 
                            
#                         if k+1 == N:
#                             break
#                         else:
#                             pass
#                             # print("No diffusion_2")   
#         if time_t > t_in and ii % iskip == 0:
#             time_array[j_10] = time_t
#             kss[:, j_10] = Ks
#             j_10 += 1  # Increment storage index
#     return time_array, kss
    
# # # Run the simulation and plot the results
# time_array, kss = run()
# # Plot the results
# plt.plot(time_array, kss[0])
# plt.xlabel('Time (s)')
# plt.ylabel('[K+] (µM)')
# plt.show()

# %%
###################################### Change in Baseline (Delta Na_i) ##################################################################################################

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
high_K_nak_baseline_1 = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\Normal BL\\comb_astro_data_1000S_HIGHK_14BL.csv')
high_K_nak_baseline_2 = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\comb_astro_data_1000S_HIGHK_24BL_JNAK.csv')
low_k_kk_baseline_1 = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\Normal BL\\comb_astro_data_1000S_VLOWK_14BL.csv')
low_k_kk_baseline_2 = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\comb_astro_data_1000S_VLOWK_24BL_JNAK.csv')

# Normalize each baseline to start from zero
high_K_nak_baseline_1['Nak_0'] -= high_K_nak_baseline_1['Nak_0'].max()
high_K_nak_baseline_2['Nak_0'] -= high_K_nak_baseline_2['Nak_0'].max()
low_k_kk_baseline_1['Nak_0'] -= low_k_kk_baseline_1['Nak_0'].min()
low_k_kk_baseline_2['Nak_0'] -= low_k_kk_baseline_2['Nak_0'].min()

K_highlight_start = 100
K_highlight_end = 220

# Plot for Nak baseline comparison
plt.figure(figsize=(10, 5))
plt.plot(high_K_nak_baseline_1['Time'], high_K_nak_baseline_1['Nak_0'], label='Low Baseline', color='blue')
plt.plot(high_K_nak_baseline_2['Time'], high_K_nak_baseline_2['Nak_0'], label='High Baseline', color='orange')
plt.xlabel('Time [s]')
plt.ylabel('Δ[Na$^+$]$_i$ [mM]')
plt.axvspan(K_highlight_start, K_highlight_end, color='gray', alpha=0.3)
plt.legend()
plt.grid(True)

# Show the Nak baseline plot
plt.show()

# Plot for Kk baseline comparison
plt.figure(figsize=(10, 5))
plt.plot(low_k_kk_baseline_1['Time'], low_k_kk_baseline_1['Nak_0'], label='Low Baseline', color='green')
plt.plot(low_k_kk_baseline_2['Time'], low_k_kk_baseline_2['Nak_0'], label='High Baseline', color='red')
plt.xlabel('Time [s]')
plt.ylabel('Δ[Na$^+$]$_i$ [mM]')
plt.axvspan(K_highlight_start, K_highlight_end, color='gray', alpha=0.3)
plt.legend()
plt.grid(True)

# Show the Kk baseline plot
plt.show()

# %%
################################################# Test for isoform plots ##############################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
folders = [
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv"
]  # Replace with your file paths

# Define isoform symbols
isoform_symbols = {
    "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize the plot
plt.figure(figsize=(10, 6))

# Define colors for each isoform
colors = ["blue", "green", "red", "purple"]

# Loop through each file and plot data
for i, folder in enumerate(folders):
    # Load data
    data = pd.read_csv(folder)
    
    # Ensure the necessary columns exist
    if "Time" not in data.columns or "Nak_0" not in data.columns:
        print(f"Error: 'Time' or 'Nak_0' column missing in {folder}")
        continue
    
    # Extract isoform type from file path
    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    
    # Plot data
    plt.plot(data["Time"], data["Nak_0"], label=isoform_sym, color=colors[i])
    # K_highlight_start = 22
    # K_highlight_end = 23
    # plt.axvspan(K_highlight_start, K_highlight_end, color='gray', alpha=0.3)

# plt.plot(data["Time"], data["Kos_0"], label="[K$^+$]$_o$", color="black")
# Add labels, legend, and grid
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("[Na$^+$]$_i$ [mM]", fontsize=12)
# plt.title("NKA Pump Currents Across Isoforms", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

# %%
# ################################################# Test for isoform plots ( High K+ and NORMALIZEDDDD) ##############################################################################


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
folders = [
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv"
]  # Replace with your file paths

# Define isoform symbols
isoform_symbols = {
    "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize the plot
plt.figure(figsize=(10, 6))

# Define colors for each isoform
colors = ["blue", "green", "red", "purple"]

# Loop through each file and plot normalized data
for i, folder in enumerate(folders):
    # Load data
    data = pd.read_csv(folder)
    
    # Ensure the necessary columns exist
    if "Time" not in data.columns or "Nak_0" not in data.columns:
        print(f"Error: 'Time' or 'Nak_0' column missing in {folder}")
        continue
    
    # Normalize the Nak_0 values
    max_value = data["Nak_0"].min()
    
    data["Normalized_Nak_0"] = data["Nak_0"] - max_value
    
    # Extract isoform type from file path
    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    
    # extract Baseline value
    BL_value = data["Nak_0"].iloc[0]
    # Plot normalized data
    plt.plot(data["Time"], data["Normalized_Nak_0"], label=f"{isoform_sym}: {np.round(BL_value)} mM", color=colors[i])

K_highlight_start = 100
K_highlight_end = 220
plt.axvspan(K_highlight_start, K_highlight_end, color='gray', alpha=0.3)

# Add labels, legend, and grid
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Δ[Na$^+$]$_i$ [mM]", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

# %%
################################################# Test for isoform plots for ATP Consumption Rate ( High K+ and NORMALIZEDDDD) ##############################################################################


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
folders = [
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_HIGHK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\comb_astro_data_HIGHK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_VLOWK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\comb_astro_data_VLOWK+.csv"
]  

isoform_symbols = {
    # "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    # "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(16, 8))  # 2x4 grid of subplots
axes = axes.flatten()  # Flatten to easily index each subplot

# Define colors for each isoform
colors = ["blue", "green"]#, "red", "purple"]

# Loop through each file and plot normalized data
for i, folder in enumerate(folders):
    # Load data
    data = pd.read_csv(folder)
    
    # Ensure the necessary columns exist
    if "Time" not in data.columns or "NKA_pump_0" not in data.columns:
        print(f"Error: 'Time' or 'NKA_pump_0' column missing in {folder}")
        continue
    
    # Extract isoform type from file path
    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    K_type_str = folder.split("\\")[-1].split("_")[-1]
    if K_type_str == "HIGHK+.csv":
        K_type_high = "High K+"
    elif K_type_str == "VLOWK+.csv":
        K_type_low = "Low K+"
    
    # Plot data on the corresponding subplot
    ax = axes[i]
    
    if i == 0 or i ==2:
        ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[0])
    elif i == 1 or i ==3:
        ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[1])
    # elif i == 2 or i ==6:
    #     ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[2])
    # elif i == 3 or i ==7:
    #     ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[3])

    ax.axvspan(100, 220, color='gray', alpha=0.3)  # Highlight region
    if i == 2 or i == 3:
        ax.set_xlabel("Time (s)", fontsize=15)
    if i == 0 or i == 2:
        ax.set_ylabel(" ATP Consumption Rate [mM/s]", fontsize=15) #Na$^+$/K$^+$ Pump Current
        # Add centered text for High K+ or Low K+ based on row
    row_label = K_type_high if i < 2 else K_type_low
    ax.text(0.5, 0.5, row_label, transform=ax.transAxes,
            fontsize=14, color='black', alpha=0.6,
            ha='center', va='center', weight='bold')
    ax.set_ylim([0, 20])
    ax.legend(fontsize=15)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# %%
################################################# Test for isoform plots for NKA PUMP Current ( High K+ and NORMALIZEDDDD) ##############################################################################


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
folders = [
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_HIGHK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_VLOWK+.csv",
    # "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv"
]  

isoform_symbols = {
    # "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    # "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize the plot
fig, axes = plt.subplots(2, 2, figsize=(16, 8))  # 2x4 grid of subplots
axes = axes.flatten()  # Flatten to easily index each subplot

# Define colors for each isoform
colors = ["blue", "green"]#, "red", "purple"]

# Loop through each file and plot normalized data
for i, folder in enumerate(folders):
    # Load data
    data = pd.read_csv(folder)
    
    # Ensure the necessary columns exist
    if "Time" not in data.columns or "NKA_pump_0" not in data.columns:
        print(f"Error: 'Time' or 'NKA_pump_0' column missing in {folder}")
        continue
    
    # Extract isoform type from file path
    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    K_type_str = folder.split("\\")[-1].split("_")[-1]
    if K_type_str == "HIGHK+.csv":
        K_type_high = "High K+"
    elif K_type_str == "VLOWK+.csv":
        K_type_low = "Low K+"
    
    # Plot data on the corresponding subplot
    ax = axes[i]
    
    if i == 0 or i ==2:
        ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[0])
    elif i == 1 or i ==3:
        ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[1])
    # elif i == 2 or i ==6:
    #     ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[2])
    # elif i == 3 or i ==7:
    #     ax.plot(data["Time"], data["NKA_pump_0"], label=isoform_sym, color=colors[3])

    ax.axvspan(100, 220, color='gray', alpha=0.3)  # Highlight region
    if i == 2 or i == 3:
        ax.set_xlabel("Time (s)", fontsize=15)
    if i == 0 or i == 2:
        ax.set_ylabel("Na$^+$/K$^+$ Pump Current [mM/s]", fontsize=15)
        # Add centered text for High K+ or Low K+ based on row
    row_label = K_type_high if i < 2 else K_type_low
    ax.text(0.5, 0.5, row_label, transform=ax.transAxes,
            fontsize=14, color='black', alpha=0.6,
            ha='center', va='center', weight='bold')
    # ax.set_ylim([0, 20])
    ax.legend(fontsize=15)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

#%%
############################################# SAME AS ABOVE BUT TWO PLOTS #############################################################
import os
import pandas as pd
import matplotlib.pyplot as plt

folders = [
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv"
]

isoform_symbols = {"alpha2beta1": "α₂β₁", "alpha2beta2": "α₂β₂"}
colors = {"alpha2beta1": "blue", "alpha2beta2": "green"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
titles = ["High K⁺", "Low K⁺"]

def which_panel(path):
    return 0 if "HIGHK+" in os.path.basename(path) else 1

for fpath in folders:
    data = pd.read_csv(fpath)
    if "Time" not in data.columns or "NKA_pump_0" not in data.columns:
        print(f"Error: 'Time' or 'NKA_pump_0' column missing in {fpath}")
        continue

    y = data["NKA_pump_0"]

    isoform = fpath.split("\\")[-2]
    ax = axes[which_panel(fpath)]
    ax.plot(data["Time"], y,
            label=isoform_symbols.get(isoform, isoform),
            color=colors.get(isoform))

for i, ax in enumerate(axes):
    ax.axvspan(100, 220, color='gray', alpha=0.3)
    ax.set_title(titles[i], fontsize=24, weight='bold')
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.tick_params(axis='x', labelsize=16)   
    ax.tick_params(axis='y', labelsize=14)

axes[0].set_ylim(13, 16)
axes[0].set_ylabel("Na$^+$/K$^+$ Pump Current [mM/s]", fontsize=18)
axes[0].legend(fontsize=18)

plt.tight_layout()
plt.show()



#%%
################################################ Test for isoform plots ( Low K+ and NORMALIZEDDDD) ##############################################################################


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define file paths
folders = [
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\comb_astro_data_VLOWK+.csv"
]  # Replace with your file paths

# Define isoform symbols
isoform_symbols = {
    "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize the plot
plt.figure(figsize=(10, 6))

# Define colors for each isoform
colors = ["blue", "green", "red", "purple"]

# Loop through each file and plot normalized data
for i, folder in enumerate(folders):
    # Load data
    data = pd.read_csv(folder)
    
    # Ensure the necessary columns exist
    if "Time" not in data.columns or "Nak_0" not in data.columns:
        print(f"Error: 'Time' or 'Nak_0' column missing in {folder}")
        continue
    
    # Normalize the Nak_0 values
    min_value = data["Nak_0"].min()
    
    data["Normalized_Nak_0"] = data["Nak_0"] - min_value
    
    # Extract isoform type from file path
    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    
    # extract Baseline value
    BL_value = data["Nak_0"].iloc[0]
    # Plot normalized data
    plt.plot(data["Time"], data["Normalized_Nak_0"], label=f"{isoform_sym}: {np.round(BL_value)} mM", color=colors[i])

K_highlight_start = 100
K_highlight_end = 220
plt.axvspan(K_highlight_start, K_highlight_end, color='gray', alpha=0.3)

# Add labels, legend, and grid
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Δ[Na$^+$]$_i$ [mM]", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()


# %%
######****************________________________________#########################**************************************************************************************

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

# Load data
data = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\Fit_lowK.csv')
time_data = data["Time"].values
Na_i_data = data["Nai"].values
Ko_data = data["Ko"].values

# Calculate dNa_i/dt numerically
dNai_dt = np.gradient(Na_i_data, time_data)

# Define the model function for JNaK
def JNaK_model(Na_i, Ko, JNaKmax, KNak, KKss):
    return JNaKmax * (Na_i**1.5 / (Na_i**1.5 + KNak**1.5)) * (Ko / (Ko + KKss))

# Define the function to fit to dNai/dt data
def fit_func(time, JNaKmax, KNak, KKss):
    return -JNaK_model(Na_i_data, Ko_data, JNaKmax, KNak, KKss)

# Perform curve fitting
p0 = [1.0, 10.0, 10.0]  # Initial guesses for JNaKmax, KNak, KKss
popt, _ = curve_fit(fit_func, time_data, dNai_dt, p0=p0)

# Extract fitted parameters
JNaKmax_fit, KNak_fit, KKss_fit = popt

print(f"Fitted parameters: JNaKmax = {JNaKmax_fit}, KNak = {KNak_fit}, KKss = {KKss_fit}") # Fitted parameters: JNaKmax = -1.8081727882526715e-05, KNak = 27.236605493975365, KKss = -1.9999981797479687
 
# Plot observed Na_i and fitted model
import matplotlib.pyplot as plt

# Generate fitted JNaK values
fitted_JNaK = JNaK_model(Na_i_data, Ko_data, JNaKmax_fit, KNak_fit, KKss_fit)

# Plot dNai/dt vs fitted JNaK
plt.plot(time_data, dNai_dt, 'bo', label='Observed dNa_i/dt')
plt.plot(time_data, -fitted_JNaK, 'r-', label='Fitted JNaK')
plt.xlabel('Time (s)')
plt.ylabel('dNa_i/dt or JNaK')
plt.legend()
plt.show()

plt.plot(time_data, Na_i_data, 'bo', label='Nai')
plt.plot(time_data, -fitted_JNaK, 'r-', label='Fitted JNaK')
plt.xlabel('Time (s)')
plt.ylabel('Na_i and JNaK')
plt.legend()
plt.show()

# %%
###################################################### Test pump strength as a function of Na #################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define constants
NaK = np.linspace(0, 35000, 1000)  # Na+ baseline values (in uM)
Ks = 10000                         # Extracellular K+ (uM)
KNak_alpha1beta1 = 11.09 * 1000
HC_alpha1beta1 = 2.63

KNak_alpha2beta1 = 10.6 * 1000
HC_alpha2beta1 = 2.39

KNak_alpha1beta2 = 6.6 * 1000
HC_alpha1beta2 = 1.79

KNak_alpha2beta2 = 6.8 * 1000
HC_alpha2beta2 = 1.55

KKs_alpha1beta1 = 0.25 * 1000

KKs_alpha2beta1 = 0.91 * 1000

KKs_alpha1beta2 = 0.67 * 1000

KKs_alpha2beta2 = 3.6 * 1000
JNakmax_base = 2.3667e4            # Base maximum pump rate (uM/s)

# Multipliers for JNakmax
multipliers = [1.5, 1.4, 1.3, 1.2, 1.1]

# Create subplots
fig, axes = plt.subplots(nrows=len(multipliers), ncols=1, figsize=(8, 12), sharex=True)

# Font size settings
label_fontsize = 14
legend_fontsize = 12
tick_fontsize = 12

# Loop through each multiplier and generate corresponding subplot
for i, multiplier in enumerate(multipliers):
    JNakmax = JNakmax_base * multiplier  # Adjust JNakmax
    JNaKk = np.zeros(len(NaK))          # Initialize pump current array
    
    # Compute pump current for each NaK value
    for k in range(len(NaK)):
        JNaKk[k] = JNakmax * NaK[k]**HC_alpha1beta1 / (NaK[k]**HC_alpha1beta1 + KNak_alpha1beta1**HC_alpha1beta1) * Ks / (Ks + KKs_alpha1beta1)
    
    # Plot current vs. Na+ baseline
    ax = axes[i]
    JNaKk_percent = (JNaKk / JNakmax) * 100
    
    if multiplier == 1.5:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s at Na$^+$$_i$ Baseline: 14mM"
    elif multiplier == 1.1:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s at Na$^+$$_i$ Baseline: 22mM"
    else:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s"
    
    ax.plot(NaK / 1000, JNaKk / 1000, label=max_pump_curr)
    
    # Set y-axis label for the middle subplot
    if multiplier == 1.3:
        ax.set_ylabel("NKA Current [mM/s]", fontsize=label_fontsize)
    
    # Add legend
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True)
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Set shared x-axis label
axes[-1].set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=label_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()

# %%
###################################################### Test pump strength as a function of K #################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define constants
# NaK = np.linspace(0, 35000, 1000)  # Na+ baseline values (in uM)
NaK = 140000
# Ks = 10000                         # Extracellular K+ (uM)
Ks = np.linspace(0, 10000, 1000)
KNak = 10000.0                     # Na+ affinity constant (uM)
KKss = 1500.0                      # K+ affinity constant (uM)
JNakmax_base = 2.3667e4            # Base maximum pump rate (uM/s)

# Multipliers for JNakmax
multipliers = [1.5, 1.4, 1.3, 1.2, 1.1]

# Create subplots
fig, axes = plt.subplots(nrows=len(multipliers), ncols=1, figsize=(8, 12), sharex=True)

# Font size settings
label_fontsize = 14
legend_fontsize = 12
tick_fontsize = 12

# Loop through each multiplier and generate corresponding subplot
for i, multiplier in enumerate(multipliers):
    JNakmax = JNakmax_base * multiplier  # Adjust JNakmax
    JNaKk = np.zeros(len(Ks))          # Initialize pump current array
    
    # Compute pump current for each NaK value
    for k in range(len(Ks)):
        JNaKk[k] = JNakmax * NaK**1.5 / (NaK**1.5 + KNak**1.5) * Ks[k] / (Ks[k] + KKss)
    
    # Plot current vs. Na+ baseline
    ax = axes[i]
    JNaKk_percent = (JNaKk / JNakmax) * 100
    
    if multiplier == 1.5:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s at Na$^+$$_i$ Baseline: 14mM"
    elif multiplier == 1.1:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s at Na$^+$$_i$ Baseline: 22mM"
    else:
        max_pump_curr = f"Max pump current: {np.round(JNakmax/1000, 2)} mM/s"
    
    ax.plot(Ks / 1000, JNaKk / 1000, label=max_pump_curr)
    
    # Set y-axis label for the middle subplot
    if multiplier == 1.3:
        ax.set_ylabel("NKA Current [mM/s]", fontsize=label_fontsize)
    
    # Add legend
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True)
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Set shared x-axis label
axes[-1].set_xlabel("[K$^+$]$_o$ [mM]", fontsize=label_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()


# %%
###################################################### Test pump strength (% of V_max) as a function of Na #################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define constants for each isoform
NaK = np.linspace(0, 35000, 1000)  # Na+ baseline values (in uM)
Ks = 10000  # Extracellular K+ (uM)

# Isoform-specific constants
isoform_params = {
    "α₁β₁": {"KNak": 11.00 * 1000, "HC": 2.63, "KKs": 0.25 * 1000},
    "α₁β₂": {"KNak": 6.6 * 1000, "HC": 1.79, "KKs": 0.67 * 1000},
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000},
    "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000}
}

JNakmax_base = 2.3667e4 * 1.5  # Base maximum pump rate (uM/s)

# Create subplots (2x2 grid)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)

# Font size settings
label_fontsize = 20
legend_fontsize = 18
tick_fontsize = 18

# Loop through each isoform and generate corresponding subplot
for i, (isoform_type, params) in enumerate(isoform_params.items()):
    # Adjust JNakmax for each isoform
    JNakmax = JNakmax_base  # No multiplier for simplicity in this case
    JNaKk = np.zeros(len(NaK))  # Initialize pump current array
    
    # Compute pump current for each NaK value
    for k in range(len(NaK)):
        JNaKk[k] = JNakmax * NaK[k]**params["HC"] / (NaK[k]**params["HC"] + params["KNak"]**params["HC"]) * Ks / (Ks + params["KKs"])

    # Get row and column index for 2x2 subplot
    row, col = divmod(i, 2)
    ax = axes[row, col]
    
    # Plot current vs. Na+ baseline
    JNaKk_percent = (JNaKk / JNakmax) * 100
    ax.plot(NaK / 1000, JNaKk_percent, label=f"{isoform_type}")

    # Set y-axis label for the middle subplot
    
    
    # Add legend
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True)
    
    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)


# Set shared xandy-axis labels
axes[0, 0].set_ylabel("% of V$_{max}$", fontsize=label_fontsize)
axes[1, 0].set_ylabel("% of V$_{max}$", fontsize=label_fontsize)
axes[1, 0].set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=label_fontsize)
axes[1, 1].set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=label_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()

# %%
###################################################### Test pump strength (% of V_max) as a function of K #################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define constants for extracellular K+ concentrations
NaK_values = list(range(12000, 9500, -200))
# NaK = 12000 # Intracellular Na+ concentration (fixed at 100mM)
Ks = np.linspace(0, 7000, 1000)  # Extracellular K+ range (in uM)

# Isoform-specific constants
isoform_params = {
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000}
    # "α₁β₁": {"KNak": 11.00 * 1000, "HC": 2.63, "KKs": 0.25 * 1000},
    # "α₁β₂": {"KNak": 6.6 * 1000, "HC": 1.79, "KKs": 0.67 * 1000},
    # "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000}
}

JNakmax_base = 2.3667e4 * 1.5  # Base maximum pump rate (uM/s)

# Font size settings
label_fontsize = 20
colors = plt.cm.tab10(np.linspace(0, 1, len(NaK_values)))
legend_fontsize = 14
tick_fontsize = 18

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop through each NaK value from 12000 to 9500 in steps of 500
for idx, NaK in enumerate(NaK_values):
    for isoform_type, params in isoform_params.items():
        JNakmax = JNakmax_base  # Use base maximum pump rate
        JNaKk = np.zeros(len(Ks))  # Initialize pump current array
        
        # Compute pump current for each K+ concentration
        for k in range(len(Ks)):
            JNaKk[k] = JNakmax * NaK**params["HC"] / (NaK**params["HC"] + params["KNak"]**params["HC"]) * Ks[k] / (Ks[k] + params["KKs"])
        
        # Plot % of Vmax vs. K+ concentration
        JNaKk_percent = (JNaKk / JNakmax) * 100
        plt.plot(Ks / 1000, JNaKk_percent, label=f"[Na$^+$]$_i$={NaK/1000}mM  {isoform_type}", color=colors[idx])

# Set axis labels and title
plt.xlabel("[K$^+$]$_o$ [mM]", fontsize=label_fontsize)
plt.ylabel("% of V$_{max}$", fontsize=label_fontsize)
# plt.title("Isoform-Specific NaK Pump Activity", fontsize=label_fontsize)

# Add legend and grid
plt.legend(fontsize=legend_fontsize, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.6, 0.53))
plt.grid(True)

# Adjust tick font size
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Enable minor ticks and adjust parameters
plt.minorticks_on()
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
plt.tick_params(axis='both', which='minor', length=4, width=1, labelsize=tick_fontsize)

# Show plot
plt.tight_layout()
plt.show()



# %%
##################################################### Test pump strength 3 D (% of V_max) as a function of artificial Nai and Ko #################################################################################

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define constants for extracellular K+ concentrations
NaK = np.linspace(0, 50000, 500)  # Intracellular Na+ concentration (fixed at 100mM)
Ks = np.linspace(0, 10000, 500)  # Extracellular K+ range (in uM)

# Isoform-specific constants
isoform_params = {
    # "α₁β₁": {"KNak": 11.00 * 1000, "HC": 2.63, "KKs": 0.25 * 1000},
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000},
    # "α₁β₂": {"KNak": 6.6 * 1000, "HC": 1.79, "KKs": 0.67 * 1000},
    "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000}
}

JNakmax_base = 2.3667e4 * 1.5  # Base maximum pump rate (uM/s)

# Create a meshgrid for NaK and Ks
NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

# Create a 2x2 subplot layout
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=list(isoform_params.keys()),
    vertical_spacing=0.2
)

# Loop through each isoform to compute data and add to the subplot
row_col_map = [(1, 1), (1, 2), (2, 1), (2, 2)]  # Mapping of subplot positions
for (isoform_type, params), (row, col) in zip(isoform_params.items(), row_col_map):
    JNaKk_grid = np.zeros(NaK_grid.shape)  # Initialize pump current array

    # Compute pump current for each NaK and Ks combination in the meshgrid
    for i in range(len(NaK)):
        for j in range(len(Ks)):
            JNaKk_grid[j, i] = (
                JNakmax_base * NaK_grid[j, i]**params["HC"] / 
                (NaK_grid[j, i]**params["HC"] + params["KNak"]**params["HC"]) * 
                Ks_grid[j, i] / (Ks_grid[j, i] + params["KKs"])
            )

    # Plot the surface for this isoform
    JNaKk_percent = (JNaKk_grid / JNakmax_base) * 100  # Convert to percentage of Vmax
    surface = go.Surface(
        z=JNaKk_percent,
        x=NaK_grid / 1000,
        y=Ks_grid / 1000,
        colorscale='Viridis',
        showscale=False
    )
    fig.add_trace(surface, row=row, col=col)

# Update axis labels for each subplot
scenes = {
    f"scene{i + 1}": dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>"
    )
    for i in range(4)
}

# Update the layout with individual scene settings
fig.update_layout(
    **scenes,
    autosize=True,
    height=1800,
    width=2000,
    margin=dict(l=0, r=0, b=0, t=40),
)

# Save the figure as an HTML file
fig.write_html("NaK_pump_activity.html")
print("Saved 2x2 plot as NaK_pump_activity.html")

# %%
##################################################### Test pump strength 3 D WIRE/LINE (% of V_max) as a function of artificial Nai and Ko ################################################################################

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define constants for extracellular K+ concentrations
NaK = np.linspace(0, 50000, 50)  # Intracellular Na+ concentration (fixed at 100mM)
Ks = np.linspace(0, 10000, 50)  # Extracellular K+ range (in uM)

# Isoform-specific constants
isoform_params = {
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000},
    "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000}
}

JNakmax_base = 2.3667e4 * 1.5  # Base maximum pump rate (uM/s)

# Create a 1x2 subplot layout for side-by-side plots
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=list(isoform_params.keys())    
)

# Loop through each isoform and add wireframe plots
for (isoform_type, params), col in zip(isoform_params.items(), range(1, 3)):
    # Create a meshgrid for NaK and Ks
    NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

    # Compute pump currents for this isoform
    JNaKk_grid = (
        JNakmax_base * NaK_grid**params["HC"] / 
        (NaK_grid**params["HC"] + params["KNak"]**params["HC"]) * 
        Ks_grid / (Ks_grid + params["KKs"])
    )
    JNaKk_percent = (JNaKk_grid / JNakmax_base) * 100  # Convert to percentage of Vmax

    # Add wireframe by using multiple traces
    for i in range(len(Ks)):
        fig.add_trace(
            go.Scatter3d(
                x=NaK / 1000,  # Convert to mM
                y=np.full_like(NaK, Ks[i] / 1000),  # Fixed Ks value for the line
                z=JNaKk_percent[i, :],
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False
            ),
            row=1, col=col
        )
    for j in range(len(NaK)):
        fig.add_trace(
            go.Scatter3d(
                x=np.full_like(Ks, NaK[j] / 1000),
                y=Ks / 1000,  # Convert to mM
                z=JNaKk_percent[:, j],
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False
            ),
            row=1, col=col
        )

# Update layout for better visualization with increased font sizes
fig.update_layout(
    # title=dict(
    #     text="Wireframe Plots of Isoforms",
    #     font=dict(size=24)  # Increase title font size
    # ),
    scene=dict(
        xaxis=dict(
            title="[Na<sup>+</sup>]<sub>i</sub> (mM)",
            titlefont=dict(size=25),  # Increase axis label font size
            tickfont=dict(size=18)  # Increase tick font size
        ),
        yaxis=dict(
            title="[K<sup>+</sup>]<sub>e</sub> (mM)",
            titlefont=dict(size=25),
            tickfont=dict(size=18)
        ),
        zaxis=dict(
            title="% of V<sub>max</sub>",
            titlefont=dict(size=25),
            tickfont=dict(size=18)
        )
    ),
    scene2=dict(  # Scene for the second subplot
        xaxis=dict(
            title="[Na<sup>+</sup>]<sub>i</sub> (mM)",
            titlefont=dict(size=25),
            tickfont=dict(size=18)
        ),
        yaxis=dict(
            title="[K<sup>+</sup>]<sub>e</sub> (mM)",
            titlefont=dict(size=25),
            tickfont=dict(size=18)
        ),
        zaxis=dict(
            title="% of V<sub>max</sub>",
            titlefont=dict(size=25),
            tickfont=dict(size=18)
        )    
    ),
    
    autosize=True,
    height=800,
    width=1600,
    margin=dict(l=0, r=0, b=0, t=40)
)

for i, annotation in enumerate(fig.layout.annotations):
    annotation.font.size = 30
# Save the figure as an HTML file
fig.write_html("test_NaK_pump_activity_wireframe_side_by_side.html")
print("Saved side-by-side wireframe plots as NaK_pump_activity_wireframe_side_by_side.html")


# %%
###################################################### Test pump strength overlayed simulated line on artificial 3 D surface  (% of V_max) as a function of Nai and Ko Matplotlib#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# File paths for highK and lowK conditions
highK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv"
lowK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv"
highK_file_alpha2beta2 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv"
lowK_file_alpha2beta2 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv"

# Load simulation data
def load_simulation_data(file_path):
    data = pd.read_csv(file_path)
    # Na_i = data['Nak_0'].values * 1000 #.iloc[0::100]
    # K_o = data['Kos_0'].values * 1000
    # perc_v_max = (data['NKA_pump_0'].values / 35.5) * 100
    Na_i = data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "Nak_0"].values * 1000
    K_o = data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "Kos_0"].values * 1000
    perc_v_max = (data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "NKA_pump_0"].values / 35.5) * 100  
    return Na_i, K_o, perc_v_max

Na_i_highK_alpha2beta1, K_o_highK_alpha2beta1, perc_v_max_highK_alpha2beta1 = load_simulation_data(highK_file_alpha2beta1)
# Na_i_lowK_alpha2beta1, K_o_lowK_alpha2beta1, perc_v_max_lowK_alpha2beta1 = load_simulation_data(lowK_file_alpha2beta1)

# Na_i_highK_alpha2beta2, K_o_highK_alpha2beta2, perc_v_max_highK_alpha2beta2 = load_simulation_data(highK_file_alpha2beta2)
# Na_i_lowK_alpha2beta2, K_o_lowK_alpha2beta2, perc_v_max_lowK_alpha2beta2 = load_simulation_data(lowK_file_alpha2beta2)

# --- Define Isoform Parameters ---
isoform_params = {
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000},
    "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000},
}
JNakmax = 35.5 * 1000  # Example maximum pump rate uM/s

# --- Artificial Data for Surface Plots ---
NaK = np.linspace(0, 150000, 500)  # Intracellular Na+ concentration 
Ks = np.linspace(0, 10000, 500)  # Extracellular K+ range (in uM)
NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

# Create figure with 2x2 subplots for both isoforms
fig = plt.figure(figsize=(18, 12))
elev, azim= -8, -165
# Subplot 1: Isoform α₂β₁ (High K+)
ax1 = fig.add_subplot(221, projection='3d')
perc_v_surface_α2β1_highK = ((JNakmax * NaK_grid**isoform_params["α₂β₁"]["HC"] / (NaK_grid**isoform_params["α₂β₁"]["HC"] + isoform_params["α₂β₁"]["KNak"]**isoform_params["α₂β₁"]["HC"])) *
                             (Ks_grid / (Ks_grid + isoform_params["α₂β₁"]["KKs"]))) / JNakmax * 100
# surf1 = ax1.plot_surface(NaK_grid / 1000, Ks_grid / 1000, perc_v_surface_α2β1_highK, cmap='viridis', edgecolor='none', alpha=0.8)
ax1.set_xlabel(r"Na$^{+}$$_i$ [mM]", fontsize=12)
ax1.set_ylabel(r"K$^{+}$$_o$ [mM]", fontsize=12)
ax1.set_zlabel("% of V$_{\mathrm{max}}$", fontsize=12)
ax1.set_title("α₂β₁ (High K+)")
ax1.plot(Na_i_highK_alpha2beta1 / 1000, K_o_highK_alpha2beta1 / 1000, perc_v_max_highK_alpha2beta1, color='black', marker='none', label="α₂β₁ High K+")
ax1.legend()
ax1.view_init(elev=elev, azim=azim)  # Change elev and azim to rotate the plot

# # Subplot 2: Isoform α₂β₁ (Low K+)
# ax2 = fig.add_subplot(222, projection='3d')
# perc_v_surface_α2β1_lowK = ((JNakmax * NaK_grid**isoform_params["α₂β₁"]["HC"] / (NaK_grid**isoform_params["α₂β₁"]["HC"] + isoform_params["α₂β₁"]["KNak"]**isoform_params["α₂β₁"]["HC"])) *
#                             (Ks_grid / (Ks_grid + isoform_params["α₂β₁"]["KKs"]))) / JNakmax * 100
# surf2 = ax2.plot_surface(NaK_grid / 1000, Ks_grid / 1000, perc_v_surface_α2β1_lowK, cmap='viridis', edgecolor='none', alpha=0.8)
# ax2.set_xlabel(r"Na$^{+}$$_i$ [mM]", fontsize=12)
# ax2.set_ylabel(r"K$^{+}$$_o$ [mM]", fontsize=12)
# ax2.set_zlabel("% of V$_{\mathrm{max}}$", fontsize=12)
# ax2.set_title("α₂β₁ (Low K+)")
# ax2.plot(Na_i_lowK_alpha2beta1 / 1000, K_o_lowK_alpha2beta1 / 1000, perc_v_max_lowK_alpha2beta1, color='blue', marker='none', label="α₂β₁ Low K+")
# ax2.legend()
# ax2.view_init(elev=elev, azim=azim)  # Change elev and azim to rotate the plot

# # Subplot 3: Isoform α₂β₂ (High K+)
# ax3 = fig.add_subplot(223, projection='3d')
# perc_v_surface_α2β2_highK = ((JNakmax * NaK_grid**isoform_params["α₂β₂"]["HC"] / (NaK_grid**isoform_params["α₂β₂"]["HC"] + isoform_params["α₂β₂"]["KNak"]**isoform_params["α₂β₂"]["HC"])) *
#                              (Ks_grid / (Ks_grid + isoform_params["α₂β₂"]["KKs"]))) / JNakmax * 100
# surf3 = ax3.plot_surface(NaK_grid / 1000, Ks_grid / 1000, perc_v_surface_α2β2_highK, cmap='viridis', edgecolor='none', alpha=0.8)
# ax3.set_xlabel(r"Na$^{+}$$_i$ [mM]", fontsize=12)
# ax3.set_ylabel(r"K$^{+}$$_o$ [mM]", fontsize=12)
# ax3.set_zlabel("% of V$_{\mathrm{max}}$", fontsize=12)
# ax3.set_title("α₂β₂ (High K+)")
# ax3.plot(Na_i_highK_alpha2beta2 / 1000, K_o_highK_alpha2beta2 / 1000, perc_v_max_highK_alpha2beta2, color='black', marker='none', label="α₂β₂ High K+")
# ax3.legend()
# ax3.view_init(elev=elev, azim=azim)  # Change elev and azim to rotate the plot

# # Subplot 4: Isoform α₂β₂ (Low K+)
# ax4 = fig.add_subplot(224, projection='3d')
# perc_v_surface_α2β2_lowK = ((JNakmax * NaK_grid**isoform_params["α₂β₂"]["HC"] / (NaK_grid**isoform_params["α₂β₂"]["HC"] + isoform_params["α₂β₂"]["KNak"]**isoform_params["α₂β₂"]["HC"])) *
#                             (Ks_grid / (Ks_grid + isoform_params["α₂β₂"]["KKs"]))) / JNakmax * 100
# surf4 = ax4.plot_surface(NaK_grid / 1000, Ks_grid / 1000, perc_v_surface_α2β2_lowK, cmap='viridis', edgecolor='none', alpha=0.8)
# ax4.set_xlabel(r"Na$^{+}$$_i$ [mM]", fontsize=12)
# ax4.set_ylabel(r"K$^{+}$$_o$ [mM]", fontsize=12)
# ax4.set_zlabel("% of V$_{\mathrm{max}}$", fontsize=12)
# ax4.set_title("α₂β₂ (Low K+)")
# ax4.plot(Na_i_lowK_alpha2beta2 / 1000, K_o_lowK_alpha2beta2 / 1000, perc_v_max_lowK_alpha2beta2, color='blue', marker='none', label="α₂β₂ Low K+")
# ax4.legend()
# ax4.view_init(elev=elev, azim=azim)  # Change elev and azim to rotate the plot

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# %%
###################################################### Test pump strength overlayed simulated line on artificial 3 D surface  (% of V_max) as a function of Nai and Ko #################################################################################

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd  # Assuming you have CSV files or similar formats

# File paths for highK and lowK conditions
highK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv"
lowK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv"
highK_file_alpha2beta2 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv"
lowK_file_alpha2beta2 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv"

# Assuming each file has columns Na_i, K_o, JNaKk (replace with actual column names if different)
def load_simulation_data(file_path):
    data = pd.read_csv(file_path)
    Na_i = data['Nak_0'].iloc[0::100].values * 1000
    K_o = data['Kos_0'].iloc[0::100].values * 1000
    perc_v_max = (data['NKA_pump_0'].iloc[0::100].values / 35.5) * 100
    # Na_i = data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "Nak_0"].values * 1000
    # K_o = data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "Kos_0"].values * 1000
    # perc_v_max = (data.loc[(data["Time"] >= 98) & (data["Time"] <= 105), "NKA_pump_0"].values / 35.5) * 100    
    return Na_i, K_o, perc_v_max

def create_point_labels_lowk(x, y, z):
    max_na_index = np.argmax(x)
    return [
        f"Baseline Na<sup>+</sup><sub>i</sub>: {x[0]:.2f} mM<br>Baseline K<sup>+</sup><sub>o</sub>: {y[0]:.2f} mM<br>% of V<sub>max</sub>: {z[0]:.1f}<br><br>"
        f"Peak Na<sup>+</sup><sub>i</sub>: {x[max_na_index]:.2f} mM<br>Peak K<sup>+</sup><sub>o</sub>: {y[max_na_index]:.2f} mM<br>% of V<sub>max</sub>: {z[max_na_index]:.1f}"
    ]

def create_point_labels_highk(x, y, z):
    min_na_index = np.argmin(x)
    return [
        f"Baseline Na<sup>+</sup><sub>i</sub>: {x[0]:.2f} mM<br>Baseline K<sup>+</sup><sub>o</sub>: {y[0]:.2f} mM<br>% of V<sub>max</sub>: {z[0]:.1f}<br><br>"
        f"Peak Na<sup>+</sup><sub>i</sub>: {x[min_na_index]:.2f} mM<br>Peak K<sup>+</sup><sub>o</sub>: {y[min_na_index]:.2f} mM<br>% of V<sub>max</sub>: {z[min_na_index]:.1f}"
    ]

# Load data for both isoforms and both conditions
Na_i_highK_alpha2beta1, K_o_highK_alpha2beta1, perc_v_max_highK_alpha2beta1 = load_simulation_data(highK_file_alpha2beta1)
Na_i_lowK_alpha2beta1, K_o_lowK_alpha2beta1, perc_v_max_lowK_alpha2beta1 = load_simulation_data(lowK_file_alpha2beta1)

Na_i_highK_alpha2beta2, K_o_highK_alpha2beta2, perc_v_max_highK_alpha2beta2 = load_simulation_data(highK_file_alpha2beta2)
Na_i_lowK_alpha2beta2, K_o_lowK_alpha2beta2, perc_v_max_lowK_alpha2beta2 = load_simulation_data(lowK_file_alpha2beta2)

# --- Define Isoform Parameters ---
isoform_params = {
    "α₂β₁": {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000},
    "α₂β₂": {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000},
}
JNakmax = 35.5 * 1000  # Example maximum pump rate uM/s

# --- Artificial Data for Surface Plots ---
NaK = np.linspace(0, 150000, 500)  # Intracellular Na+ concentration 
Ks = np.linspace(0, 10000, 500)  # Extracellular K+ range (in uM)
NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

# --- Create Subplots ---
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],
           [{'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=["α₂β₁ (High K+ application)", "α₂β₂ (High K+ application)", "α₂β₁ (Low K+ application)", "α₂β₂ (Low K+ application)"],
    vertical_spacing=0.1,
    horizontal_spacing=0.1
)

# --- Add Surface and Line Plots for Each Isoform ---
for col, (isoform, params) in enumerate(isoform_params.items(), start=1):
    # Compute surface plot data
    perc_v_surface = ((
        JNakmax * NaK_grid**params["HC"] / (NaK_grid**params["HC"] + params["KNak"]**params["HC"]) *
        Ks_grid / (Ks_grid + params["KKs"])
    ) / JNakmax) * 100

    # --- Add Surface Plot (same surface on both top and bottom panels for each isoform) ---
    fig.add_trace(go.Surface(
        z=perc_v_surface,
        x=NaK_grid/1000,
        y=Ks_grid/1000,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False,
        name=f"{isoform} Surface",
        showlegend=False
    ), row=1, col=col)
    
    fig.add_trace(go.Surface(
        z=perc_v_surface,
        x=NaK_grid/1000,
        y=Ks_grid/1000,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False,
        name=f"{isoform} Surface",
        showlegend=False
    ), row=2, col=col)

    # --- Add Line Plots for High K+ and Low K+ (on top of surface) ---
    if isoform == "α₂β₁":
        # High K+ line plot for α₂β₁
        fig.add_trace(go.Scatter3d(
            x=Na_i_highK_alpha2beta1/1000,
            y=K_o_highK_alpha2beta1/1000,
            z=perc_v_max_highK_alpha2beta1,
            mode='lines+markers+text',
            line=dict(color='black', width=1),
            marker=dict(size=1, color='black', symbol='circle'),
            text=create_point_labels_highk(Na_i_highK_alpha2beta1/1000, K_o_highK_alpha2beta1/1000, perc_v_max_highK_alpha2beta1),            
            showlegend=False
            # name="α₂β₁ High K+"
        ), row=1, col=1)

        # Low K+ line plot for α₂β₁
        fig.add_trace(go.Scatter3d(
            x=Na_i_lowK_alpha2beta1/1000,
            y=K_o_lowK_alpha2beta1/1000,
            z=perc_v_max_lowK_alpha2beta1,
            mode='lines+markers+text',
            line=dict(color='blue', width=1),
            marker=dict(size=1, color='blue', symbol='circle'),
            text=create_point_labels_lowk(Na_i_lowK_alpha2beta1/1000, K_o_lowK_alpha2beta1/1000, perc_v_max_lowK_alpha2beta1),            
            showlegend=False
            # name="α₂β₁ Low K+
        ), row=2, col=1)

    elif isoform == "α₂β₂":
        # High K+ line plot for α₂β₂
        fig.add_trace(go.Scatter3d(
            x=Na_i_highK_alpha2beta2/1000,
            y=K_o_highK_alpha2beta2/1000,
            z=perc_v_max_highK_alpha2beta2,
            mode='lines+markers+text',
            line=dict(color='black', width=1),
            marker=dict(size=1, color='black', symbol='circle'), 
            text=create_point_labels_highk(Na_i_highK_alpha2beta2/1000, K_o_highK_alpha2beta2/1000, perc_v_max_highK_alpha2beta2),                       
            name="Under High K+ Application"
        ), row=1, col=2)

        # Low K+ line plot for α₂β₂
        fig.add_trace(go.Scatter3d(
            x=Na_i_lowK_alpha2beta2/1000,
            y=K_o_lowK_alpha2beta2/1000,
            z=perc_v_max_lowK_alpha2beta2,
            mode='lines+markers+text',
            line=dict(color='blue', width=1),
            marker=dict(size=1, color='blue', symbol='circle'),
            text=create_point_labels_lowk(Na_i_lowK_alpha2beta2/1000, K_o_lowK_alpha2beta2/1000, perc_v_max_lowK_alpha2beta2),                       
            name="Under Low K+ Application"
        ), row=2, col=2)

# --- Update Layout ---
fig.update_layout(    
    legend=dict(
        x=0.5,  # Move it slightly left (default is 1 for right edge)
        y=1,    # Keep it at the top
        xanchor="right",  # Anchor the legend's right edge
        yanchor="top"     # Anchor the legend's top edge
    ),
    scene=dict(
        # xaxis=dict(range=[9, 12]), limits axis
        # yaxis=dict(range=[2.5, 7.5]),
        # zaxis=dict(range=[38, 43]),
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    ),
    scene2=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    ),
    scene3=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    ),
    scene4=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    ),
    height=1500,
    width=2200)

# Show or Save the Figure
fig.write_html("overlay_highK_and_lowK_on_surface.html")

# %%
######################################################## Simulated alpha2beta1 and alpha2beta2 on top of the surface plot of alpha2beta1 and alpha2beta2   ###################################################
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# File paths for highK conditions
highK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv"
highK_file_alpha2beta2 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv"

# Load data function
def load_simulation_data(file_path):
    data = pd.read_csv(file_path)
    Na_i = data['Nak_0'].iloc[0::100].values * 1000
    K_o = data['Kos_0'].iloc[0::100].values * 1000
    perc_v_max = (data['NKA_pump_0'].iloc[0::100].values / 35.5) * 100
    return Na_i, K_o, perc_v_max

# Load data for α₂β₁ and α₂β₂
Na_i_highK_alpha2beta1, K_o_highK_alpha2beta1, perc_v_max_highK_alpha2beta1 = load_simulation_data(highK_file_alpha2beta1)
Na_i_highK_alpha2beta2, K_o_highK_alpha2beta2, perc_v_max_highK_alpha2beta2 = load_simulation_data(highK_file_alpha2beta2)

# Isoform α₂β₁ parameters
isoform_params_alpha2beta1 = {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000}
isoform_params_alpha2beta2 = {"KNak": 6.8 * 1000, "HC": 1.55, "KKs": 3.6 * 1000}
JNakmax = 35.5 * 1000  # Maximum pump rate uM/s

# Generate artificial data for the surface plot
NaK = np.linspace(0, 150000, 500)  # Intracellular Na+ concentration
Ks = np.linspace(0, 10000, 500)  # Extracellular K+ concentration
NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

# Compute surface values
perc_v_surface_alpha2beta1 = ((JNakmax * NaK_grid**isoform_params_alpha2beta1["HC"] /
                   (NaK_grid**isoform_params_alpha2beta1["HC"] + isoform_params_alpha2beta1["KNak"]**isoform_params_alpha2beta1["HC"]) *
                   Ks_grid / (Ks_grid + isoform_params_alpha2beta1["KKs"])) / JNakmax) * 100

perc_v_surface_alpha2beta2 = ((JNakmax * NaK_grid**isoform_params_alpha2beta2["HC"] /
                   (NaK_grid**isoform_params_alpha2beta2["HC"] + isoform_params_alpha2beta2["KNak"]**isoform_params_alpha2beta2["HC"]) *
                   Ks_grid / (Ks_grid + isoform_params_alpha2beta2["KKs"])) / JNakmax) * 100

# Create the plot
fig = go.Figure()

# Add surface plot for α₂β₁
fig.add_trace(go.Surface(
    z=perc_v_surface_alpha2beta1,
    x=NaK_grid / 1000,  # Convert to mM
    y=Ks_grid / 1000,   # Convert to mM
    colorscale='Viridis',
    opacity=0.8,
    showscale=True,
    name="α₂β₁ Surface"
))

fig.add_trace(go.Surface(
    z=perc_v_surface_alpha2beta2,
    x=NaK_grid / 1000,  # Convert to mM
    y=Ks_grid / 1000,   # Convert to mM
    colorscale='Blues',
    opacity=0.8,
    showscale=False,
    name="α₂β₂ Surface"
))

# Overlay highK data for α₂β₁
fig.add_trace(go.Scatter3d(
    x=Na_i_highK_alpha2beta1 / 1000,  # Convert to mM
    y=K_o_highK_alpha2beta1 / 1000,   # Convert to mM
    z=perc_v_max_highK_alpha2beta1,
    mode='lines+markers',
    line=dict(color='black', width=2),
    marker=dict(size=5, color='black', symbol='circle'),
    name="High K+ α₂β₁"
))

# Overlay highK data for α₂β₂
fig.add_trace(go.Scatter3d(
    x=Na_i_highK_alpha2beta2 / 1000,  # Convert to mM
    y=K_o_highK_alpha2beta2 / 1000,   # Convert to mM
    z=perc_v_max_highK_alpha2beta2,
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=5, color='blue', symbol='circle'),
    name="High K+ α₂β₂"
))

# Update layout
fig.update_layout(
    title="Surface Plot with High K+ Overlay for α₂β₁",
    scene=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        xaxis=dict(            
            range=[8, 23]  
        ),
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
        zaxis=dict(            
            range=[37, 44]  
        )
    ),
    legend=dict(
        x=0.8, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.7)'
    )
)

fig.write_html("overlay_highK_and_lowK_on_alpha2beta1_alpha2beta2_surface_both_isoforms_z_lim.html")


# %%
###################################################### SEPARATE SURFACE AND LINE Test pump strength overlayed simulated line on artificial 3 D surface  (% of V_max) as a function of Nai and Ko #################################################################################
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd  # Assuming you have CSV files or similar formats

# File path for high K+ condition
highK_file_alpha2beta1 = "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv"

# Load data function
def load_simulation_data(file_path):
    data = pd.read_csv(file_path)
    Na_i = data['Nak_0'].iloc[0::100].values * 1000
    K_o = data['Kos_0'].iloc[0::100].values * 1000
    perc_v_max = (data['NKA_pump_0'].iloc[0::100].values / 35.5) * 100
    # Na_i = data.loc[(data["Time"] >= 99) & (data["Time"] <= 101), "Nak_0"].values * 1000
    # K_o = data.loc[(data["Time"] >= 99) & (data["Time"] <= 101), "Kos_0"].values * 1000
    # perc_v_max = (data.loc[(data["Time"] >= 99) & (data["Time"] <= 101), "NKA_pump_0"].values / 35.5) * 100    
    return Na_i, K_o, perc_v_max

# Load data for α₂β₁ under high K+ condition
Na_i_highK_alpha2beta1, K_o_highK_alpha2beta1, perc_v_max_highK_alpha2beta1 = load_simulation_data(highK_file_alpha2beta1)

# Isoform parameters
isoform_params_alpha2beta1 = {"KNak": 10.6 * 1000, "HC": 2.39, "KKs": 0.91 * 1000}
JNakmax = 35.5 * 1000

# Artificial data for surface plot
NaK = np.linspace(0, 150000, 500)  # Intracellular Na+ concentration
Ks = np.linspace(0, 10000, 500)  # Extracellular K+ concentration (uM)
NaK_grid, Ks_grid = np.meshgrid(NaK, Ks)

# Compute surface plot data
perc_v_surface = (
    (JNakmax * NaK_grid**isoform_params_alpha2beta1["HC"] /
     (NaK_grid**isoform_params_alpha2beta1["HC"] + isoform_params_alpha2beta1["KNak"]**isoform_params_alpha2beta1["HC"]) *
     Ks_grid / (Ks_grid + isoform_params_alpha2beta1["KKs"])) / JNakmax
) * 100

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    specs=[[{'type': 'surface'}],
           [{'type': 'scatter3d'}]],
    subplot_titles=["α₂β₁ (High K+ Application) - Surface", "α₂β₁ (High K+ Application) - Scatter"],
    vertical_spacing=0.1
)

# Add surface plot
fig.add_trace(
    go.Surface(
        z=perc_v_surface,
        x=NaK_grid / 1000,
        y=Ks_grid / 1000,
        colorscale='Viridis',
        opacity=0.8,
        showscale=False
    ),
    row=1, col=1
)

# Add scatter/line plot
fig.add_trace(
    go.Scatter3d(
        x=Na_i_highK_alpha2beta1 / 1000,
        y=K_o_highK_alpha2beta1 / 1000,
        z=perc_v_max_highK_alpha2beta1,
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=4, color='red', symbol='circle'),
        name="High K+ Data"
    ),
    row=2, col=1
)

# Update layout
fig.update_layout(
    height=1500,
    width=2200,
    title="α₂β₁ Isoform - High K+ Application",
    scene=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    ),
    scene2=dict(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
    )
)

filename = "separate_surface_line_alpha2beta1.html"
fig.write_html(filename)

# %%
###################################################### Test pump strength 3 D surface (% of V_max) as a function of Simulated Nai and Ko ONLYYY #################################################################################

   
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata

# Define file paths
folders = [
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha1beta2\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\comb_astro_data_VLOWK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\comb_astro_data_HIGHK+.csv",
    "F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\comb_astro_data_VLOWK+.csv"
]

isoform_symbols = {
    "alpha1beta1": "α₁β₁",
    "alpha2beta1": "α₂β₁",
    "alpha1beta2": "α₁β₂",
    "alpha2beta2": "α₂β₂"
}

# Initialize subplot grid: 4 rows (for each isoform), 2 columns (Low K+ and High K+)
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=[f"{isoform_symbols[isoform]} Low K+" for isoform in isoform_symbols]
    + [f"{isoform_symbols[isoform]} High K+" for isoform in isoform_symbols],
    specs=[[{'type': 'surface'}, {'type': 'surface'}]] * 4,
    vertical_spacing=0.07,  # Reduced space between rows
    horizontal_spacing=0.01  # Reduced space between columns
)

# Variables to track the global min and max for z_axis
global_z_min = float('inf')
global_z_max = float('-inf')

# First pass to determine the global min and max for z_axis
for folder in folders:
    # Load data
    data = pd.read_csv(folder)

    # Data for plot
    z_axis = (data["NKA_pump_0"].iloc[0::1000] / 35.5) * 100
    global_z_min = min(global_z_min, z_axis.min())
    global_z_max = max(global_z_max, z_axis.max())

# Create the color scale for the entire range of z_axis values
coloraxis = {
    'colorbar': {
        'title': "% of V<sub>max</sub>",
        'tickvals': np.linspace(global_z_min, global_z_max, 6),
        'ticktext': [f"{np.round(val)}" for val in np.linspace(global_z_min, global_z_max, 6)],
        'len': 0.4,  # Adjust length
        'thickness': 20  # Thickness of the colorbar
    },
    'colorscale': 'Viridis',
    'cmin': global_z_min,
    'cmax': global_z_max
}

# Loop through each file and add the data to the appropriate subplot
for folder in folders:
    # Load data
    data = pd.read_csv(folder)

    isoform_type = folder.split("\\")[-2]
    isoform_sym = isoform_symbols.get(isoform_type, f"Unknown ({isoform_type})")
    K_type = "High K+" if "HIGHK+" in folder else "Low K+"

    # Determine subplot row and column
    row = list(isoform_symbols.keys()).index(isoform_type) + 1
    col = 2 if K_type == "High K+" else 1
    
    # Data for plot
    x_axis = data["Nak_0"].iloc[0::1000].values  # Na_i values (as numpy array)
    y_axis = data["Kos_0"].iloc[0::1000].values  # K_o values (as numpy array)
    z_axis = (data["NKA_pump_0"].iloc[0::1000].values / 35.5) * 100  # NKA pump activity as z values
    # print(x_axis.shape)
    # Create meshgrid for Na_i and K_o
    Na_i_grid, K_o_grid = np.meshgrid(x_axis, y_axis)
    
    # Use griddata for interpolation of z_axis values onto the Na_i_grid, K_o_grid meshgrid
    Z = griddata((x_axis, y_axis), z_axis, (Na_i_grid, K_o_grid), method='linear')

    # Add trace to the appropriate subplot
    fig.add_trace(
        go.Surface(
            x=Na_i_grid,
            y=K_o_grid,
            z=Z,
            coloraxis="coloraxis",  # Use the same coloraxis for all traces
        ),
        row=row, col=col
    )

    # Update axes titles for each subplot
    fig.update_scenes(
        xaxis_title="Na<sup>+</sup><sub>i</sub> [mM]",
        yaxis_title="K<sup>+</sup><sub>o</sub> [mM]",
        zaxis_title="% of V<sub>max</sub>",
        row=row, col=col
    )

# Update layout for the entire figure
fig.update_layout(
    height=1800,  # Adjust height for better visualization
    width=2000,   # Adjust width for better visualization
    coloraxis=coloraxis,  # Apply the global coloraxis
    showlegend=False,
)

# Save and open the combined HTML
filename = "3D_NKA_Na_K_Isoform_Plots_Surface.html"
fig.write_html(filename)
print(f"Combined plot saved to {filename}")


###################################################### Na+ Pathway plots ####################################################################################################################

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load only required columns and skip rows to reduce memory usage
df = pd.read_csv('G://Alok//Isoform_combo//1_max_NKA_current//alpha2beta1 dominant//0.95_alpha2beta1_0.05_alpha2beta2//comb_astro_data.csv',
    # 'F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\leak_fixed_comb_astro_data.csv',
    usecols=['Time', 'Nak_0', 'NaLeak_0', 'NKA_pump_0', 'JNBCk_0', 'JNHE_0', 'JNKCC1k_0', 'JNCX_0', 'Kks_0', 'Kos_0', 'V_0']
    # skiprows=lambda x: x % 20!= 0
)

# Define data columns for plotting and their titles
data_columns = [
    ('Nak_0', "[Na$^+$]$_i$ [mM] vs Time [s]"),
    ('NaLeak_0', "[Na$^+$]$_i$ Leak [mM/s] vs Time [s]"),
    ('NKA_pump_0', "[Na$^+$]/[K$^+$] Pump [mM/s] vs Time [s]"),
    ('JNBCk_0', "NBCe Current [mM/s] vs Time [s]"),
    ('JNHE_0', "NHE Current [mM/s] vs Time [s]"),
    ('JNKCC1k_0', "NKCC1k Current [mM/s] vs Time [s]"),
    ('JNCX_0', "NCX Current [mM/s] vs Time [s]"),
    ('Kks_0', "[K$^+$]$_i$ [mM] vs Time [s]"),
    ('Kos_0', "[K$^+$]$_o$ [mM] vs Time [s]"),
    ('V_0', "V [mV] vs Time [s]")
]

# Create a figure and 3x3 grid of subplots
fig, axes = plt.subplots(4, 3, figsize=(25, 22))  # Adjust figsize to provide enough space
axes = axes.flatten()  # Flatten the 3x3 grid for easy iteration

# Plot each dataset in its respective subplot
for i, (col, title) in enumerate(data_columns):
    if i == 1 or i == 4:
        df[col] = -1.0 * df[col]
    elif i == 2 or i == 6:
        df[col] = df[col] * -3.0
 
    ax = axes[i]
    ax.plot(df['Time'], df[col], label=col, linewidth=1.5)
    # x_lim_start = 98
    # x_lim_end = 102
    # ax.set_xlim(x_lim_start, x_lim_end)
    # y_values_in_range = df[(df["Time"] >= x_lim_start) & (df["Time"] <= x_lim_end)][col]
    # ax.set_ylim(y_values_in_range.min()-0.1, y_values_in_range.max()+0.1)
    # ax.set_xlim(50, 1000)
    ax.set_title(title, fontsize=20)
    # if i == 9:
    #     ax.set_ylim(-81.5, -81)
    # ax.set_xlabel("Time [s]", fontsize=10)
    # ax.set_ylabel(col, fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True)

    ax.axvspan(100, 220, color='gray', alpha=0.3) 

    # # Create zoomed-in inset for the third subplot
    # if i == 2:  # Adjusting the third subplot (index 2)
    #     zoom_start = 99
    #     zoom_end = 102
    #     ax_inset = fig.add_axes([0.8, 0.77, 0.08, 0.08])  # Position and size of the inset
    #     ax_inset.plot(df['Time'], df[col], label=col, linewidth=1.5)
    #     ax_inset.set_xlim(zoom_start, zoom_end)  # Set x-axis limits for zoomed-in view
    #     # ax_inset.set_ylim(max(df[col][zoom_start:zoom_end]), min(df[col][zoom_start:zoom_end]))  # Adjust y-limits based on the zoomed section
    #     # ax_inset.set_title("Zoomed In", fontsize=10)
    #     ax_inset.tick_params(axis='both', which='major', labelsize=8)
    #     ax_inset.grid(True)

# Remove the last empty subplot
for j in range(len(data_columns), len(axes)):
    fig.delaxes(axes[j])  # Delete extra subplots

plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Show the plot
plt.show()


# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Load your dataframe
# # Replace with your actual file path if loading from a file
# columns_to_load = ['Time', 'Nak_0', 'NaLeak_0', 'NKA_pump_0', 'JNBCk_0', 'JNHE_0', 'JNKCC1k_0', 'JNCX_0', 'Kks_0']
# df = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv',
#                     usecols=columns_to_load,
#                     skiprows=lambda x: x % 100000 != 0,  
#                     low_memory=False)  

# # Create a 3x3 subplot grid
# fig = make_subplots(
#     rows=3, cols=3, 
#     subplot_titles=[
#         "Na<sup>+</sup><sub>i</sub> [mM] vs Time", "Na<sup>+</sup><sub>i</sub> Leak [mM/s] vs Time", "NKA_pump_0 vs Time", 
#         "NBCe Current [mM/s] vs Time", "NHE Current [mM/s] vs Time", "NKCC1k Current [mM/s] vs Time", 
#         "NCX Current [nM/s] vs Time", "Na<sup>+</sup><sub>i</sub> [mM] vs Time", ""
#     ],
#     horizontal_spacing=0.1,  # Adjust spacing as needed
#     vertical_spacing=0.15
# )

# # Define data columns and their subplot positions
# data_columns = [
#     ('Nak_0', 1, 1), ('NaLeak_0', 1, 2), ('NKA_pump_0', 1, 3),
#     ('JNBCk_0', 2, 1), ('JNHE_0', 2, 2), ('JNKCC1k_0', 2, 3),
#     ('JNCX_0', 3, 1), ('Kks_0', 3, 2)
# ]

# # Add traces for each subplot
# for col, row, subplot_col in data_columns:
#     print(row, subplot_col)
#     if col == "JNCX_0":
#         df[col] = col * 1000000 # mM/s to nM/s
#     fig.add_trace(go.Scatter(
#         x=df['Time'], 
#         y=df[col],
#         mode='lines',
#         name=col,
#         line=dict(width=2)
#     ), row=row, col=subplot_col)

# # # Update axis labels and layout
# # for row in range(1, 4):  # 3 rows
# #     for col in range(1, 4):  # 3 columns
# #         # Update x-axis label for bottom row
# #         if row == 3:
# #             fig.update_xaxes(
# #                 title_text="Time (s)", 
# #                 title_font=dict(size=12),
# #                 row=row, col=col
# #             )
# #         # Update y-axis label for the first column
# #         if col == 1 and row <= 3:
# #             fig.update_yaxes(
# #                 title_text=data_columns[(row-1)*3 + (col-1)][0], 
# #                 title_font=dict(size=12),
# #                 row=row, col=col
# #             )

# # Update layout for font size
# fig.update_layout(
#     height=900, width=900,  # Adjust figure size
#     title=dict(text="Subplots of Different Variables vs Time", font=dict(size=14)),
#     showlegend=False  # Turn off legend if not needed
# )

# filename = "Na+_pathways_and_K_i.html"
# fig.write_html(filename)

################################################################## Look only at the stimulation duration for NKA pump ###################################################################
#%%
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_HIGHK+.csv', usecols = ["Time", "NKA_pump_0", "Nak_0", "Kks_0", "Kos_0"])

# Filter the data for time between 90s and 130s
df_filtered = df[(df['Time'] >= 90) & (df['Time'] <= 140)]

fig, axs = plt.subplots(4, 1, figsize=(20, 18), sharex=True)

axs[0].plot(df_filtered['Time'], df_filtered['Kos_0'], color='black')

# Highlight the range between 100s and 120s for Nak_0
highlight_nak = df_filtered[(df_filtered['Time'] >= 100) & (df_filtered['Time'] <= 140)]
axs[0].plot(highlight_nak['Time'], highlight_nak['Kos_0'], label='Low K+ Stimulation', color='red', linewidth=2)

# Plot Nak_0 vs Time
axs[1].plot(df_filtered['Time'], df_filtered['Nak_0'], color='black')

# Highlight the range between 100s and 120s for Nak_0
highlight_nak = df_filtered[(df_filtered['Time'] >= 100) & (df_filtered['Time'] <= 140)]
axs[1].plot(highlight_nak['Time'], highlight_nak['Nak_0'], color='red', linewidth=2)

# Plot K_i vs Time
axs[2].plot(df_filtered['Time'], df_filtered['NKA_pump_0'], color='black')

# Highlight the range between 100s and 120s for K_i
highlight_ki = df_filtered[(df_filtered['Time'] >= 100) & (df_filtered['Time'] <= 140)]
axs[2].plot(highlight_ki['Time'], highlight_ki['NKA_pump_0'], color='red', linewidth=2)

# Plot K_i vs Time
axs[3].plot(df_filtered['Time'], df_filtered['Kks_0'], color='black')

# Highlight the range between 100s and 120s for K_i
highlight_ki = df_filtered[(df_filtered['Time'] >= 100) & (df_filtered['Time'] <= 140)]
axs[3].plot(highlight_ki['Time'], highlight_ki['Kks_0'], color='red', linewidth=2)

# Adding labels and title
axs[0].set_ylabel('[K$^+$]$_o$ [mM]', fontsize=23)
axs[1].set_ylabel('[Na$^+$]$_i$ [mM]', fontsize=23)
axs[2].set_ylabel('Na$^+$/K$^+$ Pump Current [mM/s]', fontsize=23)
axs[3].set_ylabel('[K$^+$]$_i$ [mM]', fontsize=23)
axs[3].set_xlabel('Time (s)', fontsize=23)
# fig.suptitle('Nak_0 and K_i vs Time (90s to 130s)', fontsize=14)

# Show legend
axs[0].legend(fontsize=20)

# Display grid
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)
axs[3].grid(True)

axs[0].minorticks_on()
axs[1].minorticks_on()
axs[2].minorticks_on()
axs[3].minorticks_on()

axs[0].tick_params(axis='x', labelsize=20, which="major")  # Change x-tick font size for the first subplot
axs[0].tick_params(axis='y', labelsize=20, which="major")  # Change y-tick font size for the first subplot
axs[1].tick_params(axis='x', labelsize=20, which="major")  # Change x-tick font size for the second subplot
axs[1].tick_params(axis='y', labelsize=20, which="major")  # Change y-tick font size for the second subplot
axs[2].tick_params(axis='x', labelsize=20, which="major")  # Change x-tick font size for the first subplot
axs[2].tick_params(axis='y', labelsize=20, which="major")  # Change y-tick font size for the first subplot
axs[3].tick_params(axis='x', labelsize=20, which="major")  # Change x-tick font size for the first subplot
axs[3].tick_params(axis='y', labelsize=20, which="major")  # Change y-tick font size for the first subplot

axs[0].tick_params(axis='x', labelsize=20, which="minor")  # Change x-tick font size for the first subplot
axs[0].tick_params(axis='y', labelsize=20, which="minor")  # Change y-tick font size for the first subplot
axs[1].tick_params(axis='x', labelsize=20, which="minor")  # Change x-tick font size for the second subplot
axs[1].tick_params(axis='y', labelsize=20, which="minor")  # Change y-tick font size for the second subplot
axs[2].tick_params(axis='x', labelsize=20, which="minor")  # Change x-tick font size for the first subplot
axs[2].tick_params(axis='y', labelsize=20, which="minor")  # Change y-tick font size for the first subplot
axs[3].tick_params(axis='x', labelsize=20, which="minor")  # Change x-tick font size for the first subplot
axs[3].tick_params(axis='y', labelsize=20, which="minor")  # Change y-tick font size for the first subplot


# Show the plot
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the data for alpha2beta1
df_alpha2beta1 = pd.read_csv(
    'F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_VLOWK+.csv',
    usecols=['Time', 'Kks_0'],
    skiprows=lambda x: x % 1000 != 0
)

# Load the data for alpha2beta2
df_alpha2beta2 = pd.read_csv(
    'F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta2\\combb_astro_data_VLOWK+.csv',
    usecols=['Time', 'Kks_0'],
    skiprows=lambda x: x % 1000 != 0
)

# Normalize Kks_0 by subtracting the minimum value
df_alpha2beta1['Kks_0_normalized'] = df_alpha2beta1['Kks_0'] - df_alpha2beta1['Kks_0'].max()
df_alpha2beta2['Kks_0_normalized'] = df_alpha2beta2['Kks_0'] - df_alpha2beta2['Kks_0'].max()

# Plot the data
plt.figure(figsize=(8, 6))
plt.plot(df_alpha2beta1['Time'], df_alpha2beta1['Kks_0_normalized'], label='α₂β₁', color='blue', linewidth=2)
plt.plot(df_alpha2beta2['Time'], df_alpha2beta2['Kks_0_normalized'], label='α₂β₂', color='red', linewidth=2)

# Add labels and legend
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Δ[K$^+$]$_i$ [mM]', fontsize=18)
# plt.title('Normalized Plot of $Kks_0$ vs Time', fontsize=14)
plt.legend(fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=10)  # Font size for major ticks
plt.tick_params(axis='both', which='minor', labelsize=10)   # Font size for minor ticks

# Add grid and show the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


# %%
#############################################subtract Na Leak and NKA pump values element-wise, and plot the result against a Time column##########################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
HighK_alpha2beta1 = pd.read_csv("F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\no_nhe_nbce_ncx_nkcc_combb_astro_data_HIGHK+.csv")

# Calculate the sum
HighK_alpha2beta1['Sum'] = HighK_alpha2beta1['NaLeak_0'] * -1.0 + HighK_alpha2beta1['NKA_pump_0'] * -3.0 #+ HighK_alpha2beta1['JNBCk_0'] + HighK_alpha2beta1['JNHE_0'] * -1.0 + HighK_alpha2beta1['JNKCC1k_0'] + HighK_alpha2beta1['JNCX_0'] * -3.0

# Create side-by-side plots
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
fontsize = 18

# First plot: NaLeak_0 * -1.0
axs[0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['NaLeak_0'] * -1.0, color='blue')
axs[0].set_title('[Na$^+$]$_i$ Leak Current[mM/s]', fontsize=fontsize)
axs[0].set_xlabel('Time', fontsize=fontsize)
axs[0].tick_params(axis='both', labelsize=fontsize)
axs[0].minorticks_on()  # Enable minor ticks
axs[0].grid(True, which='both')  # Show both major and minor grids

# Second plot: NKA_pump_0 * -3.0
axs[1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['NKA_pump_0'] * -3.0, color='orange')
axs[1].set_title('[Na$^+$]/[K$^+$] Pump Current[mM/s]', fontsize=fontsize)
axs[1].set_xlabel('Time', fontsize=fontsize)
axs[1].tick_params(axis='both', labelsize=fontsize)
axs[1].minorticks_on()  # Enable minor ticks
axs[1].grid(True, which='both')  # Show both major and minor grids

# # Third plot: JNBCk_0
# axs[1, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNBCk_0'], color='green')
# axs[1, 0].set_title('NBCe[mM/s]', fontsize=fontsize)
# axs[1, 0].set_xlabel('Time', fontsize=fontsize)
# axs[1, 0].tick_params(axis='both', labelsize=fontsize)
# axs[1, 0].minorticks_on()  # Enable minor ticks
# axs[1, 0].grid(True, which='both')  # Show both major and minor grids

# # Fourth plot: JNKCC1k_0
# axs[1, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNKCC1k_0'], color='red')
# axs[1, 1].set_title('NKCC1[mM/s]', fontsize=fontsize)
# axs[1, 1].set_xlabel('Time', fontsize=fontsize)
# axs[1, 1].tick_params(axis='both', labelsize=fontsize)
# axs[1, 1].minorticks_on()  # Enable minor ticks
# axs[1, 1].grid(True, which='both')  # Show both major and minor grids

# # Fifth plot: JNHE_0 * -1.0
# axs[2, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNHE_0'] * -1.0, color='purple')
# axs[2, 0].set_title('NHE[mM/s]', fontsize=fontsize)
# axs[2, 0].set_xlabel('Time', fontsize=fontsize)
# axs[2, 0].tick_params(axis='both', labelsize=fontsize)
# axs[2, 0].minorticks_on()  # Enable minor ticks
# axs[2, 0].grid(True, which='both')  # Show both major and minor grids

# # Sixth plot: JNCX_0 * -3.0
# axs[2, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNCX_0'] * -3.0, color='brown')
# axs[2, 1].set_title('NCX[mM/s]', fontsize=fontsize)
# axs[2, 1].set_xlabel('Time', fontsize=fontsize)
# axs[2, 1].tick_params(axis='both', labelsize=fontsize)
# axs[2, 1].minorticks_on()  # Enable minor ticks
# axs[2, 1].grid(True, which='both')  # Show both major and minor grids

# Seventh plot: Sum (spanning both columns)
axs[2].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['Sum'], color='black')
axs[2].set_title('Sum of Currents[mM/s]', fontsize=fontsize)
axs[2].set_xlabel('Time', fontsize=fontsize)
axs[2].tick_params(axis='both', labelsize=fontsize)
axs[2].minorticks_on()  # Enable minor ticks
axs[2].grid(True, which='both')  # Show both major and minor grids

# # 8th plot
# axs[3, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['Nak_0'], color='black')
# axs[3, 1].set_title('[Na$^+$]$_i$ [mM]', fontsize=fontsize)
# axs[3, 1].set_xlabel('Time', fontsize=fontsize)
# axs[3, 1].tick_params(axis='both', labelsize=fontsize)
# axs[3, 1].minorticks_on()  # Enable minor ticks
# axs[3, 1].grid(True, which='both')  # Show both major and minor grids

# fig.suptitle('α₂β₁ High K+ Stimulation', fontsize=22)

# Adjust layout
plt.tight_layout()
plt.show()
# %%
#############################################subtract Na Leak and NKA pump values element-wise, and plot the result against a Time column##########################################################
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
HighK_alpha2beta1 = pd.read_csv("F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\combb_astro_data_HIGHK+.csv")

# Calculate the sum
HighK_alpha2beta1['Sum'] = HighK_alpha2beta1['NaLeak_0'] * -1.0 + HighK_alpha2beta1['NKA_pump_0'] * -3.0 + HighK_alpha2beta1['JNBCk_0'] + HighK_alpha2beta1['JNHE_0'] * -1.0 + HighK_alpha2beta1['JNKCC1k_0'] + HighK_alpha2beta1['JNCX_0'] * -3.0

# Create side-by-side plots
fig, axs = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
fontsize = 18

# First plot: NaLeak_0 * -1.0
axs[0, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['NaLeak_0'] * -1.0, color='blue')
axs[0, 0].set_title('[Na$^+$]$_i$ Leak Current[mM/s]', fontsize=fontsize)
axs[0, 0].set_xlabel('Time', fontsize=fontsize)
axs[0, 0].tick_params(axis='both', labelsize=fontsize)
axs[0, 0].minorticks_on()  # Enable minor ticks
axs[0, 0].grid(True, which='both')  # Show both major and minor grids

# Second plot: NKA_pump_0 * -3.0
axs[0, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['NKA_pump_0'] * -3.0, color='orange')
axs[0, 1].set_title('[Na$^+$]/[K$^+$] Pump Current[mM/s]', fontsize=fontsize)
axs[0, 1].set_xlabel('Time', fontsize=fontsize)
axs[0, 1].tick_params(axis='both', labelsize=fontsize)
axs[0, 1].minorticks_on()  # Enable minor ticks
axs[0, 1].grid(True, which='both')  # Show both major and minor grids

# Third plot: JNBCk_0
axs[1, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNBCk_0'], color='green')
axs[1, 0].set_title('NBCe[mM/s]', fontsize=fontsize)
axs[1, 0].set_xlabel('Time', fontsize=fontsize)
axs[1, 0].tick_params(axis='both', labelsize=fontsize)
axs[1, 0].minorticks_on()  # Enable minor ticks
axs[1, 0].grid(True, which='both')  # Show both major and minor grids

# Fourth plot: JNKCC1k_0
axs[1, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNKCC1k_0'], color='red')
axs[1, 1].set_title('NKCC1[mM/s]', fontsize=fontsize)
axs[1, 1].set_xlabel('Time', fontsize=fontsize)
axs[1, 1].tick_params(axis='both', labelsize=fontsize)
axs[1, 1].minorticks_on()  # Enable minor ticks
axs[1, 1].grid(True, which='both')  # Show both major and minor grids

# Fifth plot: JNHE_0 * -1.0
axs[2, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNHE_0'] * -1.0, color='purple')
axs[2, 0].set_title('NHE[mM/s]', fontsize=fontsize)
axs[2, 0].set_xlabel('Time', fontsize=fontsize)
axs[2, 0].tick_params(axis='both', labelsize=fontsize)
axs[2, 0].minorticks_on()  # Enable minor ticks
axs[2, 0].grid(True, which='both')  # Show both major and minor grids

# Sixth plot: JNCX_0 * -3.0
axs[2, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['JNCX_0'] * -3.0, color='brown')
axs[2, 1].set_title('NCX[mM/s]', fontsize=fontsize)
axs[2, 1].set_xlabel('Time', fontsize=fontsize)
axs[2, 1].tick_params(axis='both', labelsize=fontsize)
axs[2, 1].minorticks_on()  # Enable minor ticks
axs[2, 1].grid(True, which='both')  # Show both major and minor grids

# Seventh plot: Sum (spanning both columns)
axs[3, 0].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['Sum'], color='black')
axs[3, 0].set_title('Sum of Currents[mM/s]', fontsize=fontsize)
axs[3, 0].set_xlabel('Time', fontsize=fontsize)
axs[3, 0].tick_params(axis='both', labelsize=fontsize)
axs[3, 0].minorticks_on()  # Enable minor ticks
axs[3, 0].grid(True, which='both')  # Show both major and minor grids

# 8th plot
axs[3, 1].plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1['Nak_0'], color='black')
axs[3, 1].set_title('[Na$^+$]$_i$ [mM]', fontsize=fontsize)
axs[3, 1].set_xlabel('Time', fontsize=fontsize)
axs[3, 1].tick_params(axis='both', labelsize=fontsize)
axs[3, 1].minorticks_on()  # Enable minor ticks
axs[3, 1].grid(True, which='both')  # Show both major and minor grids

fig.suptitle('α₂β₁ High K+ Stimulation', fontsize=22)

# Adjust layout
plt.tight_layout()
plt.show()


###################################################### Reversal potential plot ################################################################################

# %%

import pandas as pd
import matplotlib.pyplot as plt

HighK_alpha2beta1 = pd.read_csv("F:\\spatial\\Python\\figures_astrocytes\\NMO_73320\\Glutamate Stimulation files\\pump uninhibited\\NAK Pump\\alpha2beta1\\constant_V_comb_astro_data_HIGHK+_alpha2beta1.csv")
HighK_alpha2beta1["E_Na_0"] = HighK_alpha2beta1["V_0"] - (HighK_alpha2beta1["NaLeak_0"] / (226.94 * 1.3))

plt.plot(HighK_alpha2beta1['Time'], HighK_alpha2beta1["E_Na_0"])
plt.show()

# %%

############################################################### K_m (K+) and Voltage fit #################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data points
data = pd.read_csv("F:/spatial/Python/alpha2beta1_voltage_data.csv", header=None)
x_data = data[0]
y_data = data[1]
# print(data)
# Fit a quartic polynomial (degree 4)
coeffs_quartic = np.polyfit(x_data, y_data, 4)

# Generate fitted data for plotting
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_quartic = np.polyval(coeffs_quartic, x_fit)

# Plot original data and the quartic fit
plt.scatter(x_data, y_data, color='blue', label='Data points')
plt.plot(x_fit, y_quartic, color='purple', label='Quartic fit (degree 4)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quartic Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()

# Print the quartic equation
print("Quartic fit equation:")
print(f"y = {coeffs_quartic[0]:.5e} * x^4 + {coeffs_quartic[1]:.5e} * x^3 + "
      f"{coeffs_quartic[2]:.5e} * x^2 + {coeffs_quartic[3]:.5e} * x + {coeffs_quartic[4]:.5e}")

# %%
################### Plot distribution of number of cells as a function of Na basline ###########################################################################################

import os
import pandas as pd

# Base directory where all the max NKA current folders are stored
base_dir = "G:/Alok/Isoform_combo"

# Dictionary to store the baseline values
baseline_data = []

# Iterate through each max NKA current folder
for nka_folder in sorted(os.listdir(base_dir)):
    nka_path = os.path.join(base_dir, nka_folder)
    if os.path.isdir(nka_path):  # Ensure it's a directory
        # Iterate through dominant type folders
        for dominant_folder in ["alpha2beta1 dominant", "alpha2beta2 dominant"]:
            dominant_path = os.path.join(nka_path, dominant_folder)
            if os.path.isdir(dominant_path):
                # Iterate through the different weight folders
                for weight_folder in os.listdir(dominant_path):
                    weight_path = os.path.join(dominant_path, weight_folder)
                    if os.path.isdir(weight_path):
                        csv_file = os.path.join(weight_path, "comb_astro_data.csv")
                        if os.path.exists(csv_file):
                            # Read CSV and get the max Na baseline concentration
                            df = pd.read_csv(csv_file)
                            max_na = df["Nak_0"].max()  # Highest value in the file
                            
                            # Store results
                            baseline_data.append((csv_file, max_na))

# Convert results to DataFrame
df_baseline = pd.DataFrame(baseline_data, columns=["File Path", "Na Baseline"])

# Save to CSV for reference
output_csv = os.path.join(base_dir, "Na_Baseline_Results.csv")
df_baseline.to_csv(output_csv, index=False)

print(f"Baseline data saved to: {output_csv}")
print(df_baseline.head())  # Show first few results

# %%
################################################### Distribution plot of Na baseline ###########################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Load the saved baseline results
output_csv = "G:/Alok/Isoform_combo/Na_Baseline_Results.csv" 
df_baseline = pd.read_csv(output_csv)
# Plot distribution of Na Baseline Concentration
plt.figure(figsize=(8, 6))

# df_baseline["Baseline Astrocytes (mM)"] = pd.to_numeric(df_baseline["Baseline Astrocytes (mM)"].dropna())
# print(df_baseline["Baseline Astrocytes (mM)"].dtype)
# sns.histplot(df_baseline["Baseline Astrocytes (mM)"] , bins=30, kde=False, color="skyblue")

sns.histplot(df_baseline["Na Baseline"], bins=30, kde=False, color="skyblue")
plt.xlabel("[Na$^+$]$_i$ [mM]", fontsize=30)
plt.ylabel("Frequency", fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20, width = 2)
# plt.title("Distribution of Na Baseline Concentration")
plt.grid(True)

plt.show()

################################################### Color-coded Distribution plot of Na baseline ###########################################################################################################

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# the csv file input directory
output_csv = "G:/Alok/Isoform_combo/Na_Baseline_Results.csv"
df_baseline = pd.read_csv(output_csv)
# '/' to '\\' in File Path

df_baseline["File Path"] = df_baseline["File Path"].str.replace(r'[\\/]', r'\\\\', regex=True)
# extract max_NKA_current and isoform dominant information from the File Path
df_baseline["% of V_max"] = (df_baseline["File Path"].dropna()).apply(lambda x: x.split("\\")[6].split("_")[0])
df_baseline["isoform_dominant"] = (df_baseline["File Path"].dropna()).apply(lambda x: x.split("\\")[8])

# create the "Isoform Dominance" column with corresponding symbols
df_baseline["Isoform Dominance"] = df_baseline["isoform_dominant"].map({
    "alpha2beta1 dominant": "α₂β₁",
    "alpha2beta2 dominant": "α₂β₂"
})

df_baseline["% of V_max"] = np.round((df_baseline["% of V_max"].dropna()).astype(float)*100)

# color palette
unique_isoforms = df_baseline["Isoform Dominance"].unique()
palette = dict(zip(unique_isoforms, sns.color_palette("Set1", len(unique_isoforms))))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# plot
sns.histplot(data=df_baseline, x="Na Baseline", hue="Isoform Dominance", bins=30, kde=False, palette=palette, multiple="stack", ax=axes[0],legend=True)

# first subplot
axes[0].set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=30)
axes[0].set_ylabel("Frequency", fontsize=30)
axes[0].tick_params(axis='both', which='major', labelsize=20, width = 2)
axes[0].grid(True)

# second subplot with "% of V_max" hue
sns.histplot(data=df_baseline, x="Na Baseline", hue="% of V_max", bins=30, kde=False, multiple="stack", ax=axes[1], legend=True)

axes[1].set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=30)
axes[1].set_ylabel("Frequency", fontsize=30)
axes[1].tick_params(axis='both', which='major', labelsize=20, width = 2)
axes[1].grid(True)

for ax in axes:
    if ax.get_legend() is not None:  # Check if legend exists
        if ax == axes[0]:
            legend = ax.get_legend()
            legend.set_title("Isoforms")
            legend.get_title().set_fontsize(20)  # Set title font size separately
        else:
            legend = ax.get_legend()
            legend.set_title("% of V_max")
            legend.get_title().set_fontsize(20)  # Set title font size separately
            
        for text in ax.get_legend().get_texts():
            text.set_fontsize(20)  # Set font size for legend text


plt.tight_layout()
plt.show()



################################################### Na baseline only for 30_a2b1, 70_a2b2 ###########################################################################################################

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# change the csv file input directory
output_csv = "G:/Alok/Isoform_combo/Na_Baseline_Results.csv"
df_baseline = pd.read_csv(output_csv)
# '/' to '\\' in File Path
df_baseline["File Path"] = df_baseline["File Path"].str.replace(r'[\\/]', r'\\\\', regex=True)

# extract max_NKA_current and isoform dominant information from the File Path
df_baseline["% of V_max"] = df_baseline["File Path"].apply(lambda x: x.split("\\")[6].split("_")[0])
df_baseline["isoform_dominant"] = df_baseline["File Path"].apply(lambda x: x.split("\\")[8])

# since we are interested in 70% a2b2, hence just alpha2beta2 dominant
filtered_df = df_baseline[
    (df_baseline["isoform_dominant"] == "alpha2beta2 dominant")
]

# extracting 30% a2b1 (70% a2b2)
a2b1_30_filtered_df=filtered_df[
    filtered_df["File Path"].apply(lambda x: x.split("\\")[10].split("_")[0] == "0.3")
]

a2b1_30_filtered_df["% of V_max"] = np.round(a2b1_30_filtered_df["% of V_max"].astype(float)*100)


fig, axes = plt.subplots(1, 1, figsize=(16, 6))
# plot
sns.histplot(data=a2b1_30_filtered_df, x="Na Baseline", hue= "% of V_max", bins=30, kde=False)

axes.set_xlabel("[Na$^+$]$_i$ [mM]")
axes.set_title("30% α₂β₁ - 70% α₂β₂")
axes.grid(True)


plt.tight_layout()
plt.show()

################################################### Na baseline via Leak only for 100% of V_max, 30_a2b1, 70_a2b2 ###########################################################################################################

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# change the csv file input directory
output_csv = "G:/Alok/Isoform_combo/Leak_Na_Baseline_Results.csv"
df_baseline = pd.read_csv(output_csv)

fig, axes = plt.subplots(1, 1, figsize=(16, 6))
# plot
sns.histplot(data=df_baseline, x="Na Baseline", hue="Leak Conductance Change (%)", bins=30, kde=False)

# Increase x and y label font sizes to 30
axes.set_xlabel("[Na$^+$]$_i$ [mM]", fontsize=30)
axes.set_ylabel("Frequency", fontsize=30)  # Adding y-label with fontsize 30

# Increase title font size to 30
axes.set_title("30% α₂β₁ - 70% α₂β₂ at 100% of V$_{max}$", fontsize=30)

# Increase tick label font sizes to 20
axes.tick_params(axis='both', which='major', labelsize=20, width=2)

# Enable grid
axes.grid(True)

# Increase legend font size to 20
if axes.get_legend() is not None:
    legend = axes.get_legend()
    legend.set_title("Leak Conductance Change (%)")
    legend.get_title().set_fontsize(20)  # Set title font size
    for text in legend.get_texts():
        text.set_fontsize(20)  # Set text font size
    legend.set_bbox_to_anchor((1.05, 1))

plt.tight_layout()
# plt.subplots_adjust(right=0.85)
plt.show()

##################################################### randon neuron test ####################################################################
# %%

import numpy as np
import matplotlib.pyplot as plt

stim_start = 250000.0
stim_dur = 250000.0
n_stim = 10
n_stimulated=3
stim_amp = 230
Na_stim = np.zeros(50)
for i in range (0, 2000000):
    for k in range(50):
        if (i >= stim_start) and (i < stim_start + stim_dur):
            if (k >= n_stim) and (k < n_stim + n_stimulated):
                Na_stim[k] = stim_amp
        elif (i >= stim_start + stim_dur):
            if (k >= n_stim) and (k < n_stim + n_stimulated):
                Na_stim[k] = stim_amp * np.exp(10 * (stim_start + stim_dur - i) / (stim_start + stim_dur))

x = np.arange(0,50)
plt.plot(x, Na_stim)
plt.show()
# %%

