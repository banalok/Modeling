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

########################################################### SPATIAL PLOTS #####################################################################################################################


# ################################################ Distance from soma and membrane potential #################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
neuron = np.loadtxt("n276156.txt")
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_134.csv")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
time = df['Time'].values

# Extract membrane potentials (V_0, V_1, ..., V_N)
V_columns = [col for col in df.columns if col.startswith('V_')]
Vs = df[V_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Vs[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='Membrane potential [mV]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Membrane Potential')
plt.show()


# %%
# # ################################################ Distance from soma and change in Ko #################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
neuron = np.loadtxt("n276156.txt")
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
time = df['Time'].values

# Extract Ko columns
Ko_columns = [col for col in df.columns if col.startswith('Kos_')]
Kos = df[Ko_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Kos[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='[K$^+$]$_o$ [mM]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in [K$^+$]$_o$')
plt.show()

# %%
# # ################################################ Distance from soma and change in Nai #################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
neuron = np.loadtxt("n276156.txt")
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
time = df['Time'].values

# Extract Ko columns
Nai_columns = [col for col in df.columns if col.startswith('Nai_')]
Nais = df[Nai_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Nais[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='[Na$^+$]$_i$ [mM]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in [Na$^+$]$_i$')
plt.show()

# %%
# ######################################## Change in Na_i at Na_i stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in [Na$^+$]$_i$ at Na$^+$$_i$-stimulated compartments', fontsize=18)
n_stim = 1166
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nai_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Nais[index, :] - np.min(Nais[index, :]), lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    plt.text(0.98, 0.75, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('[Na$^+$]$_i$ [mM]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# ######################################## Change in K_o at Na_i stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in [K$^+$]$_o$ at Na$^+$$_i$-stimulated compartments', fontsize=18)
n_stim = 1166
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Kos[index, :] - np.min(Kos[index, :]), lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('[K$^+$]$_o$ [mM]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
# ######################################## Change in Voltage at Na_i stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in membrane potential (V) at Na$^+$$_i$-stimulated compartments', fontsize=18)
n_stim = 1166
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Vs[index, :], lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm' if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=15)
    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('Membrane Potential [mV]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# ######################################## Change in Na_i at K_o stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=18)
n_stim = 2092
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nai_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Nais[index, :] - np.min(Nais[index, :]), lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('[Na$^+$]$_i$ [mM]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# ######################################## Change in K_o at K_o stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in [K$^+$]$_o$ at K$^+$$_o$-stimulated compartments', fontsize=18)
n_stim = 2092
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Kos[index, :] - np.min(Kos[index, :]), lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    plt.text(0.98, 0.8, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('[K$^+$]$_o$ [mM]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
# ######################################## Change in V at K_o stimulated compartments ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in membrane potential (V) at K$^+$$_o$-stimulated compartments', fontsize=18)
n_stim =2092
n_stimulated = 50
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Vs[index, :], lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=15)
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm' if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=15, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=15)
    if i == 5:
        plt.xlabel('Time [s]', fontsize=18)
    if i == 2:
        plt.ylabel('Membrane Potential [mV]', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
######################################################### SUrface plots for change in Na_i with Na_i stimulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 0.05
stim_end_time = 0.1
stim_after_500ms = 0.6
stim_last_point = 3.9980000000000007
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nai_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[0]] - np.min(Nais[:, time_indices[0]]),
                s=8, cmap='jet')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.05s stimulation', transform=ax1.transAxes, fontsize=16, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[1]] - np.min(Nais[:, time_indices[1]]),
                s=8, cmap='jet')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.05s stimulation', transform=ax2.transAxes, va='center', fontsize=16, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[2]] - np.min(Nais[:, time_indices[2]]),
                s=8, cmap='jet')
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[Na$^+$]$_i$ [mM]', fontsize=16)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=16, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# %%

######################################################### SUrface plots for change in K_o with Na_i stimulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 0.05
stim_end_time = 0.1
stim_after_500ms = 0.6
stim_last_point = 3.9980000000000007
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[0]] - np.min(Kos[:, time_indices[0]]),
                s=8, cmap='jet')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.05s stimulation', transform=ax1.transAxes, fontsize=16, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[1]] - np.min(Kos[:, time_indices[1]]),
                s=8, cmap='jet')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.05s stimulation', transform=ax2.transAxes, va='center', fontsize=16, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[2]] - np.min(Kos[:, time_indices[2]]),
                s=8, cmap='jet')
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[K$^+$]$_o$ [mM]', fontsize=16)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=16, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()
# %%
######################################################### SUrface plots for change in V with Na_i stimulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 0.05
stim_end_time = 0.1
stim_after_500ms = 0.6
stim_last_point = 3.9980000000000007
df = pd.read_csv("G:/Alok/neuron_whole_sim/comb_neuro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[0]],
                s=8, cmap='jet')
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.05s stimulation', transform=ax1.transAxes, fontsize=16, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[1]],
                s=8, cmap='jet')
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.05s stimulation', transform=ax2.transAxes, va='center', fontsize=16, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[2]],
                s=8, cmap='jet')
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('V[mV]', fontsize=16)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=16, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


# %%
############################################## Only K-stimulated ############################################################################################################

# ######################################## Change in Na_i at only K_o stimulated simulation ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
n_stim = 1166
n_stimulated = 100
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_134.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nai_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Nais[index, :], lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm' if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('[Na$^+$]$_i$ [mM]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# ######################################## Change in K_o at only K_o stimulated simulation ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
plt.suptitle('Change in [K$^+$]$_o$ at K$^+$$_o$-stimulated compartments', fontsize=18)
n_stim = 1166
n_stimulated = 100
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Kos[index, :], lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")    
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)
    if i == 2:
        plt.ylabel('[K$^+$]$_o$ [mM]', fontsize=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
# ######################################## Change in Voltage at K_o stimulated simulation ###############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in membrane potential (V) at K$^+$$_o$-stimulated compartments', fontsize=18)
n_stim = 1166
n_stimulated = 100
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
indices = [n_stim, n_stim + n_stimulated - 20, n_stim + n_stimulated - 10, n_stim + n_stimulated, n_stim + n_stimulated + 10, n_stim + n_stimulated + 20]

# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Vs[index, :], lspeck, linewidth=lwidth, color=color)
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")    
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm' if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)
    if i == 2:
        plt.ylabel('Membrane Potential [mV]', fontsize=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
######################################################### SUrface plots for change in Na_i for K_o simulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 1.5
stim_end_time = 2.0
stim_after_500ms = 2.5
stim_last_point = 7.98
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nai_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[0]],
                s=8, cmap='jet', vmin=np.min(Nais), vmax=np.max(Nais))
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.5s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[1]],
                s=8, cmap='jet', vmin=np.min(Nais), vmax=np.max(Nais))
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.5s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[2]],
                s=8, cmap='jet', vmin=np.min(Nais), vmax=np.max(Nais))
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[Na$^+$]$_i$ [mM]', fontsize=18)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# %%

######################################################### SUrface plots for change in K_o with Ko stimulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 1.5
stim_end_time = 2.0
stim_after_500ms = 2.5
stim_last_point = 7.98
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[0]],
                s=8, cmap='jet', vmin=np.min(Kos), vmax=np.max(Kos))
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.5s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[1]],
                s=8, cmap='jet', vmin=np.min(Kos), vmax=np.max(Kos))
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.5s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[2]],
                s=8, cmap='jet', vmin=np.min(Kos), vmax=np.max(Kos))
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[K$^+$]$_o$ [mM]', fontsize=16)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


# %%
######################################################### SUrface plots for change in V with Na_i stimulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]
# Calculate distance from soma
N = 2894
N1 = 0
N2 = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 1.5
stim_end_time = 2.0
stim_after_500ms = 2.5
stim_last_point = 7.98
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_134.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time
print(Vs.shape)

stim_start_index = df.index[df['Time'] == stim_start_time].values[0]
stim_end_index = df.index[df['Time'] == stim_end_time].values[0]
stim_after_500ms_index = df.index[df['Time'] == stim_after_500ms].values[0]
stim_last_point_index = df.index[df['Time'] == stim_last_point].values[0]

time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[0]],
                s=8, cmap='jet', vmin=np.min(Vs), vmax=np.max(Vs))
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 0.5s stimulation', transform=ax1.transAxes, fontsize=16, va='center', wrap=True)
# ax1.view_init(elev=30, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[1]],
                s=8, cmap='jet', vmin=np.min(Vs), vmax=np.max(Vs))
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 0.5s after the end of 0.5s stimulation', transform=ax2.transAxes, va='center', fontsize=16, wrap=True)
# ax2.view_init(elev=30, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[2]],
                s=8, cmap='jet', vmin=np.min(Vs), vmax=np.max(Vs))
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('V[mV]', fontsize=16)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=16, wrap=True)
# ax3.view_init(elev=30, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()






# %%
################################################################ ASTROCYTES ########################################################################################################
# ######################################## Change in Na_i at only K_o stimulated simulation ###############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
length = np.zeros(N)
area = np.zeros(N)
volume = np.zeros(N)
parents = astrocyte[:, 6].astype(int)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)
    if i == 0:
        length[i] = radius[i]
        area[i] = 4.0 * np.pi * radius[i]**2
        volume[i] = 4.0/3.0 * np.pi * radius[i]**3
    else:
        j = parents[i]
        length[i] = np.sqrt((xaxis[i] - xaxis[j-1])**2 + (yaxis[i] - yaxis[j-1])**2 + (zaxis[i] - zaxis[j-1])**2)
        area[i] = 2.0 * np.pi * radius[i] * length[i] 
        volume[i] = np.pi * radius[i]**2 * length[i]
lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
# n_stim = 25
# n_stimulated = 50
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
indices = [78, 80, 121, 128, 163, 212]

# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nak_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    # color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Nais[index, :], lspeck, linewidth=lwidth) #, color=color
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    vol = volume[index]
    label_text = f'Stimulated Comp: {index}'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('[Na$^+$]$_i$ [mM]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# ######################################## Change in K_o at only K_o stimulated simulation ###############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
# n_stim = 25
# n_stimulated = 50
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
# df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
indices = [78, 80, 121, 128, 163, 212]

# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Kos_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    # color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Nais[index, :], lspeck, linewidth=lwidth) #, color=color
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}' #if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('[K$^+$]$_o$ [mM]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
# ######################################## Change in Voltage at K_o stimulated simulation ###############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
# n_stim = 25
# n_stimulated = 50
df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
# df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
indices = [78, 80, 121, 128, 163, 212]

# Fix orientation: transpose if needed
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    # color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Vs[index, :], lspeck, linewidth=lwidth) #, color=color
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm' #if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('Membrane Potential [mV]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
# ######################################## Change in Ca2+ ###############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
length = np.zeros(N)
area = np.zeros(N)
volume = np.zeros(N)
parents = astrocyte[:, 6].astype(int)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)
    if i == 0:
        length[i] = radius[i]
        area[i] = 4.0 * np.pi * radius[i]**2
        volume[i] = 4.0/3.0 * np.pi * radius[i]**3
    else:
        j = parents[i]
        length[i] = np.sqrt((xaxis[i] - xaxis[j-1])**2 + (yaxis[i] - yaxis[j-1])**2 + (zaxis[i] - zaxis[j-1])**2)
        area[i] = 2.0 * np.pi * radius[i] * length[i] 
        volume[i] = np.pi * radius[i]**2 * length[i]

lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
# n_stim = 25
# n_stimulated = 50
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
indices = [0, 1, 3, 10, 11, 13]#[78, 80, 121, 128, 163, 212]

# Fix orientation: transpose if needed
Cais = df[[col for col in df.columns if col.startswith('Cak_')]].values
Cais = Cais.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    # color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Cais[index, :], lspeck, linewidth=lwidth) #, color=color
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    vol = volume[index]
    label_text = f'Distance: {distance:.2f} µm\nVolume: {vol:.2f} µm$^3$' #  Stimulated Comp: {index}\n \nVolume: {vol:.2f} µm^3 #if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('[Ca$^2+$]$_i$ [nM]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
# ######################################## Change in IP3 ###############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
length = np.zeros(N)
area = np.zeros(N)
volume = np.zeros(N)
parents = astrocyte[:, 6].astype(int)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)
    if i == 0:
        length[i] = radius[i]
        area[i] = 4.0 * np.pi * radius[i]**2
        volume[i] = 4.0/3.0 * np.pi * radius[i]**3
    else:
        j = parents[i]
        length[i] = np.sqrt((xaxis[i] - xaxis[j-1])**2 + (yaxis[i] - yaxis[j-1])**2 + (zaxis[i] - zaxis[j-1])**2)
        area[i] = 2.0 * np.pi * radius[i] * length[i] 
        volume[i] = np.pi * radius[i]**2 * length[i]
lwidth = 2  # Line width for plots
lspeck = '-'  # Line style for plots
plt.figure(figsize=(15, 15))
# plt.suptitle('Change in [Na$^+$]$_i$ at K$^+$$_o$-stimulated compartments', fontsize=30)
# n_stim = 25
# n_stimulated = 50
df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
# df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
indices = [78, 80, 121, 128, 163, 107]

# Fix orientation: transpose if needed
Iks = df[[col for col in df.columns if col.startswith('Ik_')]].values
Iks = Iks.T  # Transpose so rows are distances and columns are time

time = df['Time'].values

for i, index in enumerate(indices):
    plt.subplot(6, 1, i + 1)
    # color = 'red' if n_stim <= index < n_stim + n_stimulated else 'blue'
    plt.plot(time, Iks[index, :], lspeck, linewidth=lwidth) #, color=color
    plt.tick_params(axis='both', which='major', labelsize=30, width = 2)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', labelsize=10, width = 2, color="black")
    distance = distance_from_soma[index]
    vol = volume[index]
    label_text = f'Stimulated Comp: {index}\nDistance: {distance:.2f} µm\nVolume: {vol:.2f} µm^3' #if color == 'red' else f'Comp: {index}\nDistance: {distance:.2f} µm'
    # plt.text(1, 1, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))
    plt.legend([label_text], loc='upper left', bbox_to_anchor=(1, 1), fontsize=30)
    # plt.text(0.98, 0.8, f'{label_text}', transform=plt.gca().transAxes, va='center', ha='right', fontsize=18, color=color, bbox=dict(facecolor='white', edgecolor=color))
    # label_text = f'Stimulated Comp: {index}' if color == 'red' else f'Comp: {index}'
    # plt.text(0.98, 0.5, f'{label_text}\nDistance: {distance:.2f} µm', transform=plt.gca().transAxes, va='center', ha='right', fontsize=30, color=color, bbox=dict(facecolor='white', edgecolor=color))

    if i == 5:
        plt.xlabel('Time [s]', fontsize=30)      
       
    if i == 2:
        plt.ylabel('[IP$_3$]$_i$ [µM]', fontsize=30)        
        

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%
######################################################### SUrface plots for change in Na_i for K_o simulation ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('dark_background')  # Dark theme
astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 22
stim_end_time = 23
stim_after_500ms = 24
stim_last_point = 29.9984
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Nais = df[[col for col in df.columns if col.startswith('Nak_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

stim_end_index = (df['Time'] - stim_end_time).abs().idxmin()
stim_after_500ms_index = (df['Time'] - stim_after_500ms).abs().idxmin()
stim_last_point_index = (df['Time'] - stim_last_point).abs().idxmin()

print(stim_end_index, stim_after_500ms_index, stim_last_point_index)
time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[0]],
                s=50, cmap='turbo', vmin=np.min(Nais), vmax=np.max(Nais))
ax1.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax1.set_facecolor('black') 
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 1s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=45, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[1]],
                s=50, cmap='turbo', vmin=np.min(Nais), vmax=np.max(Nais))
ax2.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax2.set_facecolor('black') 
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 1s after the end of 1s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=45, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Nais[:, time_indices[2]],
                s=50, cmap='turbo', vmin=np.min(Nais), vmax=np.max(Nais))
ax3.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax3.set_facecolor('black') 
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[Na$^+$]$_i$ [mM]', fontsize=18)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=45, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# %%

######################################################### SUrface plots for change in K_o  ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('dark_background')  # Dark theme
astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 22
stim_end_time = 23
stim_after_500ms = 24
stim_last_point = 29.9984
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Kos = df[[col for col in df.columns if col.startswith('Kos_')]].values
Kos = Kos.T  # Transpose so rows are distances and columns are time

stim_end_index = (df['Time'] - stim_end_time).abs().idxmin()
stim_after_500ms_index = (df['Time'] - stim_after_500ms).abs().idxmin()
stim_last_point_index = (df['Time'] - stim_last_point).abs().idxmin()

print(stim_end_index, stim_after_500ms_index, stim_last_point_index)
time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[0]],
                s=50, cmap='turbo', vmin=np.min(Kos), vmax=np.max(Kos))
ax1.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax1.set_facecolor('black') 
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 1s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=45, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[1]],
                s=50, cmap='turbo', vmin=np.min(Kos), vmax=np.max(Kos))
ax2.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax2.set_facecolor('black') 
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 1s after the end of 1s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=45, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Kos[:, time_indices[2]],
                s=50, cmap='turbo', vmin=np.min(Kos), vmax=np.max(Kos))
ax3.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax3.set_facecolor('black') 
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[K$^+$]$_o$ [mM]', fontsize=18)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=45, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


# %%
######################################################### SUrface plots for change in Ca2+ ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('dark_background')  # Dark theme
astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 22
stim_end_time = 23
stim_after_500ms = 24
stim_last_point = 30
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Cais= df[[col for col in df.columns if col.startswith('Cak_')]].values
Cais = Cais.T  # Transpose so rows are distances and columns are time

stim_end_index = (df['Time'] - stim_end_time).abs().idxmin()
stim_after_500ms_index = (df['Time'] - stim_after_500ms).abs().idxmin()
stim_last_point_index = (df['Time'] - stim_last_point).abs().idxmin()

print(stim_end_index, stim_after_500ms_index, stim_last_point_index)
time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Cais[:, time_indices[0]],
                s=50, cmap='turbo', vmin=np.min(Cais), vmax=np.max(Cais))
ax1.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax1.set_facecolor('black') 
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 1s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=45, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Cais[:, time_indices[1]],
                s=50, cmap='turbo', vmin=np.min(Cais), vmax=np.max(Cais))
ax2.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax2.set_facecolor('black') 
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 1s after the end of 1s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=45, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Cais[:, time_indices[2]],
                s=50, cmap='turbo', vmin=np.min(Cais), vmax=np.max(Cais))
ax3.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax3.set_facecolor('black') 
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('[Ca$^2+$]$_i$ [nM]', fontsize=18)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=45, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# %%
######################################################### SUrface plots for change in V ############################################################################## at 1.5s for 0.5s, so ends at 2s
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('dark_background')  # Dark theme
astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

stim_start_time = 22
stim_end_time = 23
stim_after_500ms = 24
stim_last_point = 30
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv") #, usecols=["Time", "Nai_1"]
# Fix orientation: transpose if needed
Vs= df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

stim_end_index = (df['Time'] - stim_end_time).abs().idxmin()
stim_after_500ms_index = (df['Time'] - stim_after_500ms).abs().idxmin()
stim_last_point_index = (df['Time'] - stim_last_point).abs().idxmin()

print(stim_end_index, stim_after_500ms_index, stim_last_point_index)
time_indices = [stim_end_index, stim_after_500ms_index, stim_last_point_index]
zoom_fac = 0.4
# time_indices = [
#     (neuron.stim_start_nt + neuron.stim_dur-1) // N_skip,
#     (neuron.stim_start_nt + neuron.stim_dur -1 + 500 / neuron.dt) // N_skip,
#     int(T / N_skip) - 1
# ]
# time_indices = [int(idx) for idx in time_indices]
fig = plt.figure(figsize=(18, 15))
from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 3, width_ratios=[5, 5, 5])
# Plot 1
ax1 = fig.add_subplot(gs[0], projection='3d')
sc1 = ax1.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[0]],
                s=50, cmap='turbo', vmin=np.min(Vs), vmax=np.max(Vs))
ax1.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax1.set_facecolor('black') 
ax1.set_box_aspect([1, 1, 1])
ax1.set_xlabel('x-axis [µm]')
ax1.set_ylabel('y-axis [µm]')
ax1.set_zlabel('z-axis [µm]')
ax1.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax1.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax1.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc1, ax=ax1, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax1.grid(True)
ax1.text2D(0.1, -0.2, s = f't = {stim_end_time}s;\nat the end of 1s stimulation', transform=ax1.transAxes, fontsize=18, va='center', wrap=True)
# ax1.view_init(elev=45, azim=45)

# Plot 2
ax2 = fig.add_subplot(gs[1], projection='3d')
sc2 = ax2.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[1]],
                s=50, cmap='turbo', vmin=np.min(Vs), vmax=np.max(Vs))
ax2.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax2.set_facecolor('black') 
ax2.set_box_aspect([1, 1, 1])
ax2.set_xlabel('x-axis [µm]')
ax2.set_ylabel('y-axis [µm]')
ax2.set_zlabel('z-axis [µm]')
ax2.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax2.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax2.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
# fig.colorbar(sc2, ax=ax2, label='[Na$^+$]$_i$ [mM]', shrink=0.5, aspect=10)
ax2.grid(True)
ax2.text2D(0.1, -0.2, s = f't = {stim_after_500ms}s;\n 1s after the end of 1s stimulation', transform=ax2.transAxes, va='center', fontsize=18, wrap=True)
# ax2.view_init(elev=45, azim=45)

# Plot 3
ax3 = fig.add_subplot(gs[2], projection='3d')
sc3 = ax3.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], c=Vs[:, time_indices[2]],
                s=50, cmap='turbo', vmin=np.min(Vs), vmax=np.max(Vs))
ax3.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')
# ax3.set_facecolor('black') 
ax3.set_box_aspect([1, 1, 1])
ax3.set_xlabel('x-axis [µm]')
ax3.set_ylabel('y-axis [µm]')
ax3.set_zlabel('z-axis [µm]')
ax3.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax3.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax3.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
cbar = fig.colorbar(sc3, ax=[ax1, ax2, ax3], orientation='horizontal')
cbar.ax.tick_params(labelsize=14)  # Change the size of the ticks
cbar.set_label('Membrane Potential [mV]', fontsize=18)  # Change the size of the label
ax3.grid(True)
ax3.text2D(0.18, -0.2, s = f't = {np.round(stim_last_point)}s;\nat the end of simulation', va='center', transform=ax3.transAxes, fontsize=18, wrap=True)
# ax3.view_init(elev=45, azim=45)

ax1.set_axis_off()
ax2.set_axis_off()
ax3.set_axis_off()
plt.subplots_adjust(top=0.85, bottom=0.3)
# plt.suptitle('Change in [Na$^+$]$_i$', fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# %%
# # ################################################ Distance from soma and change in Nai #################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
astrocyte = np.loadtxt("NMO_73320.txt")
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data_const_vol.csv")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
time = df['Time'].values

# Extract Ko columns
Nai_columns = [col for col in df.columns if col.startswith('Nak_')]
Nais = df[Nai_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Nais[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='[Na$^+$]$_i$ [mM]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in [Na$^+$]$_i$ (constant compartmental volume)')
plt.show()

############################################################## Ko distance ##########################################################################################
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
astrocyte = np.loadtxt("NMO_73320.txt")
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
time = df['Time'].values

# Extract Ko columns
Ko_columns = [col for col in df.columns if col.startswith('Kos_')]
Kos = df[Ko_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Kos[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='[K$^+$]$_o$ [mM]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in [K$^+$]$_o$')
plt.show()


########################################################## Ca2+ distance ##########################################################################
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
astrocyte = np.loadtxt("NMO_73320.txt")
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
time = df['Time'].values

# Extract Ko columns
Ca_columns = [col for col in df.columns if col.startswith('Cak_')]
Cais = df[Ca_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Cais[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='[Ca$^2+$]$_i$ [nM]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in [Ca$^2+$]$_i$')
plt.show()
# %%
################################################## V distance ######################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
astrocyte = np.loadtxt("NMO_73320.txt")
# df = pd.read_csv("G:/Alok/F_Drive_CARD/spatial/Python/figures_astrocytes/NMO_73320/Glutamate Stimulation files/pump uninhibited/1330uM_globally_decay_100ms/comb_astro_data.csv")
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
time = df['Time'].values

# Extract Ko columns
V_columns = [col for col in df.columns if col.startswith('V_')]
Vs = df[V_columns].values.T  # Transpose so that rows correspond to distances

# Calculate distance from soma
N = 213
N1 = 0
N2 = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Set indices for the region of interest
N_1, N_2 = 0, len(distance_from_soma)  # Adjust if needed

# Prepare the data for plotting
y = distance_from_soma[N_1:N_2]
data = Vs[N_1:N_2, :]

# Plot the heatmap
plt.figure()
plt.imshow(data, aspect='auto', extent=[time.min(), time.max(), y.min(), y.max()], origin='lower', cmap='jet')
plt.colorbar(label='Membrane Potential [mV]')

plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title('Change in Membrane Potential')
plt.show()



# %%
################################## ANIMATION MP4 ############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import os

# Load the neuron morphology data
neuron = np.loadtxt("n276156.txt")
xaxis, yaxis, zaxis, radius, identifier = neuron[:, 2], neuron[:, 3], neuron[:, 4], neuron[:, 5], neuron[:, 1]

# Calculate distance from soma
N = 2894
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Load the simulation data
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
time_points = df['Time'].values
Nais = df[[col for col in df.columns if col.startswith('Kos_')]].values
Nais = Nais.T  # Transpose so rows are distances and columns are time

# Calculate global min and max for consistent color scaling
global_min = np.min(Nais)
global_max = np.max(Nais)
print(f"Global Ko+ concentration range: {global_min:.4f} to {global_max:.4f}")

# Match the exact time points used in your static plots
stim_end_time = 0.1
stim_after_500ms = 0.6
stim_last_point = 3.998

# Find indices for these specific time points (for reference)
stim_end_idx = np.argmin(np.abs(time_points - stim_end_time))
stim_after_500ms_idx = np.argmin(np.abs(time_points - stim_after_500ms))
stim_last_point_idx = np.argmin(np.abs(time_points - stim_last_point))

print(f"Static plot time points: {time_points[stim_end_idx]:.2f}s, {time_points[stim_after_500ms_idx]:.2f}s, {time_points[stim_last_point_idx]:.2f}s")

# Use all available time points for maximum temporal resolution
selected_frames = list(range(0, len(time_points)))

# Print information about frames
print(f"Using all {len(selected_frames)} frames from the original data")

# Set up the figure for animation with the same settings as your original plots
zoom_fac = 0.4
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
ax.set_axis_off()

# Initial scatter plot with global color scale
initial_frame = selected_frames[0]
scat = ax.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], 
                 c=Nais[:, initial_frame],
                 s=8, cmap='jet', vmin=global_min, vmax=global_max)

# Don't set a specific viewing angle - use the same as your original plots
# Your original code doesn't explicitly set a view_init, so we'll use the matplotlib default
# If you need to match a specific orientation, uncomment and adjust these values:
# ax.view_init(elev=30, azim=-60)  # Try different values to match your static plots

# Title with time
title = ax.text2D(0.4, 0.9, "", transform=ax.transAxes, 
                 fontsize=25, ha='center')

# Color bar
cbar = fig.colorbar(scat, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('[K$^+$]$_o$ [mM]', fontsize=25) # [K$^+$]$_o$ [mM]
cbar.ax.tick_params(labelsize=20)
# Function to update the plot for each frame
def update(frame_idx):
    frame = selected_frames[frame_idx]
    current_time = time_points[frame]
    
    # Use actual Na+ concentration values
    color_data = Nais[:, frame]
    
    # Update color data
    scat.set_array(color_data)
    
    # Update title
    title.set_text(f'Time: {current_time:.2f} s')
    
    # Use fixed global color scale for all frames
    scat.set_clim(global_min, global_max)
    
    return scat, title

# Create the animation with a faster playback speed
ani = FuncAnimation(fig, update, frames=len(selected_frames), 
                   blit=True, interval=20)  # reduced interval for faster playback

# Save as MP4 with higher fps for faster playback
output_mp4 = "neuron_Ko_animation.mp4"
print(f"Saving animation as {output_mp4}...")
ani.save(output_mp4, writer='ffmpeg', fps=30, dpi=100)  # increased fps from 10 to 30

print("Animation saved successfully!")

# Show the animation in the notebook/IDE if desired
plt.tight_layout()
plt.show()


# %%
######################################################## ANIMATED ASTROCYTE ###############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Load the astrocyte morphology data
astrocyte = np.loadtxt("NMO_73320.txt")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]

# Calculate distance from soma
N = 213
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Load the simulation data
df = pd.read_csv("G:/Alok/astro_whole_sim/glutamate stimulation/comb_astro_data.csv")
time_points = df['Time'].values
Vs = df[[col for col in df.columns if col.startswith('V_')]].values
Vs = Vs.T  # Transpose so rows are distances and columns are time

# Calculate global min and max for consistent color scaling
global_min = np.min(Vs)
global_max = np.max(Vs)
print(f"Global membrane potential range: {global_min:.4f} to {global_max:.4f} mV")

# Select frames at specified time intervals
start_time = 20.0
end_time = 30.0
time_step = 0.5  # seconds

# Calculate time points at specified intervals
target_times = np.arange(start_time, end_time + time_step, time_step)

# Find the closest indices in your actual time_points data
selected_frames = []
for target_time in target_times:
    closest_idx = np.argmin(np.abs(time_points - target_time))
    selected_frames.append(closest_idx)
    print(f"Time {target_time} s → actual time {time_points[closest_idx]:.2f} s (index {closest_idx})")

# Important time points
stim_start_time = 22.0
stim_end_time = 23.0

# Set up the figure for animation
zoom_fac = 0.4
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(min(xaxis)*zoom_fac, max(xaxis)*zoom_fac)
ax.set_ylim(min(yaxis)*zoom_fac, max(yaxis)*zoom_fac)
ax.set_zlim(min(zaxis)*zoom_fac, max(zaxis)*zoom_fac)
ax.set_axis_off()

# Initial scatter plot with global color scale
initial_frame = selected_frames[0]
scat = ax.scatter(xaxis[0:N], yaxis[0:N], zaxis[0:N], 
                 c=Vs[:, initial_frame],
                 s=50, cmap='turbo', vmin=global_min, vmax=global_max)

# Mark the soma
soma = ax.scatter(xaxis[0], yaxis[0], zaxis[0], s=100, edgecolors='black', marker='o')

# Title with time
title = ax.text2D(0.25, 0.9, "", transform=ax.transAxes, 
                 fontsize=25, ha='center')

# Add stimulation status indicator
stim_status = ax.text2D(0.25, 0.85, "", transform=ax.transAxes,
                       fontsize=14, ha='center', color='red')

# Color bar
cbar = fig.colorbar(scat, ax=ax, orientation='horizontal', pad=0.05)
cbar.set_label('Membrane Potential [mV]', fontsize=25) #[K$^+$]$_o$ [mM]
cbar.ax.tick_params(labelsize=20)
# Function to update the plot for each frame
def update(frame_idx):
    frame = selected_frames[frame_idx]
    current_time = time_points[frame]
    
    # Use actual membrane potential values
    color_data = Vs[:, frame]
    
    # Update color data
    scat.set_array(color_data)
    
    # Update title
    title.set_text(f'Time: {current_time:.2f} s')
    
    # Update stimulation status
    # if stim_start_time <= current_time <= stim_end_time:
    #     stim_status.set_text("Glutamate Stimulation ON")
    # else:
    #     stim_status.set_text("")
    
    # Use fixed global color scale for all frames
    scat.set_clim(global_min, global_max)
    
    return scat, title, stim_status

# Create the animation
ani = FuncAnimation(fig, update, frames=len(selected_frames), 
                   blit=True, interval=200)  # interval in ms between frames

# Save as MP4
output_mp4 = "astrocyte_V_animation.mp4"
print(f"Saving animation as {output_mp4}...")
ani.save(output_mp4, writer='ffmpeg', fps=5, dpi=100)  # slower fps for better visualization

print("Animation saved successfully!")

# Show the animation in the notebook/IDE if desired
plt.tight_layout()
plt.show()
# %%





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the data
astrocyte = np.loadtxt("NMO_73320.txt")
df = pd.read_csv("G:/Alok/neuron_whole_sim/only_K_stim/comb_neuro_data_ast_67.csv")
xaxis, yaxis, zaxis, radius, identifier = astrocyte[:, 2], astrocyte[:, 3], astrocyte[:, 4], astrocyte[:, 5], astrocyte[:, 1]
time = df['Time'].values

# Extract K+ columns
Ko_columns = [col for col in df.columns if col.startswith('Kos_')]
Ko_data = df[Ko_columns].values.T  # Transpose so that rows correspond to compartments

# Calculate distance from soma for each compartment
N = len(xaxis)
distance_from_soma = np.zeros(N)
for i in range(N):
    distance_from_soma[i] = np.sqrt((xaxis[i] - xaxis[0])**2 + (yaxis[i] - yaxis[0])**2 + (zaxis[i] - zaxis[0])**2)

# Check the data range
max_ko = np.max(Ko_data)
print(f"Maximum K+ concentration: {max_ko} mM")

# Find where the maximum value occurs
max_idx = np.unravel_index(np.argmax(Ko_data), Ko_data.shape)
max_comp, max_time_idx = max_idx
max_time_val = time[max_time_idx]
max_ko_value = Ko_data[max_comp, max_time_idx]
print(f"Maximum K+ concentration ({max_ko_value:.2f} mM) occurs at row {max_comp}, time {max_time_val}s")

# Check if this compartment index exists in morphology data
if max_comp < len(distance_from_soma):
    print(f"This compartment is at distance: {distance_from_soma[max_comp]} µm from soma")
else:
    print(f"Warning: Compartment index {max_comp} is outside the range of morphology data (0-{len(distance_from_soma)-1})")
    print("This indicates a mismatch between K+ data and morphology data indexing")

# Find valid indices where we have both morphology and time series data
# Since there's a mismatch, we'll only use compartments that exist in both datasets
valid_indices = np.arange(min(N, Ko_data.shape[0]))
print(f"Using only the first {len(valid_indices)} compartments that exist in both datasets")

# Create a new array for K+ data that contains only valid indices
Ko_data_valid = Ko_data[:len(valid_indices)]

# Find the maximum in the valid data
max_ko_valid = np.max(Ko_data_valid)
max_idx_valid = np.unravel_index(np.argmax(Ko_data_valid), Ko_data_valid.shape)
max_comp_valid, max_time_idx_valid = max_idx_valid
print(f"Maximum K+ in valid range: {max_ko_valid:.2f} mM at comp {max_comp_valid}, distance {distance_from_soma[max_comp_valid]:.2f} µm")

# Create a DataFrame with distance and concentration data for better organization
data = []
for i in valid_indices:
    row = {'compartment_id': i, 'distance': distance_from_soma[i]}
    # Add time series data
    for t_idx, t in enumerate(time):
        if t_idx < Ko_data.shape[1]:
            row[f'time_{t:.2f}'] = Ko_data[i, t_idx]
    data.append(row)

# Create DataFrame from list of dictionaries
data_df = pd.DataFrame(data)

# Get time columns
time_columns = [col for col in data_df.columns if col.startswith('time_')]

# Check full distance range
print(f"Distance range: {data_df['distance'].min()} to {data_df['distance'].max()} µm")

# Option 1: Create heatmap with all compartments, properly sorted by distance
# Sort data by distance
data_df_sorted = data_df.sort_values('distance').reset_index(drop=True)

# Extract sorted arrays for plotting
sorted_distances = data_df_sorted['distance'].values
sorted_data = data_df_sorted[time_columns].values

# Set explicit color limits based on the actual data range in valid compartments
vmin = 3.0  # Baseline K+ level or min value
vmax = max_ko_valid * 1.1  # Maximum value with a bit of headroom

# Create the sorted heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(
    sorted_data, 
    aspect='auto', 
    extent=[time.min(), time.max(), sorted_distances.min(), sorted_distances.max()],
    origin='lower',
    cmap='jet',
    interpolation='nearest',
    vmin=vmin,
    vmax=vmax
)

# Add colorbar and labels
cbar = plt.colorbar(im)
cbar.set_label('[K$^+$]$_o$ [mM]')
plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title(f'Change in [K$^+$]$_o$ (All Compartments, Max: {max_ko_valid:.1f} mM)')

# Find the compartment with the maximum K+ concentration in valid range
plt.axhline(y=distance_from_soma[max_comp_valid], color='red', linestyle='-', alpha=0.5)
plt.text(time.max() + 0.1, distance_from_soma[max_comp_valid], f'Max K+ at Comp #{max_comp_valid}', 
        verticalalignment='center', fontsize=10, color='red')

# Highlight specific compartments of interest
stimulated_compartments = {
    1166: 13.95,
    1246: 155.87,
    1256: 177.28
}

for comp_id, target_dist in stimulated_compartments.items():
    if comp_id < len(valid_indices):
        # Try to find the actual distance for this compartment ID
        if comp_id in data_df['compartment_id'].values:
            comp_distance = data_df.loc[data_df['compartment_id'] == comp_id, 'distance'].values[0]
            plt.axhline(y=comp_distance, color='white', linestyle='--', alpha=0.5)
            plt.text(time.max() + 0.1, comp_distance, f'Comp #{comp_id}', 
                    verticalalignment='center', fontsize=8)
        else:
            # Find the closest match by distance
            closest_idx = np.abs(sorted_distances - target_dist).argmin()
            closest_dist = sorted_distances[closest_idx]
            plt.axhline(y=closest_dist, color='white', linestyle='--', alpha=0.5)
            plt.text(time.max() + 0.1, closest_dist, f'~Comp #{comp_id}', 
                    verticalalignment='center', fontsize=8)

plt.tight_layout()
plt.savefig('sorted_compartments_heatmap_fixed_scale.png', dpi=300, bbox_inches='tight')
plt.show()

# Option 2: Group by distance and create a more consolidated visualization
# Define a distance binning resolution (in μm)
distance_resolution = 5  # Adjust based on your data

# Create distance bins
data_df['distance_bin'] = np.round(data_df['distance'] / distance_resolution) * distance_resolution

# Group by distance bin and calculate mean K+ concentration
grouped_df = data_df.groupby('distance_bin').agg({
    'distance': 'mean',
    'compartment_id': 'count'  # Count compartments in each bin
})

# For each time point, calculate the mean of all compartments in that distance bin
for col in time_columns:
    grouped_df[col] = data_df.groupby('distance_bin')[col].mean()

# Reset index and sort by distance
grouped_df = grouped_df.reset_index().sort_values('distance')

# Extract arrays for plotting
binned_distances = grouped_df['distance'].values
binned_data = grouped_df[time_columns].values

# Check maximum in binned data
max_binned = np.max(binned_data)
print(f"Maximum K+ concentration in binned data: {max_binned} mM")

# Create the binned heatmap
plt.figure(figsize=(12, 8))
im = plt.imshow(
    binned_data, 
    aspect='auto', 
    extent=[time.min(), time.max(), binned_distances.min(), binned_distances.max()],
    origin='lower',
    cmap='jet',
    interpolation='nearest',
    vmin=vmin,
    vmax=vmax
)

# Add colorbar and labels
cbar = plt.colorbar(im)
cbar.set_label('[K$^+$]$_o$ [mM]')
plt.xlabel('Time [s]')
plt.ylabel('Distance from soma (µm)')
plt.title(f'Change in [K$^+$]$_o$ (Binned by {distance_resolution}μm, Max: {max_binned:.1f} mM)')

# Add text showing number of compartments in each bin
for i, (dist, count) in enumerate(zip(binned_distances, grouped_df['compartment_id'].values)):
    plt.text(time.max() + 0.2, dist, f"{count} comps", 
             verticalalignment='center', fontsize=8)

# Highlight bin with maximum K+
max_bin_idx = np.unravel_index(np.argmax(binned_data), binned_data.shape)
max_bin_row = max_bin_idx[0]
max_bin_dist = binned_distances[max_bin_row]
plt.axhline(y=max_bin_dist, color='red', linestyle='-', alpha=0.5)
plt.text(time.max() + 0.8, max_bin_dist, f'Max K+ bin', 
         verticalalignment='center', fontsize=10, color='red')

# Highlight stimulated distances
for comp_id, target_dist in stimulated_compartments.items():
    # Find the closest bin
    closest_bin_idx = np.abs(binned_distances - target_dist).argmin()
    closest_bin_dist = binned_distances[closest_bin_idx]
    
    plt.axhline(y=closest_bin_dist, color='white', linestyle='--', alpha=0.5)
    plt.text(time.max() + 0.8, closest_bin_dist, f'~Comp #{comp_id}', 
             verticalalignment='center', fontsize=8)

plt.tight_layout()
plt.savefig('binned_distance_heatmap_fixed_scale.png', dpi=300, bbox_inches='tight')
plt.show()

# Function to create a high-resolution interpolated visualization
def create_interpolated_heatmap(use_binned=True):
    """
    Creates a smooth interpolated heatmap with higher resolution.
    
    Parameters:
    use_binned : bool
        If True, uses the binned data; if False, uses the sorted data with all compartments
    """
    # Choose which dataset to use
    if use_binned:
        source_distances = binned_distances
        source_data = binned_data
        title_prefix = f"Interpolated [K$^+$]$_o$ (Binned, Max: {max_binned:.1f} mM)"
    else:
        source_distances = sorted_distances
        source_data = sorted_data
        title_prefix = f"Interpolated [K$^+$]$_o$ (All Compartments, Max: {max_ko_valid:.1f} mM)"
    
    # Extract time values from column names
    time_values = [float(col.split('_')[1]) for col in time_columns]
    
    # Create a regular grid with more points for smoother visualization
    grid_times = np.linspace(time_values[0], time_values[-1], 200)
    grid_distances = np.linspace(source_distances.min(), source_distances.max(), 300)
    
    # Create a meshgrid
    time_mesh, dist_mesh = np.meshgrid(grid_times, grid_distances)
    
    # Prepare points for interpolation
    points = []
    values = []
    
    for i, dist in enumerate(source_distances):
        for j, t in enumerate(time_values):
            if j < source_data.shape[1]:
                points.append([t, dist])
                values.append(source_data[i, j])
    
    # Interpolate values onto the regular grid
    grid_z = griddata(points, values, (time_mesh, dist_mesh), method='linear')
    
    # Plot the interpolated heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        grid_z,
        aspect='auto', 
        extent=[time_values[0], time_values[-1], grid_distances.min(), grid_distances.max()],
        origin='lower',
        cmap='jet',
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar and labels
    cbar = plt.colorbar(im)
    cbar.set_label('[K$^+$]$_o$ [mM]')
    plt.xlabel('Time [s]')
    plt.ylabel('Distance from soma (µm)')
    plt.title(title_prefix)
    
    # Mark important distances
    for comp_id, dist in stimulated_compartments.items():
        plt.axhline(y=dist, color='white', linestyle='--', alpha=0.5)
        plt.text(time_values[-1] + 0.1, dist, f'Comp #{comp_id}', 
                verticalalignment='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'interpolated_{"binned" if use_binned else "all"}_heatmap_fixed_scale.png', dpi=300, bbox_inches='tight')
    plt.show()

# Uncomment to create interpolated versions
# create_interpolated_heatmap(use_binned=True)  # Using binned data
# create_interpolated_heatmap(use_binned=False) # Using all compartments
# %%
