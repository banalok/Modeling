# Astrocyte-Ion-Dynamics Model

## Overview  
This repository contains Python scripts for simulating ionic homeostasis in an astrocyte.  
Running **`ast_modeling.py`** produces a CSV named **`comb_astro_data.csv`** that records:

* Membrane potential ( *V* )  
* Intracellular & extracellular ion concentrations (K⁺, Na⁺, Cl⁻, Ca²⁺, …)  
* Currents and fluxes through pumps, channels, transporters, and cotransporters  

The code supports both point and spatiotemporal modeling, depending on the specified number of compartments to simulate (N). For example, if N = 10, the simulated compartments will range from 0 (N1) to 9 (N2), and these indices should be adjusted accordingly within the code.

Along with the main data file, the simulation generates additional three CSV output files:

- **`distance_from_soma.csv`** – Contains the distance of each compartment from the soma  
- **`branch_to_comps.csv`** – Lists compartment indices corresponding to each branch  
- **`branch_to_max_distance.csv`** – Provides the maximum soma distance for each branch  

For a **single-compartment (point) model**, these files will be empty.

## Sample Output

A sample output is included in **`sample_data.zip`** for reference.  
The file **`comb_astro_data_highKFull.csv`** represents a complete spatiotemporal simulation under global high K⁺ stimulation using the provided morphology.

## Utility Scripts

**'utils.py'** includes the Euler solver that integrates the differential equations.  
Scripts inside **`plot_utils.py`** help to visualise generated data.

---

## Quick Start

```bash
# 1. Create the exact Python version the model was validated on
conda create -n astro38 python=3.8.8 -y
conda activate astro38

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model (≈5 min for default 30 s simulation for a single compartment)
python ast_modeling.py

Output – comb_astro_data.csv appears in the working directory along with other csv files as mentioned above.

```
## Helpful Details

| Parameter          | Default value | Notes                                                                                                                      |
|--------------------|---------------|----------------------------------------------------------------------------------------------------------------------------|
| Total duration | 0 – 30 s      | Adjustable in `ast_modeling.py` (see comments within the script)                                                               |
| K⁺ stimulation | 22 s – 23 s   | Adjustable Amplitude & timing (see comments within the script)                                                                 |
| Morphology file| `NMO_282188.txt` | Contains multi-compartment astrocyte geometry                                                                               |

