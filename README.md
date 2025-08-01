# Astrocyte-Ion-Dynamics Model

## Overview  
This repository contains Python scripts for simulating ionic homeostasis in an astrocyte.  
Running **`ast_modeling.py`** produces a CSV named **`comb_astro_data.csv`** that records:

* Membrane potential ( *V* )  
* Intracellular & extracellular ion concentrations (K⁺, Na⁺, Cl⁻, Ca²⁺, …)  
* Currents and fluxes through pumps, channels, transporters, and cotransporters  

A sample output is provided in **`sample_comb_astro_data.csv`** for reference.  
**'utils.py'** includes the Euler solver that integrates the differential equations.
Utility functions in **`plot_utils.py`** help visualise any of the generated traces.

---

## Quick Start

```bash
# 1. Create the exact Python version the model was validated on
conda create -n astro38 python=3.8.8 -y
conda activate astro38

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model (≈5 min for the default 30 s simulation)
python ast_modeling.py

Output – comb_astro_data.csv appears in the working directory.

## Simulation Details

```
| Parameter          | Default value | Notes                                                                                                                      |
|--------------------|---------------|----------------------------------------------------------------------------------------------------------------------------|
| Total duration | 0 – 30 s      | Adjustable in `ast_modeling.py` (see comments within the script)                                                               |
| K⁺ stimulation | 22 s – 23 s   | Adjustable Amplitude & timing (see comments within the script)                                                                           |
| Morphology file| `NMO_73320.txt` | Contains multi-compartment astrocyte geometry; current script uses a single-compartment with “whole-cell” model parameters |

