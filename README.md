# Diffraction Map Processing Pipeline (REyes v3.6.0)
[![License](https://img.shields.io/badge/License-Academic_Use_Only-blue.svg)](https://github.com/theNelsonLab/pyREyes/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16971798.svg)](https://doi.org/10.5281/zenodo.16971798)

## Quick Start
```bash
# Install the package
pip install .

# Run automated pipeline
reyes-monitor --microscope Arctica-CETA

# Run automated pipeline with movie processing (requires a pyautoprocess preset)
reyes-monitor --microscope Arctica-CETA --autoprocess --microscope-config Arctica-CETA-mrc-SM
```

## Overview
pyREyes is a Python package to support REyes (Reciprocal Eyes), an end-to-end autonomous electron diffraction data collection and processing pipeline. REyes generates diffraction heatmaps, identifies key targets, and manages navigator files for SerialEM.

📚 [Full Documentation](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/README_FULL.md)

📝 [Changelog](https://github.com/theNelsonLab/pyREyes/blob/main/CHANGELOG.md) - what is new in 3.6.0 and how to upgrade from 3.4.x

## Key Features
- Grid squares detection and atlas management
- Manual or automated grid square selection
- Eucentricity correction
- Diffraction pattern quality classification using DQI (Diffraction Quality Index)
- Frame loading for different detectors: stacks are summed, large frames are cleaned of bad lines and binned to 2048x2048, and small frames use dedicated detection parameters
- Diffraction map blocks recorded across midnight are processed as one block
- Navigation file generation with spatial awareness
- Comprehensive visualization suite
- Automated data processing with state machine monitoring
- Optional movie processing and structure solution integration
- Restart-safe movie processing and timeouts for hung processing steps
- Keeps going when SerialEM sets a grid square or crystal aside or aborts a grid, and records the session outcome in `session_status.json`
- SerialEM script packages for Ceta and DE Apollo cameras ([docs/scripts](https://github.com/theNelsonLab/pyREyes/tree/main/docs/scripts))

## Diffraction Quality Analysis

REyes uses advanced quality metrics to assess diffraction patterns:

**LQP (Lattice Quality Peaks)**: A number of peaks in an FFT of the thresholded and binarized diffraction snapshot that correspond to lattice quality.

**DQI (Diffraction Quality Index)**: A quality metric calculated as the ratio of LQP to total diffraction peaks (DQI = LQP / Diffraction Peaks). DQI values above 3.0 indicate higher crystalline quality with better-ordered lattice structures. This index is used to automatically classify and prioritize crystal targets for data collection.

## Pipeline Components

### Core Modules
1. [Grid Squares Searcher](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/grid-squares.md)
   - 1.1. [Manual Grid Squares Adder](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/manual-squares.md)
2. [Navigator Eucentricity Corrector](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/eucentricity.md)
3. [Diffraction Map Processor](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/diffraction-map.md)
4. [Navigation File Generator](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/navigation.md)
   - 4.1. [Selected Targets Adder](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/targets.md)
5. [Final Targets Processor](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/final-targets.md)
6. [Collection Executor](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/map-plotter.md)

**Pipeline Orchestration:**
- [Processing Monitor](https://github.com/theNelsonLab/pyREyes/blob/main/src/pyREyes/docs/modules/monitor.md) - Automatically runs all steps above and pauses for user input on optional manual steps

## Directory Structure
```
working_directory/
├── dif_maps/               # Diffraction maps and visualizations
│   └── diff_blocks_maps/   # Block-specific maps
├── grid_squares/           # Grid square & eucentricity outputs
├── movies/                 # Continuous rotation diffraction movies
├── REyes_logs/             # Processing logs
└── targets/                # Navigation files and diffraction snapshots
```

## Reference
The REyes methodology and its applications to small molecules, materials, and protein crystallography are described in detail in:
  
Eremin, D.B.; Jha, K.K.; Delgadillo, D.A.; Zhang, H.; Foxman, S.H.; Johnson, S.N.; Vlahakis, N.W.; Cascio, D.; Lavallo, V.; Rodríguez, J.A.; Nelson, H.M. **"Spatially-Aware Diffraction Mapping Enables Fully Autonomous MicroED"** *ChemRxiv* **2025**, DOI: [10.26434/chemrxiv-2025-4p4c3](https://doi.org/10.26434/chemrxiv-2025-4p4c3)

## Data Availability
Electron diffraction data are publicly available through the Caltech.

**Repository:** https://miledd.caltech.edu/shared/

**ZENODO Dataset:** Complete Raw and Processed REyes Dataset: Autonomous MicroED Collection for (S,S)-Salen Ligand - https://doi.org/10.5281/zenodo.16971798  

## Support
For support and questions, please contact:
- 📫 Dmitry Eremin - eremin@caltech.edu

## Developers
- [Dmitry Eremin (mit-eremin)](https://github.com/mit-eremin)
- [Hongyu Zhang (Eta2Zeta)](https://github.com/Eta2Zeta)

## Patent Disclosure
Dmitry B. Eremin, Hongyu Zhang, Jose A. Rodríguez, and Hosea M. Nelson have filed a provisional patent application related to the methods implemented in this software:

**Patent Application:** CIT-9295-P2, filed May 8, 2025  

This disclosure is provided in accordance with academic transparency standards. The software remains available for academic and research use under the terms specified in the license.

Copyright © 2025, California Institute of Technology (Caltech). All rights reserved.
Use of this software is permitted for academic and non‑commercial research only. Commercial use is prohibited without a license, please contact Caltech Office of Technology Transfer & Corporate Partnerships.