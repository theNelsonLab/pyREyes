# REyes – Automated Electron Diffraction Software
## Installation Guide

REyes is a software suite that allows for fully autonomous microED (microcrystal electron diffraction) data acquisition and processing. REyes was designed to be executed on two computers: (i) acquisition PC and (ii) processing PC. Thus, installation consists of two parts:

- pyREyes package installation on the processing PC
- SerialEM configuration on the acquisition PC

## Prerequisites

For installation and operation to run smoothly:

- **Processing PC** should have Python installed. In this installation guide Python is installed through micromamba as an example. Other Python installations will suffice. Processing PC also requires a working installation of [XDS](https://xds.mr.mpg.de) for data reduction and [SHELX](https://shelx.uni-goettingen.de) for phasing. Please follow the installation instructions provided by each respective source. pyREyes has been most extensively tested using Linux and macOS systems, as well as Linux terminal on Windows (WSL). Cross-platform libraries used for the development will allow it to operate natively on Windows, however, XDS data reduction package requires a UNIX operating system.

- **Acquisition PC** should have an installed and calibrated SerialEM. Note that REyes has been developed and tested using SerialEM 4.1.

### Suggested Processing PC Configuration

**Hardware:**
- CPU: Intel Core i7-5820K @ 3.30GHz
- RAM: 16 GB (lower memory will result in slow down of grid atlas processing step)

**Operating System:**
- Preferred: Linux (latest Ubuntu 24 LTS)
- Optional: Windows 10 Version 1607 that can run WSL Ubuntu 24 LTS

**Apple Silicon Mac:**
- REyes has been tested on Mac with M3 and M4 Apple Silicon processors running latest macOS Sequoia 15

Follow this guide to install the REyes suite.

## Part 1: Processing PC

### Installation Steps

1. Download pyREyes package from the Nelson Lab GitHub repository: https://github.com/theNelsonLab/pyREyes

2. In terminal, UNIX or WSL, create a new Python environment by running and follow prompt instructions:

```bash
micromamba create -n reyes_env python=3.12 -c conda-forge
```

3. Change directory to the downloaded pyREyes package, activate the environment and install the package locally:

```bash
cd /your/path/to/downloaded/pyREyes-main
micromamba activate reyes_env
pip install .
```

### Verification

Test the installation by running:

```bash
python -c "import pyREyes; print('pyREyes installation successful')"
```

This step concludes installation of pyREyes. Note that AutoProcess (Python package `pyautoprocess`, https://github.com/theNelsonLab/autoprocess) is one of the dependencies and will be installed automatically (pyREyes 3.6.0 requires pyautoprocess 0.5.5 or later). AutoSolve is not yet public and can be obtained from the developers via email.

## Part 2: Acquisition PC

### Configure SerialEM Low Dose

Configure SerialEM Low Dose as follows:

- **Autosave log** is off
- **Eucentricity in Search** is off
- **Low Dose View** is LM magnification
  - E.g. 210x for Ceta (Talos Arctica)
  - E.g. 115x for DE Apollo (Talos Arctica); on a Talos F200C with DE Apollo use 155x
- **Low Dose Record** is DIFFRACTION under parallel beam condition
  - E.g. 960 mm with 50 um condenser aperture for Ceta (Talos Arctica)
  - E.g. 420 mm with 50 um condenser aperture for DE Apollo (Talos Arctica)

### Other Suggested Settings

- **Low Dose Focus** is SA magnification ca. 22000x
- **Low Dose Search** is SA magnification ca. 4300x under parallel beam condition identical to Record mode

### Script Configuration

Two script packages are provided in `docs/scripts/` of the pyREyes repository:

- `SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt` for a Talos Arctica with a Ceta camera
- `SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt` for a Talos Arctica with a DE Apollo camera

Both packages are set up for a Talos Arctica. On a Talos F200C with a DE Apollo camera, use the DE Apollo package, change the montage magnification in `REyes_MNT` (and the montage name used on resume in `Automated_Pipeline_Helper`) from 115x to 155x LM (see [Configurable_SerialEM_params.md](Configurable_SerialEM_params.md), section 1), and start the monitor with `--microscope F200C-Apollo`.

Depending on the available camera, copy one of them to the acquisition PC and point `ScriptPackagePath` in SerialEMsettings.txt to it. The line to be changed is line 3 of the SerialEMsettings.txt file.

```
SerialEMSettings
SystemPath     C:\ProgramData\SerialEM\
ScriptPackagePath      C:\ProgramData\SerialEM\Scripts\SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt
```

Start a REyes session by running the `START_REYES` script, and batched collection on navigator points by running `BatchED_Main` (`BatchED_Main_CETA` in the Ceta package). The system-specific values to check in the scripts are listed in [Configurable_SerialEM_params.md](Configurable_SerialEM_params.md), and the differences between the two packages in `docs/scripts/README.md`.

The DE Apollo and Ceta packages shipped with pyREyes 3.6.0 can resume an interrupted automated run from any stage (`START_REYES` asks whether the run is new) and set aside a grid square or crystal that fails instead of stopping the grid. They require pyREyes 3.6.0 or later on the processing PC. See `docs/scripts/README.md`.

> **NOTE:** The User Guide PDF (`docs/tutorials/User_guide.pdf`) shows the SerialEM script workflow of REyes 3.4.0. In the current scripts the entry points are `START_REYES` and `BatchED_Main`; see `docs/scripts/README.md`.

This step concludes configuration of SerialEM for REyes. Note that minor adjustments of microscope settings and SerialEM scripts might be required to tailor a particular system.