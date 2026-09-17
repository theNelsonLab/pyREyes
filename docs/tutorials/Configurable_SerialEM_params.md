# SerialEM Script Suites - User Configurable Parameters

> **IMPORTANT:** These script suites contain system-specific hardcoded values that MUST be adjusted for your microscope setup.

Two SerialEM script packages are provided in `docs/scripts/`:

- `SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt` - **Ceta suite** (Talos Arctica with Ceta camera)
- `SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt` - **DE Apollo suite** (Talos Arctica with DE Apollo camera)

See `docs/scripts/README.md` for the differences between the two suites.

> **NOTE:** The User Guide PDF (`docs/tutorials/User_guide.pdf`) shows the SerialEM script workflow of REyes 3.4.0. In the current scripts the entry points are `START_REYES` and `BatchED_Main`; see `docs/scripts/README.md`.

## Macro Scripts in Both Script Suites

### Entry Points (run these):
- **START_REYES** - REyes session for one grid on the stage or several grids from the autoloader. It asks for:
  - acquisition mode: single grid or multi-grid
  - workflow: automated, or manual (pause for manual grid square selection)
  - pipeline mode:
    - **FSD** (Full Self Driving): montage, eucentricity, StepScan, targets, rotation movies
    - **Screen**: montage, eucentricity, StepScan, no movies
    - **Montage**: grid overview only (automated workflow)
- **BatchED_Main** (Ceta suite: **BatchED_Main_CETA**) - Batch collection on navigator points added by the operator: continuous rotation, still snapshots, multi-angle snapshots (+60, +30, 0, -30, -60 degrees), or Search-mode images
- **REYES_Setup_Standalone** - Sets sample name, directory, montage/map size and tilt range so that `REyes_MNT`, `REyes_STPSCN` or `REyes_DFMV` can be run on their own
- **AutoloaderScreen** - Loads each occupied autoloader slot in a fixed range and acquires a montage map of it

### Pipeline Helper Scripts (called by START_REYES):
- **Setup_Helper** - Collects all session parameters
- **Slot_Verification_Helper** - Checks that the selected autoloader slots contain grids (multi-grid)
- **Directory_Setup_Helper** - Creates the `<sample>-reyes` session folders
- **Automated_Pipeline_Helper** - Runs the automated FSD / Screen / Montage pipeline for one grid
- **Manual_Pipeline_Part1** - Montage before manual grid square selection
- **Manual_Pipeline_Part2** - Rest of the pipeline after manual grid square selection
- **REyes_Setup_Automated** - Prompts the operator to start REyes in every sample folder
- **REyes_Setup_Optional** - Montage mode notice (REyes not required)

### Batched Collection Scripts (called by BatchED_Main):
- **BatchED_Rotation** - Continuous rotation diffraction (Record)
- **BatchED_Still** - Still diffraction snapshots (Record)
- **BatchED_MultiAngle** - Still snapshots at five tilt angles (Record)
- **BatchED_Imaging** - Search-mode images

In the Ceta suite these carry a `_CETA` suffix (e.g. `BatchED_Rotation_CETA`).

### Core Function Scripts:
- **REyes_MNT** - Montage acquisition
- **REyes_ECNT** - Eucentricity determination
- **REyes_STPSCN** - StepScan mapping
- **REyes_DFMV** - Diffraction Movie collection
- **REyes_SpotGuard** (DE Apollo suite only) - Checks that Low Dose is on and that the spot size in the Record area matches the expected value before StepScan and movie collection

### Helper Scripts:
- **REyes_postECNT** - Post-eucentricity processing
- **REyes_premanualECNT** - Pre-manual eucentricity setup
- **REyes_saveNAV** - Navigator file saving
- **ECNT_nav_ready** - Wait for eucentricity navigator files
- **ECNT_corrected** - Wait for eucentricity correction
- **Targets_nav_ready** - Wait for target navigator files
- **Manual_ready** - Wait for manual selection processing

---

> **NOTE ON LINE NUMBERS:** Line ~numbers mentioned in this document refer to the script package files as shipped and change if you add or remove lines. Use the macro names, parameter names and context descriptions to locate the correct settings in your scripts.

Below are all parameters that typically require modification for different systems:

## Acquisition Parameters

### 1. Montage Settings (Macro: REyes_MNT)

#### Ceta Script Suite - ALL MAGNIFICATION REFERENCES:
- **Variable definition (Line ~2244):** `montage = _210LM_$mnt_sizex$mnt_size`
- **Section header (Line ~2299):** `########TAKE INITIAL MONTAGE AT 210 LM########`
- **Start message (Line ~2317):** `Echo Starting 210x LM montage collection`
- **Completion message (Line ~2341):** `Echo Completed 210x LM montage data collection`

→ **Change ALL instances of "210" to your desired magnification**

#### DE Apollo Script Suite - ALL MAGNIFICATION REFERENCES:
- **Variable definition (Line ~2405):** `montage = _115LM_$mnt_sizex$mnt_size`
- **Section header (Line ~2460):** `########TAKE INITIAL MONTAGE AT 115 LM########`
- **Start message (Line ~2478):** `Echo Starting 115x LM montage collection`
- **Completion message (Line ~2502):** `Echo Completed 115x LM montage data collection`

→ **Change ALL instances of "115" to your desired magnification**

> **Talos F200C:** the DE Apollo suite is set up for a Talos Arctica (115x LM). On a Talos F200C use 155x LM: change all four "115" references above to "155", and set the Low Dose View magnification to 155x.

> **NOTE:** REyes finds the montage log from the `<number>LM` part of the file name and waits for the `Completed <number>x LM montage data collection` message, so the number must be the same in the variable definition and the completion message.

#### Camera Settings

**Ceta Camera Settings (Lines ~2329-2331):**
```
SetContinuous M 0
SetExposure M 3        ← Optimized (adjust only if performance issues)
SetBinning M 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~2490-2492):**
```
SetContinuous M 0
SetExposure M 1        ← Optimized (adjust only if performance issues)
SetBinning M 4         ← ADJUST: Binning to achieve 2048x2048 images
```

### 2. Eucentricity Settings (Macro: REyes_ECNT)

**Ceta Camera Settings (Lines ~2463-2465):**
```
SetContinuous V 0
SetExposure V 1.5      ← Optimized (adjust only if performance issues)
SetBinning V 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~2614-2617):**
```
SetContinuous V 0
SetExposure V 1        ← Optimized (adjust only if performance issues)
SetBinning V 4         ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime V 1       ← Optimized (adjust only if performance issues)
```

### 3. StepScan Settings (Macro: REyes_STPSCN)

**Grid Parameters - BOTH SCRIPT SUITES (Ceta Lines ~2565-2566, DE Apollo Lines ~2706-2707):**
```
increment_x = 5        ← ADJUST: Step size in X (microns)
increment_y = 5        ← ADJUST: Step size in Y (microns)
```

**Ceta Camera Settings (Lines ~2586-2588):**
```
SetContinuous R 0
SetExposure R 1        ← Optimized (adjust only if performance issues)
SetBinning R 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~2749-2757):**
```
SetContinuous R 0
SetExposure R 0.25     ← Keep equal to the frame time (one frame per image)
SetBinning R 4         ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime R 0.25    ← Keep equal to the exposure
```

> **NOTE:** Do not insert commands (including `Echo`) between the file report and `ReportStageXYZ` in the StepScan loop. REyes reads stage coordinates a fixed number of log lines after each file name (`serialEM_log` in `microscope_configs.json`).

### 4. Movie Acquisition Settings (Macro: REyes_DFMV)

#### Rotation Increment Reference Table
The `rotincr` parameter controls rotation steps and must match your rotation rate:

| rotincr Value | Rotation Rate (dps) | Use Case |
|---------------|--------------------|-----------|
| 0.004         | 0.3 dps           | Ceta default (compensates for frame buffering delays) |
| 0.010         | 0.3 dps           | Theoretical 0.3 dps |
| 0.020         | 0.67 dps          | Medium speed |
| 0.033         | 1.0 dps           | DE Apollo default |
| 0.066         | 2.0 dps           | High speed |

**Important Notes:**
- **DE Apollo**: `rotincr` is calculated as `0.033 * rotrate`
- **Ceta**: Adjust `rotincr` to achieve desired degree wedge per 3-second exposure
- **Ceta Note**: The 0.3 dps is the apparent rotation rate; actual rotation rate is ~0.12 dps due to frame buffering delays

**Ceta Rotation Settings (Lines ~2764-2766):**
```
rotrate := 0.3         ← ADJUST: Rotation rate (degrees per second)
rotincr := 0.004       ← Optimized for 3-second frame intervals (adjust only if changing rotation rate)
acqrate := 0.33        ← Used in the movie folder name
```

**DE Apollo Rotation Settings:**
The rotation rate and frame time are asked by `START_REYES` (Setup_Helper, Lines ~1255-1265; defaults 1 deg/s and 1 s per frame). When `REyes_DFMV` is run on its own and they are not yet set, it asks for them (Lines ~3045-3051). Keep rotation rate x frame time at 1 degree per frame; `REyes_DFMV` warns otherwise. The rotation increment is calculated (Line ~3063):
```
rotincr := 0.033 * $rotrate
```

**DE Apollo Camera Length Conversion (Lines ~3084-3087):**
```
If $dist == 420
   apollodist := 560     ← ADJUST: Calibrate for your system
ElseIf $dist == 960
   apollodist := 1280    ← ADJUST: Calibrate for your system
```
Other camera lengths fall back to the nominal value (with a warning in the log). The same table is used by `BatchED_Main` (Lines ~735-738).

**Ceta Movie Settings (Lines ~2788-2790):**
```
SetContinuous R 1
SetExposure R 3        ← ADJUST: Movie exposure time
SetBinning R 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Movie Settings (Lines ~3108-3111):**
```
SetContinuous R 1
SetExposure R $recordtime    ← Calculated from rotation parameters
SetBinning R 4               ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime R $acqrate      ← Uses the frame time parameter
```

### 5. Spot Size Check (DE Apollo suite only)

`START_REYES` asks for the expected diffraction spot size (Setup_Helper, Line ~1278, default 8) and what to do on a mismatch (Line ~1288: 1 = abort the grid, 2 = warn and continue). `REyes_SpotGuard` checks it after switching to the Low Dose Record area in `REyes_STPSCN` and `REyes_DFMV`. The guard never changes the spot size. Low Dose must be on; otherwise StepScan and movie collection stop for that grid.

→ **Set the default in Setup_Helper to the spot size you use for diffraction**

### 6. Batched Collection (BatchED_Main)

All batched parameters (frame time, binning, tilt range, rotation rate, sweeps) are asked at run time. System-specific values:

**Ceta (BatchED_Main_CETA, Lines ~645 and ~683-686; BatchED_Rotation_CETA, Lines ~1794-1797):**
```
rotincr := 0.004 * ($rotrate / 0.3)   ← Calibrated for frame-saving overhead
If $dist == 840
   cetadist := 790                    ← ADJUST: Calibrate for your system
```

**DE Apollo (BatchED_Main, Lines ~698 and ~735-738):**
```
rotincr := $rotrate / 30
If $dist == 420
   apollodist := 560                  ← ADJUST: Calibrate for your system
```

## File System Paths

### 1. DE Apollo Specific Paths

Frames are written by the DE server and copied into the session folder:

**REyes_STPSCN (Line ~2827):**
```
RunInShell copy D:\DEOutput\Apollo\$samplename_$posname $set_directory\$samplename_$posname
```

**REyes_DFMV (Line ~3191):**
```
RunInShell copy D:\DEOutput\Apollo\$movdir $set_directory\movies\$movdir
```

**BatchED_Rotation, BatchED_Still, BatchED_MultiAngle, BatchED_Imaging (Lines ~2086, ~2169, ~2269, ~2340):**
```
RunInShell copy D:\DEOutput\Apollo\...
```
→ Change `D:\DEOutput\Apollo\` to your DE camera output directory in all six places

### 2. Navigator File Paths (Both Scripts)

**Macros REyes_ECNT, REyes_DFMV, REyes_postECNT, ECNT_nav_ready, Targets_nav_ready:**
- `grid_squares\eucentricity$nav_ext`
- `targets\targets$nav_ext`
- `grid_squares\grid_squares.nav`

> **IMPORTANT NOTE:** These navigator file paths are automatically created by the REyes Python package. If these files are missing when running the scripts, this indicates an issue with REyes itself, NOT a configuration problem with the SerialEM scripts. The scripts expect these files to be generated automatically by REyes during processing.

**Troubleshooting:** If navigator files are missing, check:
- REyes Python package installation and configuration
- REyes processing logs for errors
- File system permissions for REyes output directories

### 3. Autoloader Slot Range (Macro: AutoloaderScreen)

**Ceta (Lines ~2184-2185, default slots 3-6) / DE Apollo (Lines ~2355-2356, default slots 2-5):**
```
MinSlotNum := 3        ← ADJUST: first slot to image
MaxSlotNum := 6        ← ADJUST: last slot to image
```
(`START_REYES` asks for its slot range at run time.)

## Non-Adjustable Parameters

### Timing Parameters

#### 1. Processing Wait Times (Macro: REyes_DFMV)
**Ceta Script (Line ~2846, after each movie; Line ~2866, after the last movie):**
```
Delay 120 sec
Delay 900 sec          ← Optimized for REyes processing (do not adjust)
```

**DE Apollo Script (Lines ~3058-3059):**
```
dfmv_write_settle = 5  ← Wait after each movie before copying frames (raise if a movie folder is copied short)
dfmv_tail_settle = 20  ← Wait after the last movie
```

#### 2. Stage Movement Delays (Both Scripts)
**Various locations in multiple macros:**
```
Delay 1 sec            ← Optimized for system stability (do not adjust)
Delay 2 sec            ← Optimized for system stability (do not adjust)
```

> **NOTE:** These delays are optimized for reliable operation and should not need adjustment.

### Validation Ranges

**User Input Validation (Macro: Setup_Helper) - BOTH SCRIPT SUITES:**
```
Montage size: 2-9      ← Tested range limits (do not adjust)
Map size: 3-21         ← Tested range limits (do not adjust)
Tilt range: 5-70       ← Safe operational limits (do not adjust)
```

> **NOTE:** These validation ranges represent tested operational limits and should remain unchanged.

### Image Numbering

**Both Script Suites:**
```
imgnum = 10000 (Macro: REyes_STPSCN)     ← Helper parameter for REyes package (do not adjust)
movnum = 20000 (Macro: REyes_DFMV, Ceta) ← Helper parameter for REyes package (do not adjust)
```

> **NOTE:** These numbering parameters are used by the REyes package for file organization and tracking. They are not intended for user modification and should remain as set in the original scripts to maintain compatibility with REyes processing.

### SerialEM Variable Types

Each variable is assigned either with `:=` (persistent, kept between script runs) or with `=` (regular) everywhere in a package. SerialEM stops a script if an existing variable is assigned the other way, so keep the operator that is already used when you edit a line. `START_REYES` and `BatchED_Main` clear persistent variables when they start.

## System-Specific Calibration Notes

### 1. Camera Length Calibration
- DE Apollo users must calibrate the camera length conversion table (`REyes_DFMV` and `BatchED_Main`)
- Ceta users should verify magnification values and the `cetadist` conversion in the batched scripts

### 2. File System Integration
- DE Apollo users must ensure DE software output paths are correct
- Both systems should verify directory creation permissions

### 3. Timing Optimization
- All exposure times should be optimized for your specific camera
- Stage movement delays may need adjustment based on your stage performance
- Processing wait times depend on your computational resources

### 4. Magnification Verification
- Verify that 210x (Ceta, Talos Arctica) or 115x (DE Apollo, Talos Arctica) magnifications are appropriate; a Talos F200C uses 155x
- **Note:** These values may vary instrument to instrument and should be checked even if the same exact camera is used
- Adjust based on your specific lens configuration and desired field of view

## Calibration Procedure Recommendations

1. Start with provided parameters as baseline
2. Run `REYES_Setup_Standalone`, then test individual macros (`REyes_MNT`, `REyes_ECNT`, `REyes_STPSCN`, `REyes_DFMV`) separately
3. Optimize exposure times for your camera's performance
4. Verify file system paths and permissions
5. Test the complete workflow with `START_REYES` (single grid) on a practice sample
6. Document your optimized parameters for future use

> **Remember:** These are production scripts designed for specific microscope configurations. All "hardcoded" values are intentional and should be adjusted by an expert operator familiar with the microscope system.
