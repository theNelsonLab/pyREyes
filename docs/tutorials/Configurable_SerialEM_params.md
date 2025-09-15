# SerialEM Script Suites - User Configurable Parameters

> **IMPORTANT:** These script suites contain system-specific hardcoded values that MUST be adjusted for your microscope setup. 

These are SerialEM settings files containing individual macro scripts:

## Macro Scripts in Both Script Suites

### Main Workflow Scripts:
- **REyes_FSD** - Full Self Driving microED acquisition
- **REyes_Screen** - Autonomous diffractions mapping workflow 
- **REyes_Manual-1** - Montage for manual grid square selection
- **REyes_Manual-2_SD** - Complete session after manual grid square selection
- **REyes_Manual-2_Screen** - Diffraction mapping only after manual grid square selection

### Core Function Scripts:
- **REyes_MNT** - Montage acquisition
- **REyes_ECNT** - Eucentricity determination
- **REyes_STPSCN** - StepScan mapping
- **REyes_DFMV** - Diffraction Movie collection

### Helper Scripts:
- **REyes_postECNT** - Post-eucentricity processing
- **REyes_premanualECNT** - Pre-manual eucentricity setup
- **REyes_saveNAV** - Navigator file saving
- **ECNT_nav_ready** - Wait for eucentricity navigator files
- **ECNT_corrected** - Wait for eucentricity correction
- **Targets_nav_ready** - Wait for target navigator files
- **Manual_ready** - Wait for manual selection processing

---

> **NOTE ON LINE NUMBERS:** Line ~numbers mentioned in this document are approximate and may change if you modify the scripts by adding/removing lines or comments. Use the parameter names and context descriptions to locate the correct settings in your scripts.

Below are all parameters that typically require modification for different systems:

## Acquisition Parameters

### 1. Montage Settings (Macro: REyes_MNT)

#### CETA-D Script Suite - ALL MAGNIFICATION REFERENCES:
- **Variable definition:** `montage = _210LM_$mnt_sizex$mnt_size`
- **Section header:** `########TAKE INITIAL MONTAGE AT 210 LM########`
- **Start message:** `Echo Starting 210x LM montage collection`
- **Completion message:** `Echo Completed 210x LM montage data collection`

→ **Change ALL instances of "210" to your desired magnification**

#### DE Apollo Script Suite - ALL MAGNIFICATION REFERENCES:
- **Variable definition:** `montage = _115LM_$mnt_sizex$mnt_size`
- **Section header:** `########TAKE INITIAL MONTAGE AT 115 LM########`
- **Start message:** `Echo Starting 115x LM montage collection`
- **Completion message:** `Echo Completed 115x LM montage data collection`

→ **Change ALL instances of "115" to your desired magnification**

#### Camera Settings

**CETA-D Camera Settings (Lines ~470-472):**
```
SetContinuous M 0
SetExposure M 3        ← Optimized (adjust only if performance issues)
SetBinning M 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~470-472):**
```
SetContinuous M 0
SetExposure M 1        ← Optimized (adjust only if performance issues)
SetBinning M 4         ← ADJUST: Binning to achieve 2048x2048 images
```

### 2. Eucentricity Settings (Macro: REyes_ECNT)

**CETA-D Camera Settings (Lines ~594-596):**
```
SetContinuous V 0
SetExposure V 1.5      ← Optimized (adjust only if performance issues)
SetBinning V 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~594-597):**
```
SetContinuous V 0
SetExposure V 1        ← Optimized (adjust only if performance issues)
SetBinning V 4         ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime V 1       ← Optimized (adjust only if performance issues)
```

### 3. StepScan Settings (Macro: REyes_STPSCN)

**Grid Parameters (Lines ~686-687) - BOTH SCRIPT SUITES:**
```
increment_x = 5        ← ADJUST: Step size in X (microns)
increment_y = 5        ← ADJUST: Step size in Y (microns)
```

**CETA-D Camera Settings (Lines ~707-709):**
```
SetContinuous R 0
SetExposure R 1        ← Optimized (adjust only if performance issues)
SetBinning R 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Camera Settings (Lines ~707-710):**
```
SetContinuous R 0  
SetExposure R 0.25     ← Optimized (adjust only if performance issues)
SetBinning R 4         ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime R 0.25    ← Optimized (adjust only if performance issues)
```

### 4. Movie Acquisition Settings (Macro: REyes_DFMV)

#### Rotation Increment Reference Table
The `rotincr` parameter controls rotation steps and must match your rotation rate:

| rotincr Value | Rotation Rate (dps) | Use Case |
|---------------|--------------------|-----------| 
| 0.004         | 0.3 dps           | CETA-D default (compensates for frame buffering delays) |
| 0.010         | 0.3 dps           | Theoretical 0.3 dps |
| 0.020         | 0.67 dps          | Medium speed |
| 0.033         | 1.0 dps           | DE Apollo default |
| 0.066         | 2.0 dps           | High speed |

**Important Notes:**
- **DE Apollo**: Use table values that match your `rotrate` setting
- **CETA-D**: Adjust `rotincr` to achieve desired degree wedge per 3-second exposure
- **CETA-D Note**: The 0.3 dps is the apparent rotation rate; actual rotation rate is ~0.12 dps due to frame buffering delays

**CETA-D Rotation Settings (Lines ~871-873):**
```
rotrate = 0.3          ← ADJUST: Rotation rate (degrees per second)
rotincr = 0.004        ← Optimized for 3-second frame intervals (adjust only if changing rotation rate)
acqrate = 3            ← ADJUST: Acquisition rate (seconds per frame)
```

**DE Apollo Rotation Settings (Lines ~871-873):**
```
rotrate = 1            ← ADJUST: Rotation rate (degrees per second)  
rotincr = 0.033        ← Optimized for 1 dps (see rotation increment table above)
acqrate = 1            ← ADJUST: Acquisition rate (seconds per frame)
```

**DE Apollo Camera Length Conversion (Lines ~880-884):**
```
If $dist == 420
  apollodist = 560       ← ADJUST: Calibrate for your system
ElseIf $dist == 960
  apollodist = 1280      ← ADJUST: Calibrate for your system
```

**CETA-D Movie Settings (Lines ~896-898):**
```
SetContinuous R 1
SetExposure R 3        ← ADJUST: Movie exposure time
SetBinning R 2         ← ADJUST: Binning to achieve 2048x2048 images
```

**DE Apollo Movie Settings (Lines ~896-899):**
```
SetContinuous R 1
SetExposure R $recordtime    ← Calculated from rotation parameters
SetBinning R 4               ← ADJUST: Binning to achieve 2048x2048 images
SetFrameTime R $acqrate      ← Uses acquisition rate parameter
```

## File System Paths

### 1. DE Apollo Specific Paths

**DE Apollo Script (Lines ~753-754):**
```
RunInShell copy D:\DEOutput\Apollo\$samplename_$posname
```
→ Change `"D:\DEOutput\Apollo\"` to your DE camera output directory

**DE Apollo Script (Lines ~943-944):**
```
RunInShell copy D:\DEOutput\Apollo\$movdir
```
→ Change `"D:\DEOutput\Apollo\"` to your DE camera output directory

### 2. Navigator File Paths (Both Scripts)

**Lines ~518, 794, 998:**
- `grid_squares\eucentricity$nav_ext`
- `targets\targets$nav_ext`
- `grid_squares\grid_squares.nav`

> **IMPORTANT NOTE:** These navigator file paths are automatically created by the REyes Python package. If these files are missing when running the scripts, this indicates an issue with REyes itself, NOT a configuration problem with the SerialEM scripts. The scripts expect these files to be generated automatically by REyes during processing.

**Troubleshooting:** If navigator files are missing, check:
- REyes Python package installation and configuration
- REyes processing logs for errors
- File system permissions for REyes output directories

## Non-Adjustable Parameters

### Timing Parameters

#### 1. Processing Wait Times
**CETA-D Script (Line ~966, Macro: REyes_DFMV):**
```
Delay 900 sec          ← Optimized for REyes processing (do not adjust)
```

**DE Apollo Script (Line ~959, Macro: REyes_DFMV):**
```
Delay 600 sec          ← Optimized for REyes processing (do not adjust)
```

> **NOTE:** These wait times are calibrated for REyes package processing and should not be modified.

#### 2. Stage Movement Delays (Both Scripts)
**Various locations in multiple macros:**
```
Delay 1 sec            ← Optimized for system stability (do not adjust)
Delay 2 sec            ← Optimized for system stability (do not adjust)
```

> **NOTE:** These delays are optimized for reliable operation and should not need adjustment.

### Validation Ranges

**User Input Validation (Lines ~37-56, Main workflow macros) - BOTH SCRIPT SUITES:**
```
Montage size: 2-9      ← Tested range limits (do not adjust)
Map size: 3-21         ← Tested range limits (do not adjust)  
Tilt range: 5-70       ← Safe operational limits (do not adjust)
```

> **NOTE:** These validation ranges represent tested operational limits and should remain unchanged.

### Image Numbering

**Both Script Suites:**
```
imgnum = 10000 (Line ~681, Macro: REyes_STPSCN)     ← Helper parameter for REyes package (do not adjust)
movnum = 20000 (Line ~908, Macro: REyes_DFMV)       ← Helper parameter for REyes package (do not adjust)
```

> **NOTE:** These numbering parameters are used by the REyes package for file organization and tracking. They are not intended for user modification and should remain as set in the original scripts to maintain compatibility with REyes processing.

## System-Specific Calibration Notes

### 1. Camera Length Calibration
- DE Apollo users must calibrate the camera length conversion table
- CETA-D users should verify magnification values are correct for their system

### 2. File System Integration
- DE Apollo users must ensure DE software output paths are correct
- Both systems should verify directory creation permissions

### 3. Timing Optimization
- All exposure times should be optimized for your specific camera
- Stage movement delays may need adjustment based on your stage performance
- Processing wait times depend on your computational resources

### 4. Magnification Verification
- Verify that 210x (CETA-D) or 115x (DE Apollo) magnifications are appropriate
- **Note:** These values may vary instrument to instrument and should be checked even if the same exact camera is used
- Adjust based on your specific lens configuration and desired field of view

## Calibration Procedure Recommendations

1. Start with provided parameters as baseline
2. Test individual macros (`REyes_MNT`, `REyes_ECNT`, `REyes_STPSCN`, `REyes_DFMV`) separately
3. Optimize exposure times for your camera's performance
4. Verify file system paths and permissions
5. Test complete workflow with a practice sample
6. Document your optimized parameters for future use

> **Remember:** These are production scripts designed for specific microscope configurations. All "hardcoded" values are intentional and should be adjusted by an expert operator familiar with the microscope system.