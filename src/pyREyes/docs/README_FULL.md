# REyes (Reciprocal Eyes)
## End-to-End Autonomous Electron Diffraction

**pyREyes v3.4** - Automated MicroED diffraction data collection and analysis package

This repository contains a series of scripts designed to process diffraction data, generate heatmaps, identify key targets, and manage navigation files for SerialEM. The scripts are structured to be executed in the order outlined below, ensuring that new blocks of data are processed systematically while maintaining data integrity.

## Prerequisites

Ensure you have the following Python packages installed:
- `numpy`
- `pandas`
- `matplotlib`
- `hyperspy`
- `scipy`
- `scikit-image`
- `scikit-learn`
- `shapely`
- `psutil`

You can install the package with pip after downloading the code:
```bash
pip install .
```

## Code Overview

### 1. REyes Grid Squares Searcher (GSS)

**Workflow Position: Step 1**

This script processes diffraction data from **grid squares**, enabling automated detection and navigation generation based on montage data collected from supported microscopes. It is the first step in the REyes automated MicroED workflow.

#### Key Features:

* **Automated File Detection**: Automatically detects `.mrc` and `.mdoc` montage files in subdirectories
* **Multi-Microscope Support**: Configurable presets optimized for different microscope/camera combinations
* **Intelligent Centroid Detection**: Advanced image processing with local maxima detection and filtering
* **High-Quality Visualization**: Combined stitched montage images with overlaid centroids
* **SerialEM Integration**: Compatible navigation file generation for workflow integration

#### Key Outputs:

* **Montage Plot** (`grid_squares/combined_grid_montage.png`): Combined stitched image of the montage with all detected grid square centroids overlaid
* **Eucentricity Plot** (`grid_squares/eucentricity_points.png`): Subset plot showing the selected grid squares used for eucentric height calibration
* **Navigation Files**:
  * `grid_squares/grid_squares.nav`: Navigation file containing all detected grid square targets
  * `grid_squares/eucentricity.nav`: Navigation file with selected squares for eucentric height calibration
* **Log File**:
  * `grid_squares/grid_squares.log`: Detailed text log of processing steps, inputs, and configuration settings

#### Key Processing Steps:

1. **File Discovery and Validation**: Searches and matches `.mrc` with corresponding `.mdoc` metadata file
2. **Stage Coordinate Processing**: Extracts metadata and calculates rotation angle
3. **Image Processing and Alignment**: Loads montage frames, applies intensity clipping, and constructs stitched montage based on stage coordinates
4. **Centroid Detection**: Creates binary mask, removes edge-touching regions, applies size filtering, merges nearby small regions, and extracts centroid coordinates
5. **Filtering and Selection**: Applies user-specified filtering and selects subset for eucentricity
6. **Output Generation**: Generates SerialEM-compatible navigation files and final montage images

#### Usage:
```bash
grid-squares-0 [options]
```

Options:
* `--microscope`: One of the supported microscope configurations (default: `Arctica-CETA`)
* `--filtering`: Target filtering option (default: `default`)
  * Standard options: `default`, `None`
  * Target count options: `1`, `4` (2×2), `9` (3×3), `16` (4×4), `25` (5×5), `36` (6×6), `49` (7×7), `64` (8×8), `81` (9×9), `100` (10×10), `121` (11×11), `144` (12×12), `169` (13×13), `196` (14×14)
  * Special options: `96` (8×12 grid)
* `--debug`: Enable debug mode for additional diagnostic outputs

Example:
```bash
grid-squares-0 --microscope Arctica-Apollo-Prot --filtering 16
```

---

### 2. REyes Manual Grid Squares Adder (MGSA)

**Workflow Position: Step 2 (Optional)**

The Manual Grid Squares Adder processes SerialEM navigation files containing manually added grid squares' points, converting them into the standardized format required for the REyes automated collection pipeline. This tool allows to override automated detection with custom grid square selections.

#### Key Features:
- **Automatic NAV File Detection**: Finds and validates SerialEM navigation files
- **Manual Selection Processing**: Extracts user-selected grid squares
- **Visualization Generation**: Creates overlay plots showing manual selections on the original montage

#### Prerequisites:
1. Run `grid-squares-0` first to generate initial montage data
2. Manually select grid squares in SerialEM and save `.nav` navigator file

#### Key Outputs:
- **`grid_squares.nav`**: SerialEM format of the processed navigation file
- **`manual_grid_overlay.png`**: Montage image with manual selections overlaid
- **`REyes_logs/manual_squares.log`**: Detailed processing log

#### Usage:
```bash
manual-squares-0-1
```

No command line arguments are required. The tool automatically detects REyes collected `.nav` file in the current directory and processes the selections.

---

### 3. REyes Navigator Eucentricity Corrector (NEC)

**Workflow Position: Step 3**

This script processes electron microscope stage positions and automatically corrects eucentricity tilt by fitting a plane to the stage positions and adjusting Z coordinates accordingly.

#### Key Features:
- **Automatic File Detection**: Finds SerialEM log and navigation files
- **Plane Fitting**: Linear regression to determine optimal eucentricity plane
- **Coordinate Correction**: Updates Z positions while preserving X,Y coordinates
- **Comprehensive Visualization**: 6-panel plot showing correction results

#### Required Inputs: 
- SerialEM log file ending with `_grid_squares.log` in the current directory
- `grid_squares/grid_squares.nav` file

#### Key Outputs:
- **Updated Navigation File**: Modified `grid_squares/grid_squares.nav` with corrected Z coordinates
- **Visualization**: `grid_squares/eucentricity_correction.png` showing:
  - Original 3D grid square positions
  - Corrected grid square positions
  - Fitted eucentricity plane
  - X and Y direction projections
- **Log File**: Detailed processing log in `REyes_logs/eucentricity_correction.log`

#### Key Processing Steps:
1. **File Processing**: Locates SerialEM log file and navigation file, extracts stage coordinates
2. **Plane Fitting**: Performs linear regression to find optimal eucentricity plane using equation `z = ax + by + c`
3. **Coordinate Correction**: Updates Z coordinates in navigation file based on fitted plane
4. **Visualization Generation**: Creates comprehensive 6-panel visualization with before/after comparisons

#### Usage:
```bash
eucentricity-1
```

The script automatically processes files in the current directory and the `grid_squares` subdirectory.

---

### 4. REyes Diffraction Map Processor (DMP)

**Workflow Position: Step 4**

This script processes blocks of diffraction images, classifies their quality based on diffraction spots and Fourier Transform peaks, and generates comprehensive analysis outputs and visualizations.

#### Key Features:
- **Multi-Microscope Support**: Configurable presets for different microscope/camera combinations
- **Quality Classification**: 5-category system (Good/Bad/Poor/No diffraction/Grid)
- **Batch Processing**: Handles multiple blocks with progress tracking
- **Target Selection**: Customizable top target extraction per block
- **Comprehensive Visualization**: Block-specific maps and quality overlays

#### Key Outputs:
- **CSV Files**:
  - `dif_maps/dif_map_sums.csv`: Main results containing all analysis metrics
  - `targets/targets_sum.csv`: Top targets based on diffraction intensity
  - `targets/targets_spots.csv`: Top targets based on spot count
  - `targets/targets_quality.csv`: Top targets based on diffraction quality
- **Visualization Maps** (in `dif_maps/diff_blocks_maps/`):
  - Block-specific diffraction intensity maps
  - Filtered peaks distribution maps
  - Fourier Transform peaks maps
  - Quality classification maps with color coding
- **Block Data**:
  - `reyes.json`: File indicating processed state of each block
- **Log File**: Detailed processing log in `REyes_logs/dmp_processing.log`

#### Key Processing Steps:
1. **Block Discovery**: Identifies folders containing diffraction data and determines file ranges
2. **Image Processing**: Automatic resizing to 2048×2048, Gaussian filtering, binary image creation
3. **Quality Classification**: Five-category system with primary spot detection and secondary Fourier Transform analysis
4. **Data Organization**: Coordinates extraction, DataFrame construction, and top target selection

#### Usage:
```bash
dif-map-2 [options]
```

Options:
- `--microscope`: Specify microscope configuration (default: Arctica-CETA)
- `--targets-per-block`: Number of top targets to select (default: 4, range: 1-10)
- `--skip-processed`: Skip previously processed blocks
- `--proc-blocks N`: Limit processing to N blocks
- `--folder`: Process specific folder only

Example:
```bash
dif-map-2 --microscope Arctica-CETA --targets-per-block 6 --skip-processed
```

#### Quality Classification Criteria:
- **Good diffraction**: ≥10 spots with strong pattern correlation
- **Bad diffraction**: ≥10 spots with weak pattern correlation
- **Poor diffraction**: 3-9 spots
- **No diffraction**: <3 spots
- **Grid**: Intensity below threshold (calculated as mean intensity ÷ grid_rule)

---

### 5. REyes Navigation File Generator (NFG)

**Workflow Position: Step 5**

This script processes diffraction target data and generates SerialEM-compatible navigation files with quality metrics and diffraction pattern visualizations.

#### Key Features:
- **Multi-Target Processing**: Handles quality, spots, and sum-based target CSV files
- **Navigation File Generation**: Creates SerialEM-compatible .nav files with proper item numbering
- **Diffraction Visualization**: Generates pattern snapshots with resolution rings
- **Multi-Microscope Support**: Configurable presets for different microscope/camera combinations

#### Key Outputs:
- **Navigation Files**: Generated from corresponding CSV files with appropriate item numbering:
  - `targets/targets_quality.nav`: Quality-based targets (Items 101+)
  - `targets/targets_spots.nav`: Spot count-based targets (Items 201+)
  - `targets/targets_sum.nav`: Intensity sum-based targets (Items 301+)
- **Visualization Directories**:
  - `targets_quality_diff_snapshots/`: Quality target diffraction patterns
  - `targets_spots_diff_snapshots/`: Spot target diffraction patterns
  - `targets_sum_diff_snapshots/`: Sum target diffraction patterns
- **Log File**: Detailed processing log in `REyes_logs/targets_creation.log`

#### Key Processing Steps:
1. **CSV File Processing**: Locates all target CSV files and determines target type from filename
2. **Navigation File Generation**: Assigns item numbers and creates SerialEM-compatible navigation entries
3. **Diffraction Visualization**: Generates PNG snapshots for each target with resolution rings and quality metrics

#### Usage:
```bash
write-targets-3 [options]
```

Options:
- `--microscope`: Specify microscope configuration (default: Arctica-CETA)
- `--camera-length`: Override default camera length (in mm)
- `--pixel-size`: Override default pixel size (in mm/pixel)

Example:
```bash
write-targets-3 --microscope Arctica-CETA --camera-length 2700
```

---

### 6. REyes Selected Targets Adder (STA)

**Workflow Position: Step 5a (Optional)**

This script provides functionality to append additional targets (e.g., from nXDS) to existing navigation files, with full diffraction pattern visualization support.

#### Key Features:
- **Custom Target Addition**: Reads target filenames from text file and matches with diffraction data
- **Quality-Based Ranking**: Sorts targets by diffraction quality, pattern peaks, and intensity
- **Flexible Output Options**: Create new navigation file or append to existing
- **Diffraction Visualization**: Generates pattern snapshots with resolution rings

#### Key Outputs:
Two possible output structures based on user choice:

1. **New Navigation File Option**:
   - `targets/targets_nxds.nav`: New navigation file starting at item 401
   - `targets/targets_nxds_snapshots/`: Directory containing diffraction patterns
   
2. **Append Option**:
   - Updates existing `targets.nav` file with new entries
   - `targets/targets_diff_snapshots/`: Adds new diffraction patterns

#### Key Processing Steps:
1. **Target Processing**: Reads filenames from input text file and matches with diffraction mapping data
2. **Quality Ranking**: Sorts by quality: Good → Bad → Poor → No diffraction → Grid
3. **Navigation Generation**: Creates entries starting at item 401 with proper SerialEM formatting
4. **Visualization**: Generates diffraction snapshots with calibrated resolution rings

#### Usage:
```bash
append-targets-3-1 [options]
```

Options:
- `--targets`: Input file with target filenames (default: add_targets.txt)
- `--microscope`: Microscope configuration (default: Arctica-CETA)
- `--create-nxds-nav`: Choose output mode ('yes' for new file, 'no' for append)
- `--output-folder`: Custom folder for diffraction snapshots
- `--camera-length`: Override default camera length (mm)
- `--pixel-size`: Override default pixel size (mm/pixel)

#### Input File Format:
`add_targets.txt` should contain one filename per line:
```
20240119_00123_integrated_movie.mrc
20240119_00124_integrated_movie.mrc
20240119_00125_integrated_movie.mrc
```

Example:
```bash
append-targets-3-1 --microscope Arctica-CETA --create-nxds-nav yes
```

---

### 7. REyes Final Targets Processor (FTP)

**Workflow Position: Step 6**

This script combines multiple navigation files into a final target list while ensuring proper spatial distribution of targets. It intelligently selects top targets from different sources while maintaining minimum separation distances.

#### Key Features:
- **Smart Target Selection**: Spatial separation with configurable minimum distances
- **Multiple Input Sources**: Combines nXDS, quality, spots, and sum-based targets
- **Dual Selection Modes**: Category-based (default) or block-based target selection
- **Priority-Based Processing**: Processes files in optimal order for best target selection
- **Comprehensive Validation**: Distance calculations and coordinate validation

#### Key Inputs:
Default input files (can be customized):
- `targets/targets_nxds.nav`: Targets from nXDS analysis
- `targets/targets_quality.nav`: Quality-based targets
- `targets/targets_spots.nav`: Spot count-based targets
- `targets/targets_sum.nav`: Intensity sum-based targets

#### Key Outputs:
- **Navigation File**: `targets/targets.nav` containing combined and validated targets
- **Visualization Directories** (block mode only): `targets/targets_per_block_diff_snapshots/`
- **Log Files**: `REyes_logs/processing.log` with detailed selection process and skip reasons

#### Key Processing Steps:
1. **Input Processing**: Validates input navigation files and parses target coordinates
2. **Selection Modes**: Category mode (top N per source) or block mode (top N per diffraction block)
3. **Spatial Validation**: Calculates distances and enforces minimum separation tolerance
4. **Output Generation**: Creates combined navigation file with proper SerialEM formatting

#### Usage:
```bash
create-final-targets-4 [options]
```

Options:
- `--microscope`: Microscope configuration (default: Arctica-CETA)
- `--output`: Output navigation file name (default: targets.nav)
- `--top-target-per-category`: Targets per category (category mode, default: 2)
- `--top-target-per-block`: Targets per block (block mode)
- `--tolerance`: Minimum distance between targets in microns (default: 10.1)
- `--input-files`: Custom input navigation files

Example:
```bash
create-final-targets-4 --top-target-per-category 3 --tolerance 15.0
```

#### Target Selection Process:
1. **Priority Order**: nXDS → quality → spots → sum-based targets
2. **Selection Criteria**: Must maintain minimum distance from existing targets
3. **Validation Checks**: Coordinate presence, distance calculations, file integrity

---

### 8. REyes Final Map Plotter (FMP)

**Workflow Position: Step 7 (Final)**

This script creates comprehensive overlay visualizations combining montage images with diffraction data maps and selected targets. It provides multiple visualization options and customizable display parameters.

#### Key Features:
- **Multiple Map Types**: Sum, Spots, FT Spots, and Quality classification overlays
- **Flexible Display Control**: Independent control of montage, diffraction maps, and targets
- **Smart Point Sizing**: Automatic optimization or fixed sizing for different coordinate scales
- **Manual Plot Control**: Custom plot limits and display parameters
- **Quality Color Mapping**: Five-level diffraction quality visualization

#### Key Outputs:
Generated in `dif_maps/` directory:
- **Visualization Maps**:
  - `mnt_sum_dif_map_w_targets.png`: Intensity sum overlay with logarithmic scaling
  - `mnt_dif_spots_map_w_targets.png`: Diffraction spots distribution overlay
  - `mnt_ft_spots_map_w_targets.png`: FT spots pattern overlay
  - `mnt_quality_map_w_targets.png`: Quality classification overlay with color coding
- **CSV Data**: Processed mapping data for each visualization type
- **Log File**: `REyes_logs/diffraction_maps.log`

#### Key Processing Steps:
1. **Data Integration**: Loads montage data from MRC files and processes stage positions from MDOC files
2. **Coordinate Processing**: Handles coordinate transformations and calculates proper rotations
3. **Visualization Generation**: Creates multi-layer visualizations with controlled transparency
4. **Quality Mapping**: Color-codes diffraction quality: Grid → No diffraction → Poor → Bad → Good

#### Usage:
```bash
mnt-maps-targets-5 [options]
```

Options:
- `--microscope`: Microscope configuration (default: Arctica-CETA)
- `--dpix-size`: Fixed size for scatter plot diffraction pixels (default: automatic sizing)
- `--no-targets`: Disable target overlay from targets.nav
- `--no-diff-map`: Show only montage frames without diffraction data
- `--no-mnt`: Show only diffraction map and targets without montage
- `--plot-limits`: Manual plot boundaries (XMIN XMAX YMIN YMAX)

Example:
```bash
mnt-maps-targets-5 --microscope Arctica-CETA --dpix-size 30 --plot-limits -500 500 -500 500
```

#### Required Input Files:
- `.mrc`: Montage image data
- `.mdoc`: Stage position data
- `targets/targets.nav`: Selected targets (optional)
- `dif_maps/dif_map_sums.csv`: Diffraction analysis data from dif-map-2

---

### 9. REyes Processing Monitor

**Workflow Position: Orchestrates the entire pyREyes pipeline**

This script provides automated execution and monitoring of the complete diffraction data processing pipeline using a state machine to track progress and automatically trigger each processing step.

#### Key Features:
- **Automated Workflow Execution**: State-based pipeline management with automatic progression
- **Flexible Grid Square Modes**: Support for manual or automatic grid square selection
- **Block-by-Block Processing**: Sequential diffraction map processing with progress tracking
- **Integrated Movie Processing**: Optional AutoProcess support for acquired movies
- **Structure Solution**: Optional AutoSolve integration for automatic structure determination
- **Comprehensive Monitoring**: Log file tracking and error handling across all pipeline steps

#### States and Transitions:
1. **WAITING_FOR_MONTAGE**: Monitors for montage completion messages in SerialEM logs
2. **WAITING_FOR_MANUAL_SQUARES**: Waits for user grid square selection (optional)
3. **WAITING_FOR_GRID_SQUARES**: Tracks eucentricity processing completion
4. **WAITING_FOR_DIFFRACTION_MAP**: Processes diffraction blocks sequentially
5. **GENERATING_TARGETS**: Executes target list generation and spatial filtering
6. **GENERATING_FINAL_MAP**: Creates overlay visualizations
7. **WAITING_FOR_MOVIES**: Monitors movie acquisition (optional)
8. **RUNNING_AUTOSOLVE**: Executes structure solution (optional)
9. **COMPLETED**: Final state when all processing finished

#### Usage:
```bash
reyes-monitor [options]
```

Options:
- `--microscope`: Microscope configuration (default: Arctica-CETA)
- `--filtering`: Grid square filtering type (default: default)
- `--manual-squares`: Enable manual grid squares selection mode
- `--starting-state`: Begin monitoring from a specific state
- `--current-block`: Set current diffraction map block number (default: 1)
- `--stepscan-only`: Skip movie collection and processing
- `--autoprocess`: Enable automatic movie processing
- `--autosolve`: Enable automatic structure solution (requires AutoSolve to be separately installed)

Example:
```bash
reyes-monitor --microscope Arctica-CETA --filtering 9 --autoprocess --autosolve
```

#### Output:
- **Logs**: `REyes_logs/REyes_monitor.log`
- **Processing Artifacts**: Various output files from each processing step
- **AutoSolve Results** (optional): Structure solution files

---

## Important Notes

### Automated Processing Pipeline
The REyes Monitor provides automated execution and monitoring of the complete pipeline:

```bash
reyes-monitor [--microscope <microscope-config>] [--filtering <filtering-option>]
```

The monitor automatically:
1. Detects montage completion
2. Triggers grid square processing
3. Handles eucentricity correction
4. Processes diffraction maps
5. Generates target lists
6. Creates final visualizations
7. Optionally processes movies and runs AutoSolve (if installed)

### Directory Structure
```
working_directory/
├── grid_squares/          # Grid square & eucentricity outputs
├── dif_maps/              # Diffraction analysis & final visualizations
│   └── diff_blocks_maps/  # Block-specific diffraction maps
├── targets/               # Navigation files & target snapshots
├── REyes_logs/            # Processing logs
└── movies/                # Diffraction movie data (optional)
```

### Key Processing Notes
- **Automation**: Monitor handles the complete workflow with appropriate timing
- **Flexibility**: Can start monitoring at any stage of data collection
- **Recovery**: Automatically resumes from last successful step
- **Validation**: Checks completion status before proceeding to next stage

### Common Considerations
- Ensure correct microscope configuration before starting
- Monitor log file for processing status and any warnings
- All scripts are part of the pyREyes package
- Use the `reyes-monitor` command for automated processing

The pipeline provides systematic processing of diffraction data while maintaining data integrity. For detailed information about individual components, refer to their specific sections above.

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs