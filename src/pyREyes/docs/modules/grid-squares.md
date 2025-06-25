# REyes Grid Squares Searcher (GSS)

**Part of pyREyes v3** - Automated MicroED diffraction data collection and analysis package

The Grid Squares Searcher processes diffraction data from **grid squares**, enabling automated detection and navigation generation based on montage data collected from supported microscopes. This tool is the first step in the REyes automated MicroED workflow.

---

## Key Features

- **Automated File Detection**: Automatically detects `.mrc` and `.mdoc` montage files in subdirectories
- **Multi-Microscope Support**: Configurable presets optimized for different microscope/camera combinations
- **Intelligent Centroid Detection**: Advanced image processing with local maxima detection and filtering
- **High-Quality Visualization**: Combined stitched montage images with overlaid centroids
- **SerialEM Integration**: Compatible navigation file generation for workflow integration

---


## Workflow Position

This tool is **step 1** in the pyREyes pipeline:

1. **`grid-squares-0`** ← *You are here* - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---


## Quick Start

1. Navigate to a directory containing `.mrc` and `.mdoc` montage files
2. Run the command:
   ```bash
   grid-squares-0
   ```
3. Check the `grid_squares/` output directory for results

---

### Usage
```bash
grid-squares-0 --microscope Arctica-Apollo-Prot --filtering 16
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--filtering` | string | `default` | Centroid filtering strategy |
| `--debug` | flag | `False` | Enable debug mode with additional outputs |

---

### Filtering Options

Control the number and selection of detected grid squares:

#### Standard Options
- `default` - Standard detection (equivalent to 4x4 squares)
- `None` - No filtering, return all detected squares

#### Target Count Options
Specify exact number of squares to target:

| Option | Target Squares | Option | Target Squares | Option | Target Squares |
|--------|----------------|--------|----------------|--------|----------------|
| `1` | 1 | `36` | 6×6 grid | `121` | 11×11 grid |
| `4` | 2×2 grid | `49` | 7×7 grid | `144` | 12×12 grid |
| `9` | 3×3 grid | `64` | 8×8 grid | `169` | 13×13 grid |
| `16` | 4×4 grid | `81` | 9×9 grid | `196` | 14×14 grid |
| `25` | 5×5 grid | `100` | 10×10 grid | | |

#### Special Options
- `96` - 8×12 grid (96-well plate configuration)

---

## Output Files

All outputs are saved to the `grid_squares/` directory:

### Visualization Files
- **`combined_grid_montage.png`** - Complete stitched montage with all detected centroids overlaid
- **`eucentricity_points.png`** - Subset showing selected squares for eucentric height adjustment

### Navigation Files (SerialEM Compatible)
- **`grid_squares.nav`** - Navigation file with all detected grid square targets
- **`eucentricity.nav`** - Navigation file with selected squares for eucentric height adjustment

### Log Files
- **`grid_squares.log`** - Processing log with configuration settings and results

---

## Processing Details

### 1. File Discovery and Validation
- Searches and matches `.mrc` with corresponding `.mdoc` metadata file

### 2. Stage Coordinate Processing
- Extracts metadata, and calculates rotation angle

### 3. Image Processing and Alignment
- Loads montage frames, applies intensity clipping, and constructs stitched montage based on stage coordinates

### 4. Centroid Detection
- Creates binary mask, removes edge-touching regions and applies size filtering
- Merges nearby small regions using configurable thresholds, and extracts centroid coordinates in stage coordinate system

### 5. Filtering and Selection
- Applies user-specified filtering, and selects subset for eucentricity

### 6. Output Generation
- Generates SerialEM-compatible navigation files and final montage images

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs