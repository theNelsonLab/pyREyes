# REyes Navigator Eucentricity Corrector (NEC)

**Part of pyREyes v3** - Eucentricity tilt correction tool

Automatically corrects eucentric height by fitting a plane to stage positions and adjusting Z coordinates in navigation files.

---

## Key Features

- **Automatic File Detection**: Finds SerialEM log and navigation files
- **Plane Fitting**: Linear regression to determine optimal eucentricity plane
- **Coordinate Correction**: Updates Z positions while preserving X,Y coordinates
- **Comprehensive Visualization**: 6-panel plot showing correction results

---

## Workflow Position

This tool is **step 2** in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. **`eucentricity-1`** ← *You are here* - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
- SerialEM log file ending with `_grid_squares.log` in current directory
- `grid_squares/grid_squares.nav` file from previous steps

### Usage
```bash
eucentricity-1
```

---

## Output Files

- **`grid_squares/grid_squares.nav`** - Updated with corrected Z coordinates
- **`grid_squares/eucentricity_correction.png`** - 6-panel visualization of the regression model
- **`REyes_logs/eucentricity_correction.log`** - Plane equation and statistics

---

## Processing Details

Fits plane equation `z = ax + by + c` to coordinate data using linear regression, then updates navigation file Z coordinates based on fitted plane.

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs