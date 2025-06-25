# REyes Manual Grid Squares Adder (MGSA)

**Part of pyREyes v3** - Manual grid square selection processing tool

The Manual Grid Squares Adder processes SerialEM navigation files containing manually added grid squares' points, converting them into the standardized format required for the REyes automated collection pipeline. This tool allows to override automated detection with custom grid square selections.

---

## Key Features

- **Automatic NAV File Detection**: Finds and validates SerialEM navigation files
- **Manual Selection Processing**: Extracts user-selected grid squares
- **Visualization Generation**: Creates overlay plots showing manual selections on the original montage

---

## Workflow Position

This tool is **step 1.1** (optional) in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. **`manual-squares-0-1`** ← *You are here* (optional) - Manual selection
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
1. Run `grid-squares-0` first to generate initial montage data
2. Manually select grid squares in SerialEM and save `.nav` navigator file

### Usage
```bash
manual-squares-0-1
```

No command line arguments are required. The tool automatically:
- Detects REyes collected `.nav` file in the current directory
- Processes the selections and generates outputs

---

## Output Files

- **`grid_squares.nav`** - SerialEM format of the processed navigation file
- **`manual_grid_overlay.png`** - Montage image with manual selections overlaid
- **`REyes_logs/manual_squares.log`** - Detailed processing log

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs