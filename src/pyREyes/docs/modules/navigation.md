# REyes Navigation File Generator (NFG)

**Part of pyREyes v3** - Navigation file generator for diffraction targets

Processes diffraction target data and generates SerialEM-compatible navigation files with quality metrics and diffraction pattern visualizations.

---

## Key Features

- **Multi-Target Processing**: Handles quality, spots, and sum-based target CSV files
- **Navigation File Generation**: Creates SerialEM-compatible .nav files with proper item numbering
- **Diffraction Visualization**: Generates pattern snapshots with resolution rings
- **Multi-Microscope Support**: Configurable presets for different microscope/camera combinations

---

## Workflow Position

This tool is **step 4** in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. **`write-targets-3`** ← *You are here* - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
- Target CSV files in `targets/` directory (from `dif-map-2`)
- Original MRC files referenced in CSV data

### Usage
```bash
write-targets-3
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--camera-length` | float | *config based* | Override camera length (mm) |
| `--pixel-size` | float | *config based* | Override pixel size (mm/pixel) |

---

## Output Files

### Navigation Files (in `targets/` directory)
- **`targets_quality.nav`** - Quality-based targets (item 101+)
- **`targets_spots.nav`** - Spot count-based targets (item 201+)  
- **`targets_sum.nav`** - Intensity sum-based targets (item 301+)

### Visualization Directories
- **`targets_quality_diff_snapshots/`** - Quality target diffraction patterns
- **`targets_spots_diff_snapshots/`** - Spot target diffraction patterns
- **`targets_sum_diff_snapshots/`** - Sum target diffraction patterns

### Log Files
- **`REyes_logs/targets_creation.log`** - Processing details and errors

---

## Processing Details

### 1. CSV File Processing
- Locates all `targets*.csv` files in targets directory and determines target type from filename

### 2. Navigation File Generation  
- Assigns item numbers: quality (101+), spots (201+), sum (301+) and creates SerialEM-compatible navigation entries

### 3. Diffraction Visualization
- Generates PNG snapshots for each target with resolution rings and quality metrics

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs