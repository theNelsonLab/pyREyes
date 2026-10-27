# REyes Final Targets Processor (FTP)

**Part of pyREyes v3.6.2** - Final target combination and spatial filtering tool

Combines multiple navigation files into a final target list while ensuring proper spatial distribution with minimum separation distances.

---

## Key Features

- **Smart Target Selection**: Spatial separation with configurable minimum distances
- **Multiple Input Sources**: Combines nXDS, quality, spots, and sum-based targets
- **Dual Selection Modes**: Category-based (default) or block-based target selection
- **Priority-Based Processing**: Processes files in optimal order for best target selection
- **Comprehensive Validation**: Distance calculations and coordinate validation

---

## Workflow Position

This tool is **step 5** in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. **`create-final-targets-4`** ← *You are here* - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
- Navigation files from previous steps in `targets/` directory
- `dif_maps/dif_map_sums.csv` file (for block-based mode)

### Usage
```bash
create-final-targets-4
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--output` | string | `targets.nav` | Output navigation file name |
| `--top-target-per-category` | int | `2` | Targets per category (category mode) |
| `--top-target-per-block` | int | *None* | Targets per block (block mode) |
| `--tolerance` | float | `10.1` | Minimum distance between targets (microns) |
| `--input-files` | list | *auto* | Custom input navigation files (category mode) |
| `--target-lists` | list | `sum peaks dqi` | Target lists for block mode: `sum`, `peaks`, `dqi` (1-3 values). Applies only with `--top-target-per-block` |
| `--beam-center` | string | `auto` | Centre of the resolution rings: `auto` finds the direct beam, `frame` uses the middle of the frame, or `X,Y` in pixels (block mode snapshots; see the *Beam centre* section of [navigation.md](navigation.md)) |

---

## Output Files

### Navigation Files (in `targets/` directory)
- **`targets.nav`** - Combined final navigation file with spatial filtering

### Visualization Directories (block mode only)
- **`targets/targets_per_block_diff_snapshots/`** - Diffraction patterns for selected targets, with `beam_center.json` (the beam centre the resolution rings were drawn about)

### Log Files
- **`REyes_logs/processing.log`** - Detailed selection process and skip reasons

---

## Processing Details

### 1. Input Processing
- Validates input navigation files and parses target coordinates and metadata

### 2. Selection Modes
- **Category Mode**: Selects top N from each source (nXDS → quality → spots → sum)
- **Block Mode**: Selects top N per diffraction block with quality ranking
- Ties in the ranking keep the block order (stable sort), so the same data give the same targets on every machine

### 3. Spatial Validation
- Calculates distances between targets and enforces minimum separation tolerance

### 4. Output Generation
- Creates combined navigation file with proper SerialEM formatting

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/pyREyes
- **Issues**: https://github.com/theNelsonLab/pyREyes/issues
- **Documentation**: https://github.com/theNelsonLab/pyREyes/tree/main/src/pyREyes/docs