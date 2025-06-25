# REyes Diffraction Map Processor (DMP)

**Part of pyREyes v3** - Diffraction pattern quality analysis tool

Processes blocks of diffraction images, classifies quality based on spots and Fourier Transform patterns, and generates analysis outputs and visualizations.

---

## Key Features:

- **Multi-Microscope Support**: Configurable presets for different microscope/camera combinations
- **Quality Classification**: 5-category system (Good/Bad/Poor/No diffraction/Grid)
- **Batch Processing**: Handles multiple blocks with progress tracking
- **Target Selection**: Customizable top target extraction per block
- **Comprehensive Visualization**: Block-specific maps and quality overlays

---

## Workflow Position

This tool is **step 3** in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. **`dif-map-2`** ← *You are here* - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
- Folders containing diffraction data (`.mrc` files with pattern `YYYYMMDD_XXXXX_integrated_movie.mrc`)
- Optional: SerialEM log files for coordinate extraction

### Usage

```bash
dif-map-2 --microscope Arctica-CETA --targets-per-block 6 --skip-processed
```

### Command Line Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--microscope` | `Arctica-CETA` | Microscope configuration |
| `--targets-per-block` | `4` | Top targets to select (1-10) |
| `--skip-processed` | `False` | Skip previously processed blocks |
| `--proc-blocks N` | `None` | Limit processing to N blocks |
| `--folder` | `None` | Process specific folder only |

---

## Output Files

### CSV Files
- **`dif_maps/dif_map_sums.csv`** - Main results with all analysis metrics
- **`targets/targets_sum.csv`** - Top targets by diffraction intensity
- **`targets/targets_spots.csv`** - Top targets by spot count  
- **`targets/targets_quality.csv`** - Top targets by quality classification

### Visualization Maps (`dif_maps/diff_blocks_maps/*.png`)
- Block-specific diffraction intensity maps
- Filtered peaks distribution maps
- Fourier Transform peaks maps
- Quality classification maps with color coding

### Block Data
- **`reyes.json`** - Block metadata and processing status
- **`REyes_logs/dmp_processing.log`** - Processing details

---

## Processing Details

### Quality Classification
- **Good diffraction**: ≥10 spots with strong pattern correlation
- **Bad diffraction**: ≥10 spots with weak pattern correlation  
- **Poor diffraction**: 3-9 spots
- **No diffraction**: <3 spots
- **Grid**: Intensity below threshold

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs