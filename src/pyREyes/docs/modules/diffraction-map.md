# REyes Diffraction Map Processor (DMP)

**Part of pyREyes v3.6.2** - Diffraction pattern quality analysis tool

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
- Folders containing diffraction data (`.mrc` files with pattern `YYYYMMDD_XXXXX_integrated_movie.mrc` or `YYYYMMDD_XXXXX_0_movie.mrc`)
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
| `--output-csv` | `dif_map_sums.csv` | Name of the main output CSV file |
| `--targets-sum-csv` | `targets_sum.csv` | Name of the CSV file for sum-based targets |
| `--targets-spots-csv` | `targets_spots.csv` | Name of the CSV file for spot-based targets |
| `--targets-quality-csv` | `targets_quality.csv` | Name of the CSV file for quality-based targets |
| `--legacy-parse` | `False` | Load frames exactly as REyes 3.4.0 and earlier did, to reproduce old results |
| `--good-rule DQI` | config | Minimum DQI (lattice-quality peaks per diffraction peak) for a frame to count as "Good diffraction". Precedence: this option, then the `REYES_GOOD_RULE` environment variable, then the microscope configuration's `good_rule` (since 3.6.2) |

The value in force and its source are logged once per run, e.g. `good rule: 2 (command line)`.

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
- **`reyes.json`** - Block metadata, written only for blocks that were processed completely:
  - `version`, `uid`
  - `min_id`, `max_id`, `format` - index range and a zero-padded name pattern (e.g. `20251107_%05d_integrated_movie.mrc`) for blocks recorded on a single date
  - `files` - every file of the block in acquisition order; also correct for blocks recorded across midnight
  - `classification` - the rules the block was classified with: `good_rule` (the minimum DQI for "Good diffraction", after any `--good-rule` or `REYES_GOOD_RULE` override) and `grid_rule` (since 3.6.2)
- **`REyes_logs/dmp_processing.log`** - Processing details

---

## Processing Details

### Block Files
- Files are ordered by index, so a block recorded across midnight (date in the file name changes, index keeps counting) is processed as one block
- A folder holding a repeated acquisition (a complete block, then another complete block after a large index gap) uses the latest acquisition
- Blocks with missing or duplicate indices, or files that cannot be read, are skipped with a warning

### Frame Loading
- Stacks of frames are summed into one frame
- Frames larger than 2048x2048 have narrow bad columns/rows removed and are mean-binned by one integer factor to 2048x2048
- Frames smaller than 2048x2048 use detection parameters tuned for small frames (`SMALL_FRAME_PARAMS`)
- int16 frames up to 2048x2048 are processed unchanged; all other data is converted to float32; NaN/inf pixels are set to 0
- MRC mode 0 (8-bit) frames are read as unsigned counts (0-255) unless the file header carries the IMOD stamp with the signed-bytes flag, whatever the installed MRC reader returns. Earlier releases could read 8-bit detector counts above 127 (e.g. DE Apollo frames) as negative values (since 3.6.0)
- Frame sums are exact (integer frames summed in 64-bit integers, float frames in double precision), and spot detection works in double precision, so results do not depend on the machine or library versions (since 3.6.0). `--legacy-parse` restores the 3.4.0 frame loading (including its reader-dependent 8-bit reads) and spot-detection arithmetic; only the frame sums stay exact

### Stage Coordinates
- Each frame's stage position is read from the SerialEM log, `serialEM_log` lines after the line naming the frame (see `microscope_configs.json`). If the `Stage` line is not at that offset, the following lines are searched up to the next frame, so extra log lines between a frame and its `Stage` line no longer leave the frame without coordinates (since 3.6.0)

### Quality Classification
- **Good diffraction**: ≥10 spots and DQI (lattice-quality peaks ÷ diffraction peaks) at or above the good rule (`good_rule` of the microscope configuration, or `--good-rule` / `REYES_GOOD_RULE`)
- **Bad diffraction**: ≥10 spots and DQI below the good rule
- **Poor diffraction**: 3-9 spots
- **No diffraction**: <3 spots
- **Grid**: Intensity below threshold

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/pyREyes
- **Issues**: https://github.com/theNelsonLab/pyREyes/issues
- **Documentation**: https://github.com/theNelsonLab/pyREyes/tree/main/src/pyREyes/docs