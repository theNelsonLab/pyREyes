# REyes Navigation File Generator (NFG)

**Part of pyREyes v3.6.1** - Navigation file generator for diffraction targets

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
| `--beam-center` | string | `auto` | Centre of the resolution rings: `auto` finds the direct beam, `frame` uses the middle of the frame, or `X,Y` in pixels |

### Beam centre

*Added in pyREyes 3.6.0.* The resolution rings on the diffraction snapshots are drawn
about the direct beam, which is found from the frames being plotted (one centre per
session and frame size, the median of the frames that agree). The centre is marked with a
`+` and recorded in `targets/beam_center.json`. If it cannot be found, the
rings are drawn about the middle of the frame and the snapshot says so.
`--beam-center frame` restores the frame-centred rings of pyREyes 3.4.10 and earlier (no
`beam_center.json` is written); `--beam-center X,Y` sets the centre by hand, in pixels of
the plotted frame. Only the pictures change: peak counts, quality classes and target
selection do not use the beam centre. Snapshots are loaded the same way as in `dif-map-2`
(frame stacks summed, large frames binned to 2048x2048).

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
- **`targets/beam_center.json`** - One file for all snapshot directories: the beam centre the rings were drawn about (not written with `--beam-center frame`)

### Log Files
- **`REyes_logs/targets_creation.log`** - Processing details and errors

---

## Processing Details

### 1. CSV File Processing
- Locates all `targets*.csv` files in targets directory and determines target type from filename

### 2. Navigation File Generation  
- Assigns item numbers: quality (101+), spots (201+), sum (301+) and creates SerialEM-compatible navigation entries

### 3. Diffraction Visualization
- Generates PNG snapshots for each target with resolution rings (centred on the detected direct beam, see *Beam centre*) and quality metrics

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/pyREyes
- **Issues**: https://github.com/theNelsonLab/pyREyes/issues
- **Documentation**: https://github.com/theNelsonLab/pyREyes/tree/main/src/pyREyes/docs