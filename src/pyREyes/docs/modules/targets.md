# REyes Selected Targets Adder (STA)

**Part of pyREyes v3** - Custom target addition tool

Appends additional targets (e.g., from nXDS) to existing navigation files with quality-based ranking and diffraction pattern visualization.

---

## Key Features

- **Custom Target Addition**: Reads target filenames from text file and matches with diffraction data
- **Quality-Based Ranking**: Sorts targets by diffraction quality, pattern peaks, and intensity
- **Flexible Output Options**: Create new navigation file or append to existing
- **Diffraction Visualization**: Generates pattern snapshots with resolution rings
- **Multi-Microscope Support**: Configurable presets for different microscope/camera combinations

---

## Workflow Position

This tool is **step 4.1** (optional) in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. **`append-targets-3-1`** ← *You are here* (optional) - Manually add targets
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

---

## Quick Start

### Prerequisites
- `add_targets.txt` file containing target filenames (one per line)
- `dif_map_sums*.csv` file from diffraction mapping
- Original MRC files referenced in targets list

### Usage
```bash
append-targets-3-1
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--targets` | string | `add_targets.txt` | File containing target filenames |
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--create-nxds-nav` | string | *prompt* | Create new file (`yes`) or append (`no`) |
| `--output-folder` | string | *auto* | Custom folder for diffraction snapshots |
| `--camera-length` | float | *config based* | Override camera length (mm) |
| `--pixel-size` | float | *config based* | Override pixel size (mm/pixel) |

---

## Output Files

### Navigation Files
- **`targets_nxds.nav`** - New navigation file (if `--create-nxds-nav yes`)
- **`targets.nav`** - Updated existing file (if `--create-nxds-nav no`)

### Visualization Directories
- **`targets_nxds_snapshots/`** - Diffraction patterns (new file option)
- **`targets_diff_snapshots/`** - Diffraction patterns (append option)

### Log Files
- **`targets/append_targets.log`** - Processing details and ranking information

---

## Processing Details

### 1. Target Processing
- Reads filenames from input text file and matches with diffraction mapping data

### 2. Quality Ranking
- Sorts by quality: Good → Bad → Poor → No diffraction → Grid
- Secondary sorting by FT peaks, filtered peaks, and intensity sum

### 3. Navigation Generation
- Creates entries starting at item 401 with proper SerialEM formatting
- Preserves coordinate and metadata information

### 4. Visualization
- Generates diffraction snapshots with calibrated resolution rings and quality metrics

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs