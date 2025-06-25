# REyes Final Map Plotter (FMP)

**Part of pyREyes v3** - Comprehensive overlay visualization tool

Creates overlay visualizations combining montage images with diffraction data maps and selected targets with multiple visualization options and customizable display parameters.

---

## Key Features

- **Multiple Map Types**: Sum, Spots, FT Spots, and Quality classification overlays
- **Flexible Display Control**: Independent control of montage, diffraction maps, and targets
- **Smart Point Sizing**: Automatic optimization or fixed sizing for different coordinate scales
- **Manual Plot Control**: Custom plot limits and display parameters
- **Quality Color Mapping**: Five-level diffraction quality visualization

---

## Workflow Position

This tool is **step 6** in the pyREyes pipeline:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. **`mnt-maps-targets-5`** ← *You are here* - Collection execution

---

## Quick Start

### Prerequisites
- `.mrc` and `.mdoc` montage files in current directory
- `dif_maps/dif_map_sums.csv` from diffraction mapping
- `targets/targets.nav` file (optional, for target overlay)

### Usage
```bash
mnt-maps-targets-5
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--dpix-size` | int | *auto* | Fixed size for diffraction pixels |
| `--no-targets` | flag | `False` | Disable target overlay |
| `--no-diff-map` | flag | `False` | Show only montage frames |
| `--no-mnt` | flag | `False` | Show only diffraction map and targets |
| `--plot-limits` | 4 floats | *auto* | Manual plot boundaries (xmin xmax ymin ymax) |

---

## Output Files

### Visualization Maps (in `dif_maps/` directory)
- **`mnt_sum_dif_map_w_targets.png`** - Intensity sum overlay with logarithmic scaling
- **`mnt_dif_spots_map_w_targets.png`** - Diffraction spots distribution overlay
- **`mnt_ft_spots_map_w_targets.png`** - FT spots pattern overlay
- **`mnt_quality_map_w_targets.png`** - Quality classification overlay with color coding

### CSV Data Files
- Processed mapping data for each visualization type

### Log Files
- **`REyes_logs/diffraction_maps.log`** - Processing details and plot parameters

---

## Processing Details

### 1. Data Integration
- Loads montage data from MRC files and processes stage positions from MDOC files

### 2. Coordinate Processing
- Handles coordinate transformations and calculates proper rotations and spatial relationships

### 3. Visualization Generation
- Creates multi-layer visualizations with controlled transparency and appropriate color schemes

### 4. Quality Mapping
- Color-codes diffraction quality: Grid → No diffraction → Poor → Bad → Good

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs