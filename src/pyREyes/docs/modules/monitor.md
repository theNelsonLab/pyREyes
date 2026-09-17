# REyes Processing Monitor

**Part of pyREyes v3.4.10** - Automated workflow orchestration and monitoring tool

Provides automated execution and monitoring of the complete diffraction data processing pipeline using a state machine to track progress and automatically trigger each processing step.

---

## Key Features

- **Automated Workflow Execution**: State-based pipeline management with automatic progression
- **Flexible Grid Square Modes**: Support for manual or automatic grid square selection
- **Block-by-Block Processing**: Sequential diffraction map processing with progress tracking
- **Integrated Movie Processing**: Optional AutoProcess support for acquired movies
- **Structure Solution**: Optional AutoSolve integration for automatic structure determination
- **Comprehensive Monitoring**: Log file tracking and error handling across all pipeline steps

---

## Workflow Position

This tool **orchestrates the entire pyREyes pipeline**:

1. `grid-squares-0` - Automated grid square detection
   - 1.1. `manual-squares-0-1` - Manual selection (optional)
2. `eucentricity-1` - Eucentric height refinement
3. `dif-map-2` - Diffraction mapping
4. `write-targets-3` - Target list generation
   - 4.1. `append-targets-3-1` - Manually add targets (optional)
5. `create-final-targets-4` - Final target preparation
6. `mnt-maps-targets-5` - Collection execution

**`reyes-monitor`** automatically runs all steps above and pauses for user input on optional manual steps

---

## Quick Start

### Prerequisites
- SerialEM montage and diffraction collection setup
- Optional: AutoProcess installed for movie processing
- Optional: AutoSolve installed for structure solution

### Usage
```bash
reyes-monitor

# With movie processing
reyes-monitor --autoprocess --microscope-config Arctica-CETA-mrc-SM
```

### Command Line Arguments

#### REyes arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--filtering` | string | `default` | Grid square filtering strategy |
| `--manual-squares` | flag | `False` | Enable manual grid square selection |
| `--camera-length` | float | config | Override camera length in mm |
| `--top-target-per-category` | int | `2` | Top targets per category (used by `create-final-targets-4`) |
| `--top-target-per-block` | int | none | Top targets per block; enables block-based selection in `create-final-targets-4` |
| `--target-lists` | list | `sum peaks dqi` | Target lists for block-based selection: `sum` (intensity), `peaks` (diffraction peaks), `dqi` (quality); 1-3 values. Applies only with `--top-target-per-block`; category-based selection always uses the nxds, quality, spots and sum navigation files |
| `--starting-state` | string | `WAITING_FOR_MONTAGE` | Begin from specific state |
| `--current-block` | int | `1` | Set current diffraction block number |
| `--stepscan-only` | flag | `False` | Skip movie collection and processing |
| `--montage-only` | flag | `False` | Process only the initial montage and grid squares, then exit |
| `--subprocess-timeout` | float | `1800` | Seconds before a processing subprocess is treated as hung and killed |
| `--autosolve-timeout` | float | none | Seconds before AutoSolve is killed (no limit by default) |
| `--autoprocess` | flag | `False` | Enable automatic movie processing (requires `--microscope-config`) |

#### AutoProcess arguments (passed to `autoprocess`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope-config` | string | none | Microscope configuration preset (required with `--autoprocess`) |
| `--config-file` | string | none | Path to microscope configuration file (default: microscope_configs.json) |
| `--rotation-axis` | string | none | Rotation axis (e.g. `"-1 0 0"`) |
| `--frame-size` | int | none | Frame size of the movie (e.g. 2048) |
| `--signal-pixel` | int | none | Signal pixel radius |
| `--min-pixel` | int | none | Minimum pixel radius |
| `--background-pixel` | int | none | Background pixel radius |
| `--pixel-size` | float | none | Pixel size in microns (e.g. 0.028) |
| `--wavelength` | float | none | Beam wavelength (e.g. 0.0251) |
| `--beam-center-x` | int | none | Beam center X coordinate |
| `--beam-center-y` | int | none | Beam center Y coordinate |
| `--file-extension` | string | none | File extension for movies (e.g. `.mrc` or `.ser`) |
| `--detector-distance` | float | none | Detector distance in mm |
| `--exposure` | float | none | Exposure time |
| `--rotation` | float | none | Rotation value |
| `--pointless` | flag | `False` | Run pointless for space group analysis |
| `--parallel` | flag | `False` | Use parallel XDS (xds_par) instead of serial XDS |
| `--dqa` | flag | `False` | Enable diffraction quality analysis and frame selection |
| `--verbose` | flag | `False` | Enable verbose logging for detailed conversion validation |
| `--res-range` | string | none | Manual resolution range in Angstroms (overrides calculated values) |
| `--friedel` | string | none | Set Friedel's law for XDS (`true` or `false`, default: true) |

#### AutoSolve arguments (passed to `autosolve`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--autosolve` | flag | `False` | Run AutoSolve after AutoProcess |
| `--shelx` | `t`, `d`, `td` | `td` | Which SHELX programs to run: t=SHELXT only, d=SHELXD only, td=both |
| `--ntryt` | int | `1000` | Number of trials for initial SHELXD screening |
| `--ntryf` | int | `10000` | Number of trials for final SHELXD runs |
| `--space-group` | string | none | Target specific space group (1-230) or by name without spaces (e.g. P212121) |
| `--atoms` | string | none | Comma-separated list of atoms for SFAC (default: C,H,N,O,S) |
| `--type` | `organic`, `inorganic` | none | Type of compound |
| `--composition` | string | none | Chemical formula (overrides `--atoms`), e.g. `CaTiO3` |

----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--filtering` | string | `default` | Grid square filtering strategy |
| `--manual-squares` | flag | `False` | Enable manual grid square selection |
| `--starting-state` | string | `WAITING_FOR_MONTAGE` | Begin from specific state |
| `--current-block` | int | `1` | Set current diffraction block number |
| `--stepscan-only` | flag | `False` | Skip movie collection and processing |
| `--montage-only` | flag | `False` | Process only the initial montage and grid squares, then exit |
| `--subprocess-timeout` | float | `1800` | Seconds before a processing subprocess is treated as hung and killed |
| `--autosolve-timeout` | float | none | Seconds before AutoSolve is killed (no limit by default) |
| `--autoprocess` | flag | `False` | Enable automatic movie processing (requires `--microscope-config`) |
| `--microscope-config` | string | none | pyautoprocess microscope preset for movie processing |
| `--autosolve` | flag | `False` | Enable automatic structure solution |

---

## Output Files

### Log Files
- **`REyes_logs/REyes_monitor.log`** - Complete monitoring and processing log

### Processing Artifacts
- All output files from individual pipeline steps
- AutoProcess results (if enabled)
- AutoSolve structure solution files (if enabled)

---

## Processing Details

### 1. State Machine Monitoring
- **WAITING_FOR_MONTAGE**: Monitors for montage completion messages in SerialEM logs
- **WAITING_FOR_MANUAL_SQUARES**: Waits for user grid square selection (optional)
- **WAITING_FOR_GRID_SQUARES**: Tracks eucentricity processing completion
- **WAITING_FOR_DIFFRACTION_MAP**: Processes diffraction blocks sequentially
- **GENERATING_TARGETS**: Executes target list generation and spatial filtering
- **GENERATING_FINAL_MAP**: Creates overlay visualizations
- **WAITING_FOR_MOVIES**: Monitors movie acquisition (optional)
- **RUNNING_AUTOSOLVE**: Executes structure solution (optional)
- **COMPLETED**: Final state when all processing finished

### 2. Automated Execution
- Parses SerialEM log files to detect completion signals
- Launches pyREyes tools with appropriate microscope configurations
- Handles block-by-block diffraction processing with automatic progression
- Manages file dependencies and output directory creation

### 3. Movie Processing Integration
- Monitors targets log files for newly acquired movies
- Automatically renames and processes movies with AutoProcess
- Tracks processed movies to avoid duplication
- Writes `autoprocess_done.flag` in the movie folder after autoprocess succeeds; a restarted monitor skips movies with this flag (without `--autoprocess`, movies that were already renamed) and still counts them in the totals
- Reports an autoprocess run that exits with a non-zero code as a failure, and marks a movie as done (`autoprocess_done.flag`) only after a successful run
- Supports configurable AutoProcess parameters

### 4. Error Handling and Recovery
- Comprehensive logging with timestamps and error traces
- Timeout handling for manual intervention steps
- Processing subprocesses that exceed `--subprocess-timeout` are killed together with their child processes and handled like a failed step
- Graceful failure recovery with state preservation
- Detailed progress tracking and completion detection

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/pyREyes
- **Issues**: https://github.com/theNelsonLab/pyREyes/issues
- **Documentation**: https://github.com/theNelsonLab/pyREyes/tree/main/src/pyREyes/docs