# REyes Processing Monitor

**Part of pyREyes v3.3** - Automated workflow orchestration and monitoring tool

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
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--microscope` | string | `Arctica-CETA` | Microscope configuration preset |
| `--filtering` | string | `default` | Grid square filtering strategy |
| `--manual-squares` | flag | `False` | Enable manual grid square selection |
| `--starting-state` | string | `WAITING_FOR_MONTAGE` | Begin from specific state |
| `--current-block` | int | `1` | Set current diffraction block number |
| `--stepscan-only` | flag | `False` | Skip movie collection and processing |
| `--autoprocess` | flag | `False` | Enable automatic movie processing |
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
- Supports configurable AutoProcess parameters

### 4. Error Handling and Recovery
- Comprehensive logging with timestamps and error traces
- Timeout handling for manual intervention steps
- Graceful failure recovery with state preservation
- Detailed progress tracking and completion detection

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/REyes
- **Issues**: https://github.com/theNelsonLab/REyes/issues
- **Documentation**: https://github.com/theNelsonLab/REyes/tree/main/src/pyREyes/docs