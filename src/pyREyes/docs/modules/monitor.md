# REyes Processing Monitor

**Part of pyREyes v3.6.2** - Automated workflow orchestration and monitoring tool

Provides automated execution and monitoring of the complete diffraction data processing pipeline using a state machine to track progress and automatically trigger each processing step.

---

## Key Features

- **Automated Workflow Execution**: State-based pipeline management with automatic progression
- **Flexible Grid Square Modes**: Support for manual or automatic grid square selection
- **Block-by-Block Processing**: Sequential diffraction map processing with progress tracking
- **Integrated Movie Processing**: Optional AutoProcess support for acquired movies
- **Structure Solution**: Optional AutoSolve integration for automatic structure determination
- **Comprehensive Monitoring**: Log file tracking and error handling across all pipeline steps
- **Keeps Going**: Grid squares SerialEM did not finish are set aside instead of stalling the session; SerialEM aborts are detected; the outcome is written to `session_status.json`

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
| `--good-rule` | float | config | Minimum DQI (lattice-quality peaks per diffraction peak) for "Good diffraction" in `dif-map-2`; exported as `REYES_GOOD_RULE` for every process the monitor starts and passed to `dif-map-2` explicitly (since 3.6.2) |
| `--top-target-per-category` | int | `2` | Top targets per category (used by `create-final-targets-4`) |
| `--top-target-per-block` | int | none | Top targets per block; enables block-based selection in `create-final-targets-4` |
| `--target-lists` | list | `sum peaks dqi` | Target lists for block-based selection: `sum` (intensity), `peaks` (diffraction peaks), `dqi` (quality); 1-3 values. Applies only with `--top-target-per-block`; category-based selection always uses the nxds, quality, spots and sum navigation files |
| `--starting-state` | string | `WAITING_FOR_MONTAGE` | Begin from specific state |
| `--current-block` | int | `1` | Set current diffraction block number |
| `--stepscan-only` | flag | `False` | Skip movie collection and processing |
| `--montage-only` | flag | `False` | Process only the initial montage and grid squares, then exit |
| `--subprocess-timeout` | float | `1800` | Seconds before a processing subprocess is treated as hung and killed |
| `--movie-timeout` | float | `--subprocess-timeout` (1800) | Seconds before the processing of one movie is abandoned; the movie is recorded as abandoned and the session moves on (see *Movie Processing Integration*) |
| `--idle-timeout` | float | `0` (off) | Seconds with nothing new from the microscope, while waiting on SerialEM, before the monitor stops and records the session as `interrupted`; `0` waits indefinitely |
| `--autosolve-timeout` | float | none | Seconds before AutoSolve is killed (no limit by default) |
| `--autoprocess` | flag | `False` | Enable automatic movie processing (requires `--microscope-config`) |

The good rule in force is resolved the same way in every command: an explicit `--good-rule` wins over the `REYES_GOOD_RULE` environment variable, which wins over the microscope configuration's `good_rule`; without the option the monitor exports nothing, so a value already in its environment reaches the stages unchanged.

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


---

## Output Files

### Log Files
- **`REyes_logs/REyes_monitor.log`** - Complete monitoring and processing log

### Session Files
- **`session_status.json`** (session folder) - How the session ended, written once when the monitor stops (since 3.6.0):
  - `state`: `complete`, `failed` or `interrupted`
  - `detail`: a short human-readable description (e.g. `montage only`, `step-scan only`)
  - `reason` (only when there is one): `serialem_aborted`, `stepscan_failed`, `no_diffraction_map`, `target_generation_error`, `idle_timeout` or `sigterm`
  - `utc`: time the status was written
- **`targets/set_aside_items.json`** - Grid squares (stage `STPSCN`) and crystals (stage `DFMV`) that were set aside, one entry per item with the reason, the source (`serialem` or `pyreyes`), the log record and the log file (since 3.6.0)
- **`beam_center.json`** - The beam centre the resolution rings were drawn about: `targets/beam_center.json` for the `write-targets-3` snapshots, and one in the snapshot folder of `append-targets-3-1` and of `create-final-targets-4` (`targets/targets_per_block_diff_snapshots/`); not written with `--beam-center frame`

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
- Reports an autoprocess run that exits with a non-zero code as a failure, and marks a movie as done (`autoprocess_done.flag`) after a successful run, or with an "abandoned" note when it exceeds `--movie-timeout`
- Movie logs that SerialEM opens when the movie stage is resumed (`<name>-1.log`, `<name>-2.log`, ...) are read after the original log, so the movies of a resumed stage are processed and the latest log decides when collection has ended
- The processing of each movie is limited by `--movie-timeout` (default 1800 s, the same limit as `--subprocess-timeout` in 3.4.10). A movie that exceeds it is killed and recorded as abandoned in its `autoprocess_done.flag`, so it is not offered again, even after a restart, and the session continues with the next movie
- A movie counts as processed only if autoprocess exits successfully and its output exists; an autoprocess run that exits 0 without producing output is a failure
- Crystals that SerialEM set aside (`ITEM_SKIPPED`, or `VALVE_CLOSED` in the movie stage) are listed in `targets/set_aside_items.json`; nothing waits for them
- Supports configurable AutoProcess parameters

### 4. Error Handling and Recovery
- Comprehensive logging with timestamps and error traces
- Timeout handling for manual intervention steps
- Processing subprocesses that exceed `--subprocess-timeout` are killed together with their child processes and handled like a failed step
- Graceful failure recovery with state preservation
- Detailed progress tracking and completion detection
- All plotting uses the non-interactive Agg backend, so the monitor and every stage it launches run without a display (e.g. over ssh)

### 5. Set-Aside Grid Squares (StepScan blocks)
A grid square that SerialEM did not finish no longer stalls the session (since 3.6.0). The current block is set aside, and the monitor goes on with the next block, when:
- its log reports that SerialEM set it aside (`BLOCK_SKIPPED`, or `VALVE_CLOSED` in the StepScan); its frames were never copied, so there is nothing to map;
- its log never reports completion and SerialEM has already opened a later block's log (checked on three consecutive polls);
- it has no log at all and a later block's log exists (three polls).

A grid square mapped again after a resume has a newer log beside the original (`<name>_<N>-1.log`, ...); the newest log is the block's log, and `dif-map-2` maps the block folder `<name>_<N>` with the stage positions from the newest log that names each frame. When the monitor is restarted with `--current-block`, blocks an earlier run already mapped (`dif_maps/dif_map_sums.csv`) count as mapped.

Set-aside blocks are logged as `BLOCK SET ASIDE` and listed in `targets/set_aside_items.json`. When SerialEM ends the StepScan early (`StepScan incomplete`, or a `STPSCN_RESULT` record), targets are generated from the blocks that were mapped; if no block was mapped the session ends `failed` (`stepscan_failed`).

### 6. SerialEM Abort Detection
While waiting on SerialEM, the monitor reads the SerialEM logs for an abort of the current grid (`ERROR DURING PIPELINE EXECUTION`, `ERROR DURING MONTAGE ACQUISITION`, or a FAILED line for this grid in the multi-grid summary log). An abort is logged as `SERIALEM_ABORT` and the monitor stops instead of waiting: blocks that were already collected are still mapped and targets generated from them, and `session_status.json` records `failed` with reason `serialem_aborted`. An abort counts until SerialEM resumes the grid: once a newer log of the grid exists (e.g. `<name>-1.log` from a resume), the earlier abort is ignored. A monitor started after the fact therefore still ends with SerialEM's verdict; to follow a resume, restart the monitor after SerialEM has resumed the grid.

### 7. Stopping
- `--idle-timeout` (off by default): when set, a monitor that is waiting on SerialEM and sees nothing new arrive for that many seconds stops and records `interrupted` (`idle_timeout`). SerialEM aborts are detected from the logs, so this is only a backstop.
- Stopping the monitor with SIGTERM (e.g. `kill`) records `interrupted` (`sigterm`) in `session_status.json`; the monitor then exits with status 143.

### 8. Extending the monitor
`REyesProcessor` (in `pyREyes.REyes`) is an ordinary class, and a monitor for a different workflow is written as a subclass of it (since 3.6.1). The extension points, each with a docstring in the source:

- `state_handlers()` returns the state-to-handler mapping; a subclass adds its own states and handlers to it. A handler returns `STOP`, `SKIP_DELAY` or `None`.
- `module_commands()` returns the stages the monitor can run (stage name to argv prefix, a new dict each call); `run_module_code(name, args, positional)` runs one and returns its exit status (1 when it could not run at all); `run_module(name, args)` is `run_module_code(...) == 0`.
- `after_eucentricity_state()` is the state entered once `eucentricity-1` has run (`WAITING_FOR_DIFFRACTION_MAP` unless overridden).
- `after_movie(movie_dir, ok)` is called once a movie's autoprocess run has finished (after its `autoprocess_done.flag` is written); the base class does nothing, and an exception raised here is logged without changing the movie's result.
- `status_payload(state, detail, reason)` builds the `session_status.json` document; override it to add keys.
- `abort_states` and `idle_states` (instance attributes, sets of states) are the waits in which a SerialEM abort stops the monitor at once and in which `--idle-timeout` runs; `MONITOR_OWNED` (class attribute, tuple) lists the session-root entries the monitor writes itself, which never reset the idle clock. `targets_dir` and `status_path` are where `set_aside_items.json` and `session_status.json` are written.
- `busy_elsewhere`: a handler sets it to True in a tick that has work in flight outside the process, so the idle backstop does not end the session; it is reset at the start of every tick.

These features work with both SerialEM script packages in `docs/scripts/`. The set-aside records (`BLOCK_SKIPPED`, `ITEM_SKIPPED`, `VALVE_CLOSED`, `STPSCN_RESULT`) are written by the DE Apollo and Ceta packages shipped with 3.6.0; with earlier packages the log-order rules above still apply.

---

## Support and Documentation

- **Homepage**: https://github.com/theNelsonLab/pyREyes
- **Issues**: https://github.com/theNelsonLab/pyREyes/issues
- **Documentation**: https://github.com/theNelsonLab/pyREyes/tree/main/src/pyREyes/docs