# Changelog

All notable user-facing changes to pyREyes are listed here.

## 3.6.2

One option: the DQI threshold for "Good diffraction" can be set from the
command line. Without it every stage behaves exactly as in 3.6.1; the only
difference in the outputs is one new key in each block's `reyes.json`.

### Good rule (`dif-map-2`, `reyes-monitor`)
- `--good-rule DQI` sets the minimum DQI (lattice-quality peaks per
  diffraction peak) for a frame to be classified as "Good diffraction",
  overriding the `good_rule` of the microscope configuration. The value in
  force is resolved the same way everywhere: the command-line option, else the
  `REYES_GOOD_RULE` environment variable, else the microscope configuration.
  `dif-map-2` logs the value and where it came from
  (`good rule: 2 (command line)`).
- `reyes-monitor --good-rule` exports `REYES_GOOD_RULE` for every process it
  starts and also passes `--good-rule` to `dif-map-2` explicitly. Without the
  option nothing is exported or passed, so a value already in the monitor's
  environment reaches the stages unchanged.
- Each block's `reyes.json` carries a `classification` object with the
  `good_rule` and `grid_rule` the block was classified with. Every existing
  key is kept.
- `pyREyes.lib.REyes_rules` (`resolve_good_rule`, `good_rule_source`,
  `add_good_rule_argument`, `export_good_rule`, `GOOD_RULE_ENV`) is the
  helper behind both commands, for other programs built on this package
  that want the same option with the same precedence.

## 3.6.1

A maintenance release: the monitor can be extended by subclassing, and a few
stage outputs carry more information. The command-line interface and the
behaviour of every stage are unchanged; `dif-map-2` results are identical to
3.6.0.

### Monitor (`reyes-monitor`)
- `REyesProcessor` has documented extension points, so a monitor for a
  different workflow can be built on it without copying it: `module_commands()`
  (the stages it can run), `run_module_code()` (a stage's exit status;
  `run_module()` keeps returning True/False), `status_payload()` (the
  `session_status.json` document), `abort_states` / `idle_states` (the waits
  in which a SerialEM abort or the idle backstop stops it), `MONITOR_OWNED`
  (entries of the session root it writes itself), `after_eucentricity_state()`
  (the state entered once eucentricity has run), `after_movie()` (called once a
  movie's autoprocess run has finished) and `busy_elsewhere` (a tick with work
  in flight outside the process does not count as idle). See "Extending the
  monitor" in `docs/modules/monitor.md`. With nothing overridden the
  monitor's behaviour, state sequence, log phrases and `session_status.json`
  keys are unchanged.

### Eucentricity (`eucentricity-1`)
- The fitted plane is written to `grid_squares/eucentricity_plane.json`:
  the coefficients of `z = a*x + b*y + c` at full precision, their standard
  errors and covariance, `r2`, `n_points`, and `trusted` (true when R^2 >= 0.65
  on at least 4 measurements). The navigator correction itself is unchanged;
  the file is for tools that want to put a position on the plane and need to
  know how well it was determined. If the file cannot be written the stage
  still succeeds.

### Snapshots (`write-targets-3`, `append-targets-3-1`, `create-final-targets-4`)
- The beam-centre search can borrow a centre from a sibling session (another
  session directory under the same parent) whose `targets/beam_center.json`
  holds a detected, trusted centre on frames of the same size, when its own
  frames do not settle it (`beam_center_search.find_beam_centres(fallback=
  session_fallback(...))`, `sibling_centre`, `read_sidecar`). A borrowed
  centre is marked on the snapshot and recorded in `beam_center.json` with
  `source: "session"` and `borrowed_from`; a sidecar that borrowed nothing is
  written exactly as in 3.6.0. The shipped stages do not use the fallback
, so their pictures are unchanged.

### Library
- `pyREyes.lib.mrc_io.read(path)` and `frames(path, mmap=False)` read an MRC
  file with the mode-0 signedness rule applied, for code that needs the data
  rather than a view of an array it already holds.
- `pyREyes.lib.REyes_logging.format_record(key, *fields)` builds the
  `KEY | field=value | ...` machine records the logs use (missing values
  print as `NA`, lists comma-joined, never a `|` or a line break inside a
  value).
- `microscope_configs.json` may carry two optional fields per microscope,
  `autoprocess_config` (the pyautoprocess configuration name to use for its
  movies when it differs from the pyREyes name) and `camera_length_calibrated`
  (mm). A configuration without them loads as before; a field without a
  default that is missing is still an error. Of the shipped configurations
  only `Arctica-Apollo-SM` sets `autoprocess_config` (`Talos-Apollo-SM`, the
  pyautoprocess name for that camera geometry); nothing in this release reads
  the fields.

## 3.6.0

### Monitor (`reyes-monitor`)
- A grid square that SerialEM did not finish no longer stalls the session. The
  block is set aside and the session continues with the next block when its log
  reports that SerialEM set it aside, when its log never reports completion but
  SerialEM has already started a later block, or when it has no log but a later
  block has one. Targets are chosen from the blocks that were mapped. Set-aside
  grid squares, and crystals that SerialEM set aside during movie collection, are
  listed in `targets/set_aside_items.json`. If the step-scan ends with no block
  mapped, the session ends as failed.
- A SerialEM error for the current grid ("ERROR DURING PIPELINE EXECUTION",
  "ERROR DURING MONTAGE ACQUISITION", or a FAILED line for the grid in the
  multi-grid summary log) is detected, and the monitor stops instead of waiting
  forever. Blocks already collected are still mapped.
- The monitor writes `session_status.json` in the session folder when it stops:
  `state` (`complete`, `failed` or `interrupted`), a `detail` text, an optional
  `reason` (`serialem_aborted`, `stepscan_failed`, `no_diffraction_map`,
  `target_generation_error`, `idle_timeout`, `sigterm`) and the time.
- Stopping the monitor with SIGTERM records the session as `interrupted`, and
  the monitor then exits with status 143 (128 + SIGTERM).
- An abort counts until SerialEM resumes the grid: once a newer log of the grid
  exists (for example `<name>-1.log` from a resume), the earlier abort is
  ignored. A monitor started after the fact (offline processing, or a late
  start) therefore still ends with SerialEM's verdict; when restarting a
  monitor for a resume, restart it after SerialEM has resumed the grid.
- A monitor restarted with `--current-block` counts the blocks an earlier run
  already mapped (`dif_maps/dif_map_sums.csv`), so a later abort or partial
  step-scan still generates targets from them.
- New `--idle-timeout` (seconds, default `0` = off): if set, a monitor waiting on
  SerialEM stops and records `interrupted` when nothing new arrives for that long.
- New `--movie-timeout` (seconds, defaults to the `--subprocess-timeout` value, 1800), separate from
  `--subprocess-timeout`: a movie whose processing exceeds it is killed, recorded
  as abandoned in its `autoprocess_done.flag` and not offered again, and the
  session moves on to the next movie.
- Logs that SerialEM opens when a stage is resumed (`<name>-1.log`,
  `<name>-2.log`, ...) are read after the original log, so the movies of a
  resumed movie stage are processed, and a grid square mapped again after a
  resume is mapped from its newest log (the latest log also decides whether the
  step-scan or the movie stage has finished). `dif-map-2` reads stage positions
  from the newest log that names a frame.
- All stages plot with the non-interactive Agg backend, so the monitor and the
  tools it launches run without a display (for example over ssh).

### Diffraction maps (`dif-map-2`)
- MRC mode 0 (8-bit) frames are read as unsigned counts unless the header marks
  them as signed. Depending on the installed reader, earlier releases could read
  8-bit detector counts above 127 (e.g. DE Apollo frames) as negative values.
- Results no longer depend on the machine or library versions: frame sums are
  exact (64-bit integer or double precision) and spot detection runs in double
  precision.
- A frame keeps its stage position when extra lines are logged between the frame
  file and its `Stage` line; previously such a frame could lose its coordinates.

### Targets and snapshots (`write-targets-3`, `append-targets-3-1`, `create-final-targets-4`)
- Resolution rings on the diffraction snapshots are drawn about the direct beam
  found in the session's frames instead of the frame centre. The centre is marked
  on the snapshot and recorded in `beam_center.json` (`targets/` for
  `write-targets-3`, the snapshot folder for the other two). If no
  reliable centre is found, the rings fall back to the frame centre and the
  snapshot says so. New option `--beam-center auto|frame|X,Y` (default `auto`).
  Only the pictures change; peak counts, quality classes and targets do not.
- Snapshots are loaded the same way as in `dif-map-2` (frame stacks summed, large
  frames binned).
- `create-final-targets-4` breaks ties in the block-mode ranking in a stable,
  reproducible order.

### SerialEM scripts (`docs/scripts/`)
- Updated DE Apollo and Ceta packages
  (`SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt`,
  `SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt`), both:
  - new run or resume of an interrupted run from any stage (automated FSD and
    Screen, single grid and multi-grid);
  - a grid square or crystal that fails is set aside and the grid goes on; a grid
    with set-aside items still ends as SUCCESS, and a grid with no mapped grid
    square fails at once;
  - column-valve check before data are marked done; stage moves wait for the
    stage and retry a refused move;
  - wait loops survive transient file errors.

  DE Apollo only: StepScan saved-frame check; 0.5 s eucentricity View exposure.
  Ceta only: the spot-size / Low Dose guard (`REyes_SpotGuard`) is added, as in
  the DE Apollo package (28 macros).

  Both packages require pyREyes 3.6.0 or later. See `docs/scripts/README.md`.

### Packaging
- Requires `pyautoprocess>=0.5.5,<0.6` (TIF conversion that clips instead of wrapping, XDS overload flagging, XSCALE statistics in `autoprocess.log` and `autoprocess-summary`).

### Upgrading from 3.4.x
- The command-line interface is compatible: no option was removed or changed;
  `--movie-timeout`, `--idle-timeout` and `--beam-center` are new, and their
  defaults keep the 3.4.10 behaviour except for the snapshot rings.
- `dif-map-2` results can differ from 3.4.10 on the same data because of the
  8-bit frame fix, the exact sums and the double-precision detection
  thresholds. Sums and peak counts change slightly; quality classes can change
  noticeably on some detectors (frames near a class boundary, for example
  between Poor and No diffraction, can move class in sizeable numbers), and the
  final target list can change as a result. For DE Apollo 8-bit frames the
  difference can be large. Re-run `dif-map-2` rather than mixing maps made
  with 3.4.x and 3.6.0 in one session. CSV columns are unchanged. `--legacy-parse` still restores the
  3.4.0 frame loading, including its reader-dependent 8-bit reads; only the
  frame sums stay exact.
- Snapshots now draw the resolution rings about the detected beam. Use
  `--beam-center frame` for the previous, frame-centred look.
- A movie whose processing exceeds `--movie-timeout` is no longer retried after
  a restart.
- A monitor stopped with SIGTERM exits with status 143 (3.4.10 was killed by
  the signal); it now writes `session_status.json` first.
- Upgrade pyautoprocess to 0.5.5 or later. A fresh `pip install pyREyes` installs it; in an existing environment `pip install -U pyREyes` upgrades it because 3.6.0 requires it.
- If you use the updated DE Apollo SerialEM package, use it with pyREyes 3.6.0
  or later.
