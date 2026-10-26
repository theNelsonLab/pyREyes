# SerialEM script packages

SerialEM script packages (`ScriptPackagePath` files) that drive REyes data collection.
Load the one that matches your camera. They are **not interchangeable**.
Both are set up for a Talos Arctica. On a Talos F200C with a DE Apollo camera, use the DE Apollo
package with the montage magnification in `REyes_MNT` (and the montage name used on resume in
`Automated_Pipeline_Helper`) changed from 115x to 155x LM (see
`docs/tutorials/Configurable_SerialEM_params.md`, section 1) and `--microscope F200C-Apollo`.

> **NOTE:** The User Guide PDF (`docs/tutorials/User_guide.pdf`) shows the SerialEM script
> workflow of REyes 3.4.0. In the current scripts the entry points are `START_REYES` and
> `BatchED_Main`; see below.

| File | Instrument | Macros | Pipeline modes |
|---|---|---|---|
| `SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt` | Talos Arctica + Ceta camera | 28 | FSD, Screen, Montage; batched collection; resume from any stage |
| `SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt` | Talos Arctica + DE Apollo camera | 28 | FSD, Screen, Montage; batched collection; resume from any stage |

> **Version requirement:** both packages shipped with pyREyes 3.6.0 write log records
> (set-aside grid squares and crystals, early end of a StepScan) that only
> `reyes-monitor` 3.6.0 or later understands. Use them with **pyREyes >= 3.6.0**. With an
> older monitor, a set-aside grid square leaves the monitor waiting. `reyes-monitor` 3.6.0
> still reads the logs of the earlier packages.

Both packages share the same structure:

- **`START_REYES`** is the entry point for REyes sessions. It asks for single-grid or
  multi-grid (autoloader) acquisition, automated or manual workflow, and the pipeline
  mode, then runs the pipeline for every grid.
  - **FSD**: montage, grid squares, eucentricity, StepScan diffraction map, targets, rotation movies
  - **Screen**: montage, grid squares, eucentricity, StepScan diffraction map (no movies)
  - **Montage**: grid montage only
- **`BatchED_Main`** (`BatchED_Main_CETA` in the Ceta package) collects data on navigator
  points you have already added: continuous rotation, still snapshots, multi-angle
  snapshots, or Search-mode images.
- **`REYES_Setup_Standalone`** sets the parameters needed to run `REyes_MNT`,
  `REyes_STPSCN` or `REyes_DFMV` on their own (for example after restarting SerialEM).
- **`AutoloaderScreen`** loads each occupied autoloader slot in a fixed range (set at the top
  of the macro) and acquires a montage map of it.

See `docs/tutorials/Configurable_SerialEM_params.md` for the full macro list and the
system-specific values to adjust.

## Changes in pyREyes 3.6.0 (both packages)

The DE Apollo macro list and slot numbers are the same as before. The Ceta package gains
`REyes_SpotGuard` (slot 34, as in the DE Apollo package) and now has 28 macros. All other
changes are inside the macros and apply to both packages unless marked.

- **New run or resume from any stage.** For the automated FSD and Screen modes,
  `START_REYES` asks whether this is a new run (press Enter) or a resume (answer 0). A
  resume asks for the stage to restart from (FSD: 1-7, Screen: 1-5: montage, grid squares,
  eucentricity, eucentric plane, StepScan, wait for targets, rotation movies); this number
  has no default. Use the same base directory and sample name as the interrupted run.
  Stages that collect data skip what they already finished (their done flags), so
  restarting at the stage that stopped re-collects only what is missing. If no navigator
  is open, it is read back from the montage navigator in the grid folder
  (`<sample>_115LM_NxN_nav.nav` with the DE Apollo package, `<sample>_210LM_NxN_nav.nav`
  with the Ceta package). Resuming
  at the StepScan or movie stage reuses the crystal-target block recorded in
  `targets\range_dfmv.txt` instead of merging the targets a second time. The manual
  workflow has no resume.
- **Multi-grid (autoloader) runs.** The resume applies to the first grid of the slot range
  only; later grids start from their montage. The grid must be back in its slot. After a
  cartridge exchange, `START_REYES` waits (up to 2 min) for the stage to report that it is
  no longer busy before the next stage.
- **A failed grid square or crystal is set aside; the grid goes on.** In `REyes_STPSCN`
  each grid square, and in `REyes_DFMV` each crystal, runs in its own error handler. A
  square or crystal that fails gets no done flag (and, with the DE Apollo package, is not
  copied); its folder (DE Apollo: in the DE output directory; Ceta: in the session folder)
  is renamed with a `-SKIPPED` (error) or `-VALVECLOSED` (column valve closed) suffix, and
  the log records `BLOCK_SKIPPED` / `ITEM_SKIPPED`. The StepScan logs a
  `STPSCN_SUMMARY ... set_aside=` line and a `STPSCN_RESULT` line when it ends without
  mapping every square. A grid with set-aside squares or crystals still ends as SUCCESS,
  with the count in the reason (`SUCCESS - Blocks_set_aside_N` / `Items_set_aside_N`). A
  grid where no grid square could be mapped fails at once (`All_blocks_failed`) instead
  of waiting for targets. `reyes-monitor` (>= 3.6.0) reads these records; see the monitor
  documentation.
- **Valve and stage safety.** The column valve is re-opened before each grid square and
  each crystal if something closed it (`VALVE_REOPENED`), and checked again before the
  data are marked done (and, on the DE Apollo, copied). A square or crystal exposed with the valve closed is
  set aside (`VALVE_CLOSED`) and stays not done, so a resume collects it again. Stage moves in `REyes_ECNT`,
  `REyes_STPSCN` and `REyes_DFMV` wait for the stage to be idle and retry a refused move
  (up to 5 times, 20 s apart; `STAGE_RETRY` in the log). A move that still fails sets
  that grid square or crystal aside (in `REyes_ECNT` it stops the macro).
- **StepScan frame check (DE Apollo only).** After the first exposure of each grid square, `REyes_STPSCN`
  reads back how many frames the camera saved. If it saved more than one, the exposure is
  adjusted by one sensor frame and checked again; the result is logged as
  `STPSCN_FRAMES`, and the check never stops the grid (`dif-map-2` sums a frame stack). If
  the camera does not report a frame count, the log says `frames=unknown`.
- **Movie stage.** The done flag of a movie is written before `Completed movie
  acquisition`, the end phrase is always printed, and a run in which no crystal produced a
  movie ends the grid as FAILED.
- **Wait polls survive transient errors.** In `ECNT_nav_ready`, `ECNT_corrected`,
  `Targets_nav_ready` and `Manual_ready`, every file check and delay runs in its own error
  handler; an error is logged as `WAIT_POLL_ERROR` and the wait goes on. `ECNT_corrected`
  allows 20 polls (was 5).
- **Spot-size / Low Dose guard (Ceta: new).** `START_REYES` asks for the expected
  diffraction spot size (Ceta default 10) and what to do on a mismatch. `REyes_STPSCN` and
  `REyes_DFMV` call `REyes_SpotGuard`, which stops that stage for the grid if Low Dose is
  off and checks the spot size in the Record area. It never changes the spot size.
- **Eucentricity View exposure (DE Apollo only).** `REyes_ECNT` uses a 0.5 s View exposure
  and frame time (was 1 s). The Ceta View exposure stays 1.5 s.
- The navigator is saved to disk after the eucentricity correction is applied.

## Differences between the two packages

| | DE Apollo | Ceta |
|---|---|---|
| Who writes the image files | the DE server (frames folder), then `RunInShell copy` into the session folder | SerialEM (`OpenNewFile` / `Save` / `CloseFile`) |
| `serialEM_log` offset in `microscope_configs.json` | 1 | 2 |
| StepScan exposure | 0.25 s with frame time 0.25 s | 1 s |
| Spot-size / Low Dose guard (`REyes_SpotGuard`) | yes | yes |
| Resume of StepScan / movies (done flags) | yes | yes |
| Resume from any stage (`START_REYES` prompt) | yes | yes |
| Set-aside of failed grid squares / crystals | yes | yes |
| Valve check, stage-move retry | yes | yes |
| StepScan frame check | yes | no (SerialEM saves one image per position) |

Start the monitor with the matching microscope name, e.g.
`reyes-monitor --microscope Arctica-CETA` or `reyes-monitor --microscope Arctica-Apollo-SM`.

**`serialEM_log`** is the number of log lines `dif_map` skips from the line naming a
StepScan `.mrc` file to the `Stage x y z` line. Do not insert commands (including `Echo`)
between the file report and `ReportStageXYZ` in the StepScan inner loop of either
package, or coordinates can no longer be read from the log. If you use a different
microscope name, set its `serialEM_log` to match the package you load.

**DE Apollo StepScan exposure.** Keep the Record exposure equal to the frame time
(0.25 s). A longer exposure makes the camera save several sub-frames per position, so each
diffraction image becomes a frame stack.

## SerialEM variables

Variables that SerialEM keeps between script runs are assigned with `:=` (persistent);
all others with `=` (regular). SerialEM requires a variable to be assigned the same way
every time, so each variable name uses one form throughout a package. `START_REYES` and
`BatchED_Main` both start with `ClearPersistentVars`, so running one clears the settings
of the other; run `REYES_Setup_Standalone` before running `REyes_MNT`, `REyes_STPSCN` or
`REyes_DFMV` on their own.
