# SerialEM script packages

SerialEM script packages (`ScriptPackagePath` files) that drive REyes data collection.
Load the one that matches your camera. They are **not interchangeable**.
Both are set up for a Talos Arctica. On a Talos F200C with a DE Apollo camera, use the DE Apollo
package with the montage magnification in `REyes_MNT` changed from 115x to 155x LM (see
`docs/tutorials/Configurable_SerialEM_params.md`, section 1) and `--microscope F200C-Apollo`.

> **NOTE:** The User Guide PDF (`docs/tutorials/User_guide.pdf`) shows the SerialEM script
> workflow of REyes 3.4.0. In the current scripts the entry points are `START_REYES` and
> `BatchED_Main`; see below.

| File | Instrument | Macros | Pipeline modes |
|---|---|---|---|
| `SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt` | Talos Arctica + Ceta camera | 27 | FSD, Screen, Montage; batched collection |
| `SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt` | Talos Arctica + DE Apollo camera | 28 | FSD, Screen, Montage; batched collection |

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

## Differences between the two packages

| | DE Apollo | Ceta |
|---|---|---|
| Who writes the image files | the DE server (frames folder), then `RunInShell copy` into the session folder | SerialEM (`OpenNewFile` / `Save` / `CloseFile`) |
| `serialEM_log` offset in `microscope_configs.json` | 1 | 2 |
| StepScan exposure | 0.25 s with frame time 0.25 s | 1 s |
| Spot-size / Low Dose guard (`REyes_SpotGuard`) | yes | no |
| Resume of StepScan / movies (done flags) | yes | no |

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
