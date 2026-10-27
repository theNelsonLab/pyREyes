"""
REyes Monitor
Monitoring tool for REyes processing scripts to automate the workflow.
"""

import json
import os
import signal
import time
import logging
from datetime import datetime, timezone
from subprocess import (DEVNULL, PIPE, CompletedProcess, Popen, SubprocessError,
                        TimeoutExpired, run)
import re
import sys
from enum import Enum, auto
import argparse
import threading
import pandas as pd
from pathlib import Path


from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_autoproc import (
    AUTOPROCESS_ARTEFACTS as _AUTOPROCESS_ARTEFACTS,
    AUTOPROCESS_FLAG as _AUTOPROCESS_FLAG,
    mark_done as _mark_done,
    produced_output, run_for_movie)
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs
from pyREyes.lib import REyes_serialem as serialem

MICROSCOPE_CONFIGS = load_microscope_configs()

# Wall-clock limit for a single processing subprocess. This detects stalls
# (e.g. a hung xds_par); it is not a work limit, since a movie normally
# autoprocesses in minutes. A timeout is handled like a non-zero exit: logged,
# and the monitor moves on without retrying.
DEFAULT_SUBPROCESS_TIMEOUT = 1800

# How often the monitor re-checks the filesystem. Every stage is triggered by a
# log file appearing, so this is pure latency between one stage finishing and the
# next starting.
DEFAULT_POLL_INTERVAL = 5

# Values a state handler returns to the monitor loop (compared by identity).
STOP = object()        # monitoring is finished; monitor() returns
SKIP_DELAY = object()  # go straight round the loop without the polling delay

# How long a monitor waiting on SerialEM may see nothing new arrive before it
# stops and says so (--idle-timeout). 0 = never, as in 3.4.10 and earlier. A
# backstop, not a schedule: a SerialEM abort is read from the logs directly
# (see REyes_serialem.AbortWatch); this only catches what that cannot, such as
# a microscope PC that stopped mid-stage.
DEFAULT_IDLE_TIMEOUT = 0

# Entries of a session root that the MONITOR writes. Their changes are not
# arrivals from the microscope, so they never reset the idle clock. The
# default of `REyesProcessor.MONITOR_OWNED`.
MONITOR_OWNED = (
    'REyes_logs', 'dif_maps', 'targets', 'reyes_temp', 'grid_squares',
    'session_status.json', '.DS_Store',
)


class MonitorTerminated(KeyboardInterrupt):
    """SIGTERM, raised where the monitor was so its `finally` still runs."""


def install_sigterm_handler(processor):
    """Make a SIGTERM (e.g. `kill`) leave a verdict on disk.

    Python's default for SIGTERM is to exit on the spot, skipping every
    `finally`, so session_status.json would never be written by exactly the
    runs that most need one. Raising here instead unwinds through monitor(),
    whose `finally` writes `interrupted` with the reason.
    """
    def _on_sigterm(signum, frame):
        state = getattr(getattr(processor, 'state', None), 'name', '?')
        processor.stop_reason = f'SIGTERM received in state {state}'
        raise MonitorTerminated(processor.stop_reason)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError):
        # Not the main thread, or a platform without SIGTERM: nothing to do.
        return False
    return True


def _kill_process_tree_windows(pid):
    """Kill a process and its children on Windows."""
    run(['taskkill', '/F', '/T', '/PID', str(pid)],
        stdout=DEVNULL, stderr=DEVNULL, timeout=30)


def _kill_process_group(process):
    """Kill a child and everything it spawned.

    `xds_par` spawns worker processes. Killing only the direct child would leave
    them running, so on POSIX the child is started in its own session (making it
    a process-group leader) and the whole group is signalled here. Where process
    groups are not available (Windows), the process tree is killed with
    taskkill, and the direct child is always killed as well.
    """
    if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            # Already gone, or no permission to signal the group: fall back to
            # the direct child so the caller is never left blocked.
            pass
    elif os.name == 'nt':
        try:
            _kill_process_tree_windows(process.pid)
        except (OSError, SubprocessError):
            pass

    try:
        process.kill()
    except OSError:
        pass


def run_with_timeout(cmd, cwd=None, timeout=None):
    """subprocess.run(capture, text) that also reaps orphaned grandchildren.

    subprocess.run's own `timeout=` kills only the direct child, which is not
    enough here (see _kill_process_group). Raises TimeoutExpired on expiry,
    exactly as subprocess.run would, after the process group has been killed.

    Stages get MPLBACKEND=Agg: a monitor started over ssh -X has DISPLAY set,
    matplotlib may then pick TkAgg, and Tk can abort the stage on exit
    ("Tcl_AsyncDelete: async handler deleted by the wrong thread").
    """
    env = dict(os.environ, MPLBACKEND='Agg')
    process = Popen(cmd, cwd=cwd, stdout=PIPE, stderr=PIPE, text=True,
                    start_new_session=True, env=env)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except TimeoutExpired:
        _kill_process_group(process)
        process.communicate()
        raise
    return CompletedProcess(cmd, process.returncode, stdout, stderr)

class ProcessingState(Enum):
    WAITING_FOR_MONTAGE = auto()
    WAITING_FOR_GRID_SQUARES = auto()
    WAITING_FOR_MANUAL_SQUARES = auto()
    WAITING_FOR_DIFFRACTION_MAP = auto()
    GENERATING_TARGETS = auto()
    GENERATING_FINAL_MAP = auto()
    WAITING_FOR_MOVIES = auto()
    RUNNING_AUTOSOLVE = auto()
    COMPLETED = auto()

class CustomHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('States:'):
            return text.splitlines()
        return super()._split_lines(text, width)

__version__ = '3.6.1'
__min_required_version__ = '3.4.0'

# The movie log, and the logs SerialEM writes when the movie stage is re-entered
# (`<sample>_targets-1.log`, ...): the movies a resumed stage announces are as
# real as the first ones.
_MOVIE_LOG_RE = re.compile(r'.+_targets(?:-\d+)?\.log$')

# Waits on SerialEM in which an abort stops the monitor at once. The step-scan
# and movie waits handle an abort themselves, after finishing what has already
# arrived. The default of `REyesProcessor.abort_states`.
ABORT_STATES = frozenset({ProcessingState.WAITING_FOR_MONTAGE,
                          ProcessingState.WAITING_FOR_GRID_SQUARES})

# States in which the monitor waits on SerialEM, not on itself (--idle-timeout).
# The default of `REyesProcessor.idle_states`.
IDLE_STATES = frozenset({ProcessingState.WAITING_FOR_MONTAGE,
                         ProcessingState.WAITING_FOR_GRID_SQUARES,
                         ProcessingState.WAITING_FOR_DIFFRACTION_MAP,
                         ProcessingState.WAITING_FOR_MOVIES})

# The `record` of a block the monitor itself sets aside in set_aside_items.json
# (SerialEM's own set-asides carry the log record, e.g. BLOCK_SKIPPED).
MONITOR_SET_ASIDE = 'MONITOR_SET_ASIDE'

# Polls a block may stay unfinished, while a later block's log already exists,
# before it is set aside: logs written by the acquisition PC can appear out of
# order on a network share.
MISSING_BLOCK_POLLS = 3


class REyesProcessor:
    # Entries of the session root that this monitor writes itself; their
    # changes never reset the idle clock. A subclass that writes more entries
    # may extend the tuple.
    MONITOR_OWNED = MONITOR_OWNED

    def __init__(self,
             working_directory='.',
             microscope='Arctica-CETA',
             filtering='default',
             camera_length=None,
             manual_squares=False,
             start_state=None,
             current_block=1,
             top_target_per_category=2,
             top_target_per_block=None,
             target_lists=None,
             stepscan_only=False,
             montage_only=False,
             subprocess_timeout=DEFAULT_SUBPROCESS_TIMEOUT,
             movie_timeout=None,
             autosolve_timeout=None,
             poll_interval=DEFAULT_POLL_INTERVAL,
             idle_timeout=DEFAULT_IDLE_TIMEOUT,
             # Autoprocess parameters
             autoprocess=False,
             microscope_config=None,
             config_file=None,
             rotation_axis=None,
             frame_size=None,
             signal_pixel=None,
             min_pixel=None,
             background_pixel=None,
             pixel_size=None,
             wavelength=None,
             beam_center_x=None,
             beam_center_y=None,
             file_extension=None,
             detector_distance=None,
             exposure=None,
             rotation=None,
             pointless=False,
             parallel=False,
             dqa=False,
             verbose=False,
             res_range=None,
             friedel=None,
             # Autosolve parameters
             autosolve=False,
             shelx=None,
             ntryt=None,
             ntryf=None,
             space_group=None,
             atoms=None,
             atom_type=None,
             composition=None):

        # Setup logging
        log_file_path = os.path.join('REyes_logs', 'REyes_monitor.log')

        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        class FlushingStreamHandler(logging.StreamHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        console_handler = FlushingStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logging.basicConfig(
            level=logging.INFO,  # or INFO
            handlers=[file_handler, console_handler]
        )

        self.logger = logging.getLogger(__name__)
        self.working_directory = os.path.abspath(working_directory)
        self.subprocess_timeout = subprocess_timeout
        # AutoSolve can legitimately run for a long time, so it has no limit by
        # default. It is started in its own process group, so it can be killed
        # cleanly if a limit is set.
        self.autosolve_timeout = autosolve_timeout
        self.poll_interval = poll_interval
        self.processed_logs = set()
        # What the monitor writes in the session root.
        self.targets_dir = os.path.join(self.working_directory, 'targets')
        self.status_path = os.path.join(self.working_directory,
                                        'session_status.json')

        # --- the ways a session ends without its success phrases ----------- #
        # 0/None switches the idle backstop off.
        self.idle_timeout = idle_timeout or None
        self._idle_clock = None          # (signature, monotonic time, state)
        # The waits in which a SerialEM abort stops the monitor at once, and
        # those in which the idle backstop runs. The module constants are the
        # defaults; a subclass with more waiting states adds them here.
        self.abort_states = frozenset(ABORT_STATES)
        self.idle_states = frozenset(IDLE_STATES)
        # Set by a tick that has work in flight outside this process, so the
        # idle backstop does not end the session. Reset to False at the top of
        # every monitor tick.
        self.busy_elsewhere = False
        self._status_written = False
        self.stop_reason = None          # set by the SIGTERM handler
        # An abort counts until SerialEM demonstrably resumed after it (a newer
        # log of this grid; see REyes_serialem.AbortWatch). Its age relative to
        # this monitor does not matter: a session processed after the fact, or
        # a monitor started late, must still end with SerialEM's verdict, and a
        # monitor restarted before SerialEM resumes a grid stops with that
        # verdict -- restart it once the resume has begun.
        self._abort_watch = serialem.AbortWatch(self.working_directory)
        self.serialem_abort = None       # what AbortWatch found, once found
        # Step-scan blocks SerialEM set aside (valve, error), and those the
        # monitor gave up on because SerialEM had moved past them.
        self.skipped_blocks = {}
        self.completion_detail = 'movies processed'
        self.processed_blocks = []
        # A monitor restarted at a later block (--current-block) did not map
        # the earlier blocks itself; the map an earlier run wrote says they were.
        self._mapped_before_start = current_block > 1 and os.path.isfile(
            os.path.join(self.working_directory, 'dif_maps', 'dif_map_sums.csv'))
        self._missing_block_ticks = 0
        self.state = ProcessingState[start_state] if start_state else ProcessingState.WAITING_FOR_MONTAGE

        self.nav_pattern = re.compile(r'(.+)_(\d+)LM_(\d+)x(\d+)_nav\.nav$')
        
        self.current_block = current_block
        self.microscope = microscope
        self.filtering = filtering
        self.manual_squares = manual_squares
        self.camera_length = camera_length
        self.top_target_per_category = top_target_per_category
        self.top_target_per_block = top_target_per_block
        self.target_lists = target_lists if target_lists else ['sum', 'peaks', 'dqi']

        self.stepscan_only = stepscan_only
        self.montage_only = montage_only

        self.autoprocess = autoprocess
        # Save autoprocess arguments individually
        if autoprocess:
            self.microscope_config = microscope_config
            self.config_file = config_file
            self.rotation_axis = rotation_axis
            self.frame_size = frame_size
            self.signal_pixel = signal_pixel
            self.min_pixel = min_pixel
            self.background_pixel = background_pixel
            self.pixel_size = pixel_size
            self.wavelength = wavelength
            self.beam_center_x = beam_center_x
            self.beam_center_y = beam_center_y
            self.file_extension = file_extension
            self.detector_distance = detector_distance
            self.exposure = exposure
            self.rotation = rotation
            self.pointless = pointless
            self.parallel = parallel
            self.dqa = dqa
            self.verbose = verbose
            self.res_range = res_range
            self.friedel = friedel


        self.autosolve = autosolve

        if autosolve:
            self.shelx = shelx
            self.ntryt = ntryt
            self.ntryf = ntryf
            self.space_group = space_group
            self.atoms = atoms
            self.atom_type = atom_type
            self.composition = composition

        
        self.processed_movies = set()
        self.movies_count = 0
        # Autoprocess of one movie has its own wall-clock limit, so it can be
        # set apart from the stages' --subprocess-timeout. None = the same
        # limit as the stages, as in 3.4.10 and earlier.
        self.movie_timeout = (movie_timeout if movie_timeout is not None
                              else subprocess_timeout)
        self.movies_directory = os.path.join(self.working_directory, 'movies')
        
        self.logger.info(f"Initializing with state: {self.state.name}")
        self.logger.info(f"Current block: {self.current_block}")
    
    def module_commands(self):
        """The stages this monitor can run, by name. Returns a new dict.

        Maps a stage name, as `run_module` takes it, to the argv prefix that
        starts it (the console-script name, possibly with fixed arguments).
        A subclass may override this to add stages.
        """
        return {
            'grid_squares': ['grid-squares-0'],
            'manual_squares': ['manual-squares-0-1'],
            'eucentricity': ['eucentricity-1'],
            'dif_map': ['dif-map-2'],
            'write_targets': ['write-targets-3'],
            'append_targets': ['append-targets-3-1'],
            'create_final_targets': ['create-final-targets-4'],
            'mnt_maps_targets': ['mnt-maps-targets-5'],
        }

    def run_module(self, module_name, args=None):
        """Run a package module as a subprocess command with arguments.

        True when the stage exited 0; `run_module_code` gives the code itself.
        """
        return self.run_module_code(module_name, args) == 0

    def run_module_code(self, module_name, args=(), positional=()):
        """Run a stage as a subprocess and return its exit status.

        `args` is a mapping of option name to value: a True flag becomes
        `--name`, a list `--name v1 v2 ...`, anything else `--name value`
        (underscores in the name become dashes). `positional` arguments go
        after the options. The result is the subprocess's return code; a
        stage that cannot run at all -- a name `module_commands` does not
        know, a timeout (its process group is killed) or an exception while
        starting it -- gives 1. The base class gives no exit code a special
        meaning.
        """
        try:
            self.logger.info(f"Running module: {module_name}")

            module_commands = self.module_commands()
            if module_name not in module_commands:
                self.logger.error(f"Module {module_name} not found")
                return 1

            # Construct the command with the entry point name
            cmd = list(module_commands[module_name])
            
            # Add arguments if provided
            if args:
                for key, value in args.items():
                    if isinstance(value, bool):
                        if value:  # Only add flag if True
                            cmd.append(f"--{key.replace('_', '-')}")
                    elif isinstance(value, list):
                        # Handle list arguments (e.g., --target-lists sum peaks dqi)
                        cmd.append(f"--{key.replace('_', '-')}")
                        cmd.extend([str(v) for v in value])
                    else:
                        cmd.append(f"--{key.replace('_', '-')}")
                        cmd.append(str(value))

            if positional:
                cmd.extend(str(v) for v in positional)

            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command and capture output
            result = run_with_timeout(cmd, cwd=self.working_directory,
                                      timeout=self.subprocess_timeout)

            # Log the complete output to the main log file
            if result.stdout:
                self.logger.info(f"Process output:\n{result.stdout}")

            # Log stderr as well, but as a warning or error
            if result.stderr:
                self.logger.warning(f"Process stderr:\n{result.stderr}")

            if result.returncode == 0:
                self.logger.info(f"Successfully executed {module_name}")
            else:
                self.logger.error(f"{module_name} failed with return code {result.returncode}")
            return result.returncode

        except TimeoutExpired:
            self.logger.error(
                f"{module_name} exceeded the {self.subprocess_timeout}s timeout "
                f"and its process group was killed"
            )
            return 1
        except Exception as e:
            self.logger.error(f"Error running module {module_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1

    def find_nav_files(self):
        """Find .nav files and extract their information"""
        nav_files = [f for f in os.listdir(self.working_directory) if f.endswith('_nav.nav')]
        nav_info = []
        
        for nav_file in nav_files:
            match = self.nav_pattern.match(nav_file)
            if match:
                prefix, lm_number, x_dim, y_dim = match.groups()
                nav_info.append({
                    'nav_file': nav_file,
                    'prefix': prefix,
                    'lm_number': lm_number,
                    'dimensions': f"{x_dim}x{y_dim}"
                })
                self.logger.debug(f"Found .nav file: {nav_file} with prefix {prefix}, "
                                f"LM number {lm_number}, dimensions {x_dim}x{y_dim}")
        
        return nav_info

    def find_montage_logs(self):
        """Find montage log files based on .nav file information"""
        nav_info_list = self.find_nav_files()
        montage_logs = []
        
        for nav_info in nav_info_list:
            # Look for corresponding log file with the same prefix and dimensions
            expected_log = f"{nav_info['prefix']}_{nav_info['lm_number']}LM_{nav_info['dimensions']}.log"
            if (os.path.exists(os.path.join(self.working_directory, expected_log)) and 
                expected_log not in self.processed_logs):
                montage_logs.append(expected_log)
                self.logger.debug(f"Found matching montage log file: {expected_log}")
        
        return montage_logs
    
    def find_grid_squares_logs(self):
        """Find grid squares log files"""
        return [f for f in os.listdir(self.working_directory) 
                if f.endswith('_grid_squares.log') and 
                f not in self.processed_logs]
    
    def find_diffraction_map_logs(self, nav_info):
        """The current block's log, if it has not been dealt with yet.

        A square mapped again after a resume has a newer log beside the
        original (`<prefix>_<N>-1.log`, ...); the newest one is the block's log.
        """
        block_log = self._block_logs(nav_info['prefix']).get(self.current_block)
        if block_log and block_log not in self.processed_logs:
            return [block_log]
        return []
    
    def find_movie_logs(self):
        """The movie logs, originals before their resume logs.

        `<sample>_targets.log`, then `<sample>_targets-1.log`, ... -- a resumed
        movie stage writes a new log, and the movies it announces are as real
        as the first ones.
        """
        movie_logs = [f for f in os.listdir(self.working_directory)
                      if _MOVIE_LOG_RE.match(f) and
                      f not in self.processed_logs]
        return serialem.ordered_logs(movie_logs)
    
    def find_new_movies(self, log_file):
        """Find newly acquired movies from the targets log file"""
        try:
            new_movies = []
            current_folder = None
            current_movie = None
            
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Extract movie folder information
                    if 'Movie folder:' in line:
                        current_folder = line.split('Movie folder:')[-1].strip()
                        
                    # Extract acquired movie name
                    elif 'Acquired movie:' in line:
                        current_movie = line.split('Acquired movie:')[-1].strip()
                        
                        # If we have both folder and movie name, create movie info
                        if current_folder and current_movie:
                            movie_info = {
                                'folder': current_folder,
                                'acquired_name': current_movie,
                                'suffixes': ['_integrated_movie.mrc', '_0_movie.mrc']  # Support both filename patterns
                            }
                            
                            # Create a unique identifier for tracking processed movies
                            movie_id = f"{current_folder}/{current_movie}"
                            
                            if movie_id not in self.processed_movies:
                                new_movies.append(movie_info)
                            
                            # Reset for next movie
                            current_folder = None
                            current_movie = None
                            
            return new_movies
        
        except Exception as e:
            self.logger.error(f"Error reading movie log file {log_file}: {str(e)}")
            return []

    def check_montage_completion(self, log_file):
        """Check if montage is complete"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the LM number from the filename
                match = re.search(r'_(\d+)LM_', log_file)
                if match:
                    expected_number = match.group(1)
                    completion_pattern = f'completed {expected_number}x lm montage data collection'
                    if completion_pattern.lower() in content.lower():
                        self.logger.info(f"Montage completion detected in {log_file}")
                        return True
        except Exception as e:
            self.logger.error(f"Error reading montage log file {log_file}: {str(e)}")
        return False

    def check_nav_file_ready(self, nav_file):
        """Check if nav file has more than one item"""
        try:
            with open(nav_file, 'r') as f:
                content = f.read()
                items = re.findall(r'\[Item\s*=\s*(\d+)\]', content)
                item_count = len([item for item in items if item != '1'])
                self.logger.debug(f"Found {item_count} additional items in nav file")
                return item_count > 0
        except Exception as e:
            self.logger.error(f"Error checking nav file {nav_file}: {str(e)}")
            return False
    
    def handle_manual_squares_state(self):
        """Handle the manual squares state processing"""
        start_time = time.time()
        timeout = 3600
        
        # Add clear initial message
        self.logger.info("Waiting for manual grid squares to be added...")
        self.logger.info("Please add grid squares through SerialEM")
        self.logger.info("Once all grid squares are added, initiate the next script")
        
        while time.time() - start_time < timeout:
            nav_info_list = self.find_nav_files()
            
            for nav_info in nav_info_list:
                nav_file = os.path.join(self.working_directory, nav_info['nav_file'])
                if self.check_nav_file_ready(nav_file):
                    self.logger.info("Additional grid squares detected in nav file")
                    if self.run_module('manual_squares', {}):
                        # Wait for log file to confirm completion
                        log_file = os.path.join(self.working_directory, 'REyes_logs', 'manual_squares.log')
                        if os.path.exists(log_file) and self.check_manual_squares_completion(log_file):
                            self.state = ProcessingState.WAITING_FOR_GRID_SQUARES
                            self.logger.info("Manual squares processing completed")
                            self.logger.info("Moving to eucentricity adjustment state")
                            return True
            time.sleep(self.poll_interval)
        self.logger.warning("Manual squares processing timed out")
        return False
    
    def check_manual_squares_completion(self, log_file):
        """Check if manual squares processing is complete"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'processing completed successfully!'.lower() in content.lower():
                    self.logger.info("Manual squares completion detected")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading manual squares log file: {str(e)}")
        return False

    def check_grid_squares_completion(self, log_file):
        """Check if grid squares processing is complete"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'completed fine eucentricity for all grid squares'.lower() in content.lower():
                    self.logger.info(f"Grid squares completion detected in {log_file}")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading grid squares log file {log_file}: {str(e)}")
        return False
    
    def check_diffraction_map_completion(self, log_file):
        """Check if diffraction map processing is complete for current block"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'completed stepscan for current grid square'.lower() in content.lower():
                    self.logger.info(f"Diffraction map block {self.current_block} completion detected in {log_file}")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading diffraction map log file {log_file}: {str(e)}")
        return False
    
    def check_all_maps_completion(self, log_file):
        """Check if all diffraction map blocks are complete"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'completed stepscan for all grid squares'.lower() in content.lower():
                    self.logger.info("Diffraction mapping completed")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading diffraction map log file {log_file}: {str(e)}")
        return False
    
    def check_targets_completion(self, log_file):
        """Check diffraction movies complete"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'completed movie acquisition'.lower() in content.lower():
                    self.logger.info("Diffraction movie completion detected")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading diffraction map log file {log_file}: {str(e)}")
        return False

    def check_all_targets_completion(self, log_file):
        """Check all diffraction movies complete"""
        try:
            with open(os.path.join(self.working_directory, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                if 'completed acquiring tilt movies for all crystals!'.lower() in content.lower():
                    self.logger.info("All diffraction movies completion detected")
                    return True
        except Exception as e:
            self.logger.error(f"Error reading diffraction map log file {log_file}: {str(e)}")
        return False
    
    def process_movie(self, movie_info):
        def find_movie_file(root_dir, target_name, suffixes):
            for dirpath, _, filenames in os.walk(root_dir):
                for f in filenames:
                    # Check if file matches target name exactly
                    if f == target_name:
                        return os.path.join(dirpath, f)
                    # Check if file ends with any of the supported suffixes
                    for suffix in suffixes:
                        if f.endswith(suffix):
                            return os.path.join(dirpath, f)
            return None

        """Process a single movie using autoprocess.py"""
        try:
            # Construct the full path to the movie directory
            movie_dir = os.path.join(self.movies_directory, movie_info['folder'])

            # Skip movies already finished in an earlier run. `processed_movies`
            # is in memory only, so a restarted monitor re-reads every movie
            # from the targets log.
            if self._movie_already_done(movie_dir, movie_info['acquired_name']):
                self.logger.info(
                    f"Movie {movie_info['acquired_name']} was already processed "
                    f"in {movie_dir}; skipping.")
                self.processed_movies.add(
                    f"{movie_info['folder']}/{movie_info['acquired_name']}")
                # Still counts toward the session total.
                self.movies_count += 1
                return True

            # Recursively search for the movie file
            movie_path = find_movie_file(movie_dir, movie_info['acquired_name'], movie_info['suffixes'])
            if not movie_path:
                self.logger.error(f"No integrated movie file found in directory or subdirectories: {movie_dir}")
                return False

            target_path = os.path.join(movie_dir, movie_info['acquired_name'])

            # Rename if needed
            if movie_path != target_path:
                try:
                    # Determine which suffix the original file has
                    original_suffix = None
                    for suffix in movie_info['suffixes']:
                        if movie_path.endswith(suffix):
                            original_suffix = suffix
                            break

                    if original_suffix:
                        # Get the base name (without suffix)
                        base_name = os.path.basename(movie_path)[:-len(original_suffix)]
                        source_dir = os.path.dirname(movie_path)

                        # Rename the main .mrc file
                        self.logger.info(f"Renaming {movie_path} → {target_path}")
                        os.rename(movie_path, target_path)
                        self.logger.info(f"Successfully renamed file to {movie_info['acquired_name']}")

                        # Rename associated .mrc.mdoc file if it exists
                        mdoc_source = os.path.join(source_dir, f"{base_name}{original_suffix}.mdoc")
                        mdoc_target = os.path.join(movie_dir, f"{movie_info['acquired_name']}.mdoc")
                        if os.path.exists(mdoc_source):
                            self.logger.info(f"Renaming {mdoc_source} → {mdoc_target}")
                            os.rename(mdoc_source, mdoc_target)
                            self.logger.info("Successfully renamed .mdoc file")

                        # Rename associated .txt file if it exists
                        # Check for both possible txt file patterns: {base}_info.txt and {base}.txt
                        txt_source_info = os.path.join(source_dir, f"{base_name}_info.txt")
                        txt_source_plain = os.path.join(source_dir, f"{base_name}{original_suffix[:-4]}.txt")  # Remove .mrc extension
                        txt_target = os.path.join(movie_dir, f"{movie_info['acquired_name'][:-4]}.txt")

                        if os.path.exists(txt_source_info):
                            self.logger.info(f"Renaming {txt_source_info} → {txt_target}")
                            os.rename(txt_source_info, txt_target)
                            self.logger.info("Successfully renamed _info.txt file")
                        elif os.path.exists(txt_source_plain):
                            self.logger.info(f"Renaming {txt_source_plain} → {txt_target}")
                            os.rename(txt_source_plain, txt_target)
                            self.logger.info("Successfully renamed .txt file")
                    else:
                        # Fallback if no suffix matched
                        self.logger.info(f"Renaming {movie_path} → {target_path}")
                        os.rename(movie_path, target_path)
                        self.logger.info(f"Successfully renamed file to {movie_info['acquired_name']}")

                except Exception as e:
                    self.logger.error(f"Failed to rename movie file: {str(e)}")
                    return False

            # Create unique movie identifier for tracking
            movie_id = f"{movie_info['folder']}/{movie_info['acquired_name']}"
            self.processed_movies.add(movie_id)
            self.movies_count += 1
            
            if self.autoprocess:
                self.logger.info(f"Running autoprocess on movie {movie_info['acquired_name']}")
                
                cmd = ["autoprocess"]

                # Add autoprocess arguments
                def add_arg(flag, value):
                    """
                    Add an argument to the command list.

                    For boolean flags (store_true): only add if True
                    For value arguments: only add if not None
                    """
                    if isinstance(value, bool):
                        # For boolean flags, only add if True
                        if value:
                            cmd.append(f"--{flag}")
                    elif value is not None:
                        # For non-boolean arguments, add flag and value if not None
                        cmd.append(f"--{flag}")
                        cmd.append(str(value))

                add_arg("microscope-config", self.microscope_config)
                add_arg("config-file", self.config_file)
                add_arg("rotation-axis", self.rotation_axis)
                add_arg("frame-size", self.frame_size)
                add_arg("signal-pixel", self.signal_pixel)
                add_arg("min-pixel", self.min_pixel)
                add_arg("background-pixel", self.background_pixel)
                add_arg("pixel-size", self.pixel_size)
                add_arg("wavelength", self.wavelength)
                add_arg("beam-center-x", self.beam_center_x)
                add_arg("beam-center-y", self.beam_center_y)
                add_arg("file-extension", self.file_extension)
                add_arg("detector-distance", self.detector_distance)
                add_arg("exposure", self.exposure)
                add_arg("rotation", self.rotation)
                add_arg("pointless", self.pointless)
                add_arg("parallel", self.parallel)
                add_arg("dqa", self.dqa)
                add_arg("verbose", self.verbose)
                add_arg("res-range", self.res_range)
                add_arg("friedel", self.friedel)


                self.logger.info(f"Running autoprocess command: {' '.join(cmd)}")

                ok, note = run_for_movie(
                    cmd, movie_dir, self.movie_timeout, run_with_timeout,
                    logger=self.logger)
                try:
                    self.after_movie(movie_dir, ok)
                except Exception as exc:    # a hook never changes the verdict
                    self.logger.error(
                        f"after_movie failed for {movie_dir}: {exc}")
                if ok:
                    self.logger.info(
                        f"Successfully processed movie "
                        f"{movie_info['acquired_name']}")
                    return True
                self.logger.error(
                    f"Movie {movie_info['acquired_name']}: {note}")
                return False
            else:
                self.logger.info(f"Skipping autoprocess for movie {movie_info['acquired_name']} (autoprocess disabled)")
            
            self.logger.info(f"Successfully handled movie {self.movies_count}: {movie_info['acquired_name']}")
            return True
                        
        except Exception as e:
            self.logger.error(f"Error processing movie {movie_info['acquired_name']}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    # Files autoprocess leaves behind once it has run. Any one of them is
    # enough: which ones appear depends on how far XDS got, and a movie that
    # failed to index is still a processed movie. The check is "did it run at
    # all", not "did it succeed".
    AUTOPROCESS_ARTEFACTS = _AUTOPROCESS_ARTEFACTS

    # Written by the monitor in the movie folder once the movie is finished
    # with: after autoprocess succeeds, or when it is abandoned (its content
    # then says why).
    AUTOPROCESS_FLAG = _AUTOPROCESS_FLAG

    def after_movie(self, movie_dir, ok):
        """Called once a movie's autoprocess run has finished.

        `movie_dir` is the movie's dataset directory and `ok` whether the run
        succeeded (autoprocess exited 0 and left output). Runs after
        `run_for_movie` has returned -- so after `autoprocess_done.flag` has
        been written for a processed or an abandoned movie -- and only for
        movies processed by this monitor with --autoprocess, not for movies
        found already done. An exception raised here is logged and does not
        change the movie's result. The base class does nothing.
        """

    def _movie_already_done(self, movie_dir, acquired_name):
        """True if this movie has already been handled, judged from disk.

        With --autoprocess, a movie is done when autoprocess_done.flag exists
        in its folder. The monitor writes the flag only after autoprocess exits
        0 and its output is found on disk, or when the movie exceeded
        --movie-timeout. Without --autoprocess, the renamed movie file marks
        completion.

        XDS output files alone are not used as the marker: an unindexable movie
        never writes CORRECT.LP, and autoprocess_logs is created before XDS
        finishes, so neither distinguishes a finished movie from an interrupted
        one. No marker means the movie is processed again.
        """
        if not os.path.isdir(movie_dir):
            return False
        if self.autoprocess:
            return os.path.isfile(os.path.join(movie_dir,
                                               self.AUTOPROCESS_FLAG))
        return os.path.isfile(os.path.join(movie_dir, acquired_name))

    def _mark_movie_done(self, movie_dir, note='autoprocess complete'):
        """Record that autoprocess finished this movie, for a later restart."""
        if not _mark_done(movie_dir, note):
            # Not fatal: the movie is processed. Without the flag a restart
            # processes it again.
            self.logger.warning(f"could not write {self.AUTOPROCESS_FLAG} in "
                                f"{movie_dir}")

    def _autoprocess_produced_output(self, movie_dir):
        """True if autoprocess left anything behind in or under `movie_dir`."""
        return produced_output(movie_dir, self.AUTOPROCESS_ARTEFACTS)

    def run_autosolve(self):
        def stream_reader(pipe, log_func):
            for line in iter(pipe.readline, ''):
                log_func(line.strip())

        """Run the AutoSolve pipeline after REyes is completed."""
        try:
            autosolve_cmd = ["autosolve"]

            # Helper function to add arguments
            def add_arg(flag, value):
                if value is not None:
                    autosolve_cmd.append(f"--{flag}")
                    if not isinstance(value, bool):
                        autosolve_cmd.append(str(value))

            # Add all autosolve arguments
            add_arg("shelx", self.shelx)
            add_arg("ntryt", self.ntryt)
            add_arg("ntryf", self.ntryf)
            add_arg("space-group", self.space_group)
            add_arg("atoms", self.atoms)
            add_arg("type", self.atom_type)
            add_arg("composition", self.composition)

            self.logger.info(f"Running AutoSolve command: {' '.join(autosolve_cmd)}")

            process = Popen(autosolve_cmd, cwd=self.working_directory, stdout=PIPE, stderr=PIPE, text=True,
                            start_new_session=True)

            # Start threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, self.logger.info))
            stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, self.logger.warning))
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process and threads to complete (no limit by default)
            try:
                returncode = process.wait(timeout=self.autosolve_timeout)
            except TimeoutExpired:
                self.logger.error(
                    f"AutoSolve exceeded the {self.autosolve_timeout}s timeout "
                    f"and its process group was killed"
                )
                _kill_process_group(process)
                returncode = process.wait()
            stdout_thread.join()
            stderr_thread.join()

            if returncode == 0:
                self.logger.info("AutoSolve completed successfully!")
                return True
            else:
                self.logger.error(f"AutoSolve failed with return code {returncode}")
                return False

        except Exception as e:
            self.logger.error(f"Error running AutoSolve: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    # ---------------------------------------------------------------- states #
    # One method per ProcessingState. Each returns:
    #
    #   STOP        monitoring is finished; monitor() returns
    #   SKIP_DELAY  go straight round the loop, skipping the polling delay
    #   None        normal: wait for the polling delay

    def _handle_waiting_for_montage(self):
        self.logger.debug("Waiting for montage completion...")
        montage_logs = self.find_montage_logs()

        for log_file in montage_logs:
            if self.check_montage_completion(log_file):
                self.processed_logs.add(log_file)
                if self.run_module('grid_squares', {
                    'microscope': self.microscope,
                    'filtering': self.filtering
                }):
                    if self.montage_only:
                        self.completion_detail = 'montage only'
                        self.state = ProcessingState.COMPLETED
                        self.logger.info("REyes processing completed! (Montage only)")
                        return STOP
                    elif self.manual_squares:
                        self.state = ProcessingState.WAITING_FOR_MANUAL_SQUARES
                        self.logger.info("Moving to manual squares monitoring state")
                    else:
                        self.state = ProcessingState.WAITING_FOR_GRID_SQUARES
                        self.logger.info("Moving to grid squares monitoring state")
        return None

    def _handle_waiting_for_manual_squares(self):
        # Once squares have arrived, continue without the polling delay.
        if self.handle_manual_squares_state():
            return SKIP_DELAY
        time.sleep(self.poll_interval)  # Wait before checking again
        return None

    def _handle_waiting_for_grid_squares(self):
        self.logger.debug("Waiting for eucentricity adjustment...")
        grid_logs = self.find_grid_squares_logs()

        for log_file in grid_logs:
            if self.check_grid_squares_completion(log_file):
                self.processed_logs.add(log_file)
                if self.run_module('eucentricity', {}):
                    self.state = self.after_eucentricity_state()
                    if self.state is ProcessingState.WAITING_FOR_DIFFRACTION_MAP:
                        self.logger.info("Moving to diffraction map monitoring state")
                    else:
                        self.logger.info(f"Moving to {self.state.name}")
        return None

    def after_eucentricity_state(self):
        """The state the monitor enters once eucentricity has run.

        WAITING_FOR_DIFFRACTION_MAP: the step-scan follows. A subclass whose
        workflow continues differently after eucentricity returns its own
        state (one that `state_handlers` knows).
        """
        return ProcessingState.WAITING_FOR_DIFFRACTION_MAP

    # --------------------------------------------------------- step-scan #
    # A block advances on "completed stepscan for current grid square". A
    # block that never says it (the column valve closed mid-square, or any
    # error SerialEM set aside) used to hold the monitor at that block for
    # ever, and the blocks after it were never mapped. The rule: a failed block
    # loses only that block; the grid goes on to targets with whatever blocks
    # succeeded, even if only one did.
    def _log_text(self, log_file):
        return serialem.read_text(os.path.join(self.working_directory, log_file))

    def _block_logs(self, prefix):
        """{block number: its newest log} for `<prefix>_<N>.log` / `<prefix>_<N>-k.log`."""
        pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)(?:-\d+)?\.log$')
        found = {}
        for name in os.listdir(self.working_directory):
            m = pattern.match(name)
            if m:
                found.setdefault(int(m.group(1)), []).append(name)
        return {block: serialem.ordered_logs(names)[-1]
                for block, names in found.items()}

    def _block_log_numbers(self, prefix):
        return list(self._block_logs(prefix))

    @staticmethod
    def _block_folder(log_file):
        """The block's data folder: its log name without `.log` or a resume `-k`."""
        return re.sub(r'(?:-\d+)?\.log$', '', log_file)

    def _blocks_mapped(self):
        """True when at least one step-scan block has been mapped this session,
        by this monitor or by an earlier one (a restart with --current-block)."""
        return bool(self.processed_blocks) or self._mapped_before_start

    def _set_aside_block(self, block, reason, log_file, source='serialem',
                         record='BLOCK_SKIPPED'):
        """Record a block that will never be mapped, and say so once."""
        if block in self.skipped_blocks:
            return
        self.skipped_blocks[block] = {'reason': reason, 'log': log_file}
        self.logger.warning(f"BLOCK SET ASIDE | block={block} | reason={reason} "
                            f"| log={log_file} -- this block is lost, the grid "
                            f"goes on with the others")
        self._record_set_aside([serialem.make_entry(
            'STPSCN', block, reason, source, record, log=log_file)])

    def _record_set_aside(self, entries):
        try:
            return serialem.record_set_aside(self.targets_dir, entries)
        except OSError as exc:
            self.logger.error(f"could not update the set-aside list: {exc}")
            return []

    def _after_block(self, log_file):
        """A block is settled (mapped or set aside): next block, or the end."""
        self._missing_block_ticks = 0
        if self.check_all_maps_completion(log_file):
            self.state = ProcessingState.GENERATING_TARGETS
            self.logger.info("All diffraction map blocks completed, moving to target generation")
            return
        result = serialem.stpscn_result(self._log_text(log_file))
        if result is not None:
            self._end_of_stepscan(result, log_file)
            return
        self.current_block += 1
        self.logger.info(f"Moving to next block: {self.current_block}")

    def _end_of_stepscan(self, result, log_file):
        """SerialEM ended the step-scan without the all-squares phrase."""
        lost = sorted(self.skipped_blocks)
        if result == 'failed' or not self._blocks_mapped():
            return self._halt(
                f"The step-scan ended with no block mapped (result={result}, "
                f"{log_file}).",
                f"Blocks set aside: {lost or 'none recorded'}.",
                reason='stepscan_failed')
        self.logger.warning(
            f"STEP-SCAN ENDED {result.upper()} in {log_file}: "
            f"{len(self.processed_blocks)} block(s) mapped by this monitor, set aside: "
            f"{lost or 'none'}. Generating targets from the mapped blocks.")
        self.state = ProcessingState.GENERATING_TARGETS
        return None

    def _handle_waiting_for_diffraction_map(self):
        self.logger.debug(f"Waiting for diffraction map block {self.current_block} completion...")
        nav_info_list = self.find_nav_files()
        progressed = False

        for nav_info in nav_info_list:
            if self.state != ProcessingState.WAITING_FOR_DIFFRACTION_MAP:
                break
            diffraction_logs = self.find_diffraction_map_logs(nav_info)

            for log_file in diffraction_logs:
                text = self._log_text(log_file)
                skip = serialem.block_skip(text, self.current_block)
                # First check if current block is complete and process it
                if skip is None and self.check_diffraction_map_completion(log_file):
                    self.processed_logs.add(log_file)
                    folder_name = self._block_folder(log_file)
                    progressed = True

                    success = self.run_module('dif_map', {
                        'microscope': self.microscope,
                        'folder': folder_name,
                        'skip_processed': True,
                        'proc_blocks': 1
                    })

                    if success:
                        self.processed_blocks.append(self.current_block)
                    else:
                        self.logger.warning(f"Failed to process {folder_name}, skipping to next block")
                    self._after_block(log_file)

                elif skip is not None:
                    # SerialEM set this block aside (BLOCK_SKIPPED, or the
                    # older VALVE_CLOSED). Its frames were never copied:
                    # nothing to map.
                    self.processed_logs.add(log_file)
                    progressed = True
                    self._set_aside_block(self.current_block, skip[0], log_file,
                                          record=skip[1])
                    self._after_block(log_file)

                elif any(n > self.current_block for n in
                         self._block_log_numbers(nav_info['prefix'])):
                    # SerialEM has opened a later block's log, so this one is
                    # final -- and it never said the block finished. A few
                    # polls first (see MISSING_BLOCK_POLLS).
                    self._missing_block_ticks += 1
                    if self._missing_block_ticks < MISSING_BLOCK_POLLS:
                        continue
                    self._missing_block_ticks = 0
                    self.processed_logs.add(log_file)
                    progressed = True
                    self._set_aside_block(self.current_block,
                                          'no_completion_phrase', log_file,
                                          source='pyreyes', record=MONITOR_SET_ASIDE)
                    self._after_block(log_file)

            if not diffraction_logs and \
                    self.state == ProcessingState.WAITING_FOR_DIFFRACTION_MAP:
                outcome = self._current_block_absent(nav_info)
                if outcome is STOP:
                    return STOP
                progressed = progressed or outcome is SKIP_DELAY

        if self.state == ProcessingState.WAITING_FOR_DIFFRACTION_MAP and not progressed:
            found = self._check_serialem_abort()
            if found:
                if self._blocks_mapped():
                    self.logger.warning(
                        f"{serialem.describe_abort(found)}. Generating targets "
                        f"and maps from the mapped block(s); the session will "
                        f"end failed, since no movies will follow.")
                    self.state = ProcessingState.GENERATING_TARGETS
                    return SKIP_DELAY
                return self._halt(serialem.describe_abort(found),
                                  "No step-scan block was mapped.",
                                  reason='serialem_aborted')
        return None

    def _current_block_absent(self, nav_info):
        """The current block has no log (yet). Was it skipped, or was the last
        block before it the end of the stage?

        Read in the LATEST block log that exists (normally the previous block's):
        - `BLOCK_SKIPPED | square=<this block>` is written there when the square
          failed before opening its own log -- set aside, move on;
        - the save-order race: SerialEM writes "completed stepscan for current
          grid square" and the all-squares phrase in two saves; a monitor that
          read between them moved to a block that will never exist;
        - the stage ended early ("StepScan incomplete", or STPSCN_RESULT).
        And one robustness case: a LATER block's log exists while this one has
        none and nothing says why. After MISSING_BLOCK_POLLS polls it is set
        aside.
        """
        prefix = nav_info['prefix']
        logs = self._block_logs(prefix)
        numbers = list(logs)
        earlier = [n for n in numbers if n < self.current_block]
        if earlier:
            latest = logs[max(earlier)]
            text = self._log_text(latest)
            skip = serialem.block_skip(text, self.current_block)
            if skip is not None:
                self._missing_block_ticks = 0
                self._set_aside_block(self.current_block, skip[0], latest,
                                      record=skip[1])
                self._after_block(latest)
                return SKIP_DELAY
            if self.check_all_maps_completion(latest):
                self.logger.info(f"The all-squares phrase is in {latest}; "
                                 f"block {max(earlier)} was the last")
                self.state = ProcessingState.GENERATING_TARGETS
                self.logger.info("All diffraction map blocks completed, moving to target generation")
                return SKIP_DELAY
            result = serialem.stpscn_result(text)
            if result is not None:
                outcome = self._end_of_stepscan(result, latest)
                return STOP if outcome is STOP else SKIP_DELAY

        if any(n > self.current_block for n in numbers):
            self._missing_block_ticks += 1
            if self._missing_block_ticks >= MISSING_BLOCK_POLLS:
                self._missing_block_ticks = 0
                self._set_aside_block(self.current_block, 'no_log',
                                      f"{prefix}_{self.current_block}.log",
                                      source='pyreyes', record=MONITOR_SET_ASIDE)
                self.current_block += 1
                self.logger.info(f"Moving to next block: {self.current_block}")
                return SKIP_DELAY
        else:
            self._missing_block_ticks = 0
        return None

    def _handle_generating_targets(self):
        self.logger.debug("Generating target lists...")

        # Check if dif_map_sums.csv has any viable targets
        dif_map_csv = Path(self.working_directory) / "dif_maps" / "dif_map_sums.csv"

        if not dif_map_csv.exists():
            # A session with no map is a failure, not a completion; it is what
            # a grid whose every block was set aside looks like.
            return self._halt("Diffraction mapping results not found "
                              "(dif_maps/dif_map_sums.csv missing)",
                              reason='no_diffraction_map')

        # Check for viable targets (Good or Bad diffraction)
        try:
            df = pd.read_csv(dif_map_csv)
            viable_targets = df[df['DifQuality'].isin(['Good diffraction', 'Bad diffraction'])]

            if len(viable_targets) == 0:
                self.logger.info("=" * 80)
                self.logger.info("Step scan completed, but no viable microED targets were found.")
                self.logger.info("All positions were classified as: Grid, No diffraction, or Poor diffraction.")
                self.logger.info("Generating final maps without targets...")
                self.logger.info("=" * 80)
                # Skip target generation but still create final maps
                self.state = ProcessingState.GENERATING_FINAL_MAP
                self.logger.info("Moving to final map generation state (no targets)")

            else:
                self.logger.info(f"Found {len(viable_targets)} viable microED targets for collection")

                write_targets_args = {
                    'microscope': self.microscope
                }
                if self.camera_length is not None:
                    write_targets_args['camera_length'] = self.camera_length

                if self.run_module('write_targets', write_targets_args):
                    self.logger.info("All targets identified")
                    self.logger.debug("Generating spatialy aware targets navigator...")
                    create_targets_args = {
                        'top-target-per-category': self.top_target_per_category,
                        'target-lists': self.target_lists
                    }
                    if self.top_target_per_block is not None:
                        create_targets_args['top-target-per-block'] = self.top_target_per_block

                    if self.run_module('create_final_targets', create_targets_args):
                        self.state = ProcessingState.GENERATING_FINAL_MAP
                        self.logger.info("Moving to final map generation state")

        except Exception as e:
            return self._halt(f"Error checking diffraction results: {str(e)}",
                              reason='target_generation_error')
        return None

    def _handle_generating_final_map(self):
        self.logger.debug("Generating final diffraction map...")
        if self.run_module('mnt_maps_targets', {
                'microscope': self.microscope
            }):
            if self.stepscan_only:
                self.completion_detail = 'step-scan only'
                self.state = ProcessingState.COMPLETED
                self.logger.info("REyes processing completed! (Step scan only)")
                return STOP
            else:
                self.state = ProcessingState.WAITING_FOR_MOVIES
                self.logger.info("REyes processing completed!")
                self.logger.debug("Monitoring for new movies...")
        return None

    def _note_movie_set_asides(self, movie_logs):
        """Movies SerialEM set aside (ITEM_SKIPPED / VALVE_CLOSED, stage DFMV).

        Nothing waits on them -- a set-aside movie is never announced with
        `Movie folder:` / `Acquired movie:` -- so this is bookkeeping: say so
        once, and keep them in the set-aside list.
        """
        paths = [os.path.join(self.working_directory, f) for f in movie_logs]
        skipped = serialem.skipped_items(paths, ('DFMV',))
        entries = [serialem.make_entry('DFMV', item, info['reason'], 'serialem',
                                       info['record'], log=info['log'])
                   for item, info in sorted(skipped.items())]
        for entry in self._record_set_aside(entries):
            self.logger.warning(f"MOVIE SET ASIDE by SerialEM | item={entry['item']} "
                                f"| reason={entry['reason']} | log={entry.get('log')}")

    def _handle_waiting_for_movies(self):
        self.logger.debug("Waiting for the next tilt dataset...")
        movie_logs = self.find_movie_logs()

        # Every movie log first -- the original and any resume logs -- so a
        # resumed movie stage's movies are processed before the end phrase is
        # believed.
        processed_now = 0
        for log_file in movie_logs:
            for movie_info in self.find_new_movies(log_file):
                processed_now += 1
                if self.process_movie(movie_info):
                    self.logger.info(f"Successfully processed movie {self.movies_count}")
                else:
                    self.logger.error(f"Failed to process movie: {movie_info['acquired_name']}")

        if movie_logs:
            self._note_movie_set_asides(movie_logs)

        # The end phrase counts in the LATEST movie log: a resumed stage's log
        # supersedes the original's.
        if movie_logs and self.check_all_targets_completion(movie_logs[-1]):
            # Trigger phrase found, but do a final check for new movies
            # that may have been added to the log during processing
            self.logger.debug("Completion trigger found, performing final check for new movies...")
            final_check = [m for log_file in movie_logs
                           for m in self.find_new_movies(log_file)]

            if len(final_check) == 0:
                # No new movies AND trigger found - safe to exit
                if not self.movies_count:
                    # The movie macro prints the end phrase even when no movie
                    # was collected, and then fails the grid: say so.
                    self._check_serialem_abort()
                if self.autosolve:
                    self.state = ProcessingState.RUNNING_AUTOSOLVE
                    self.logger.info(f"All movies processed ({self.movies_count} total). Starting AutoSolve.")
                else:
                    self.state = ProcessingState.COMPLETED
                    self.logger.info(f"All movies processed ({self.movies_count} total). Sample to Indexing completed.")
                    return STOP
            else:
                # New movies found after completion trigger - continue processing
                self.logger.info(f"Completion trigger found, but {len(final_check)} new movie(s) detected. Continuing processing...")
            return None

        # No end phrase. If SerialEM has abandoned the grid, none will come:
        # finish what was announced, then stop and say why.
        if not processed_now:
            found = self._check_serialem_abort()
            if found:
                return self._halt(
                    serialem.describe_abort(found),
                    f"No more movies will come; {self.movies_count} movie(s) "
                    f"were processed.",
                    reason='serialem_aborted')
        return None

    def _handle_running_autosolve(self):
        self.logger.debug("Running AutoSolve...")
        if self.run_autosolve():
            self.state = ProcessingState.COMPLETED
            self.logger.info("AutoSolve completed. Marking processing as COMPLETED.")
        else:
            self.logger.error("AutoSolve failed. Still marking processing as COMPLETED.")
            self.state = ProcessingState.COMPLETED
        return STOP

    def state_handlers(self):
        """Map each ProcessingState to its handler."""
        return {
            ProcessingState.WAITING_FOR_MONTAGE: self._handle_waiting_for_montage,
            ProcessingState.WAITING_FOR_MANUAL_SQUARES: self._handle_waiting_for_manual_squares,
            ProcessingState.WAITING_FOR_GRID_SQUARES: self._handle_waiting_for_grid_squares,
            ProcessingState.WAITING_FOR_DIFFRACTION_MAP: self._handle_waiting_for_diffraction_map,
            ProcessingState.GENERATING_TARGETS: self._handle_generating_targets,
            ProcessingState.GENERATING_FINAL_MAP: self._handle_generating_final_map,
            ProcessingState.WAITING_FOR_MOVIES: self._handle_waiting_for_movies,
            ProcessingState.RUNNING_AUTOSOLVE: self._handle_running_autosolve,
        }

    # ------------------------------------------------------- how it ends #
    def _halt(self, *lines, reason=None):
        """Stop, saying why. A halt is a result, not a crash."""
        self.logger.error("=" * 80)
        for line in lines:
            self.logger.error(line)
        self.logger.error("=" * 80)
        self._write_session_status('failed', ' '.join(str(l) for l in lines),
                                   reason=reason)
        return STOP

    def _write_session_status(self, state, detail='', reason=None):
        """Record how this session ended, once, in session_status.json.

        Nothing else says a session is finished: the monitor's exit code is not
        visible to anything watching the session folder. So the last thing the
        monitor does is say what happened, in the session root, beside the
        data it describes.

        `state` is `complete`, `failed` or `interrupted`. `reason` is a short
        machine word (`serialem_aborted`, `stepscan_failed`, `idle_timeout`,
        `sigterm`, ...), added only when there is one.
        """
        if self._status_written:
            return
        path = self.status_path
        payload = self.status_payload(state, detail, reason)
        try:
            tmp = str(path) + '.tmp'
            with open(tmp, 'w', newline='\n') as f:
                json.dump(payload, f, indent=2)
                f.write('\n')
            os.replace(tmp, path)
            self._status_written = True
            self.logger.info(f"SESSION_STATUS | state={state} | {path}")
        except OSError as exc:
            self.logger.error(f"could not write session_status.json: {exc}")

    def status_payload(self, state, detail, reason=None):
        """The session_status.json document for a state. Override to add keys.

        `state`, `detail`, `utc` (ISO 8601, UTC) and, when given, `reason`.
        """
        payload = {'state': state, 'detail': detail,
                   'utc': datetime.now(timezone.utc).strftime(
                       '%Y-%m-%dT%H:%M:%SZ')}
        if reason:
            payload['reason'] = reason
        return payload

    def _completed_status(self):
        """The verdict of a monitor that has processed everything it got."""
        if self.serialem_abort:
            self._write_session_status(
                'failed', serialem.describe_abort(self.serialem_abort) +
                ' -- what arrived before it was processed',
                reason='serialem_aborted')
        else:
            self._write_session_status('complete', self.completion_detail)

    def _final_status(self):
        """The verdict for a monitor that ended without writing one."""
        if self._status_written:
            return
        if self.state == ProcessingState.COMPLETED:
            self._completed_status()
        elif self.stop_reason:
            self._write_session_status('interrupted', self.stop_reason,
                                       reason='sigterm')
        else:
            self._write_session_status(
                'interrupted', f'monitor stopped in state {self.state.name}')

    def _check_serialem_abort(self):
        """SerialEM's abort of THIS grid, read from its logs, or None.

        Remembered once found, so the final status can say it even when the
        monitor first finishes the blocks it already has.
        """
        try:
            found = self._abort_watch.check()
        except Exception as exc:          # never let the watch kill the monitor
            self.logger.warning(f"abort check failed: {exc}")
            return None
        if found and not self.serialem_abort:
            self.logger.error(f"SERIALEM_ABORT | reason={found['reason']} | "
                              f"log={found['log']} | {found['text']}")
        if found:
            self.serialem_abort = found
        return found

    def _arrival_signature(self):
        """(entries, newest mtime, log bytes) of what SerialEM wrote.

        None while the session holds nothing from the microscope -- a monitor
        started before its grid is not idle, it is early.
        """
        count, newest, log_bytes = 0, 0, 0
        try:
            entries = list(os.scandir(self.working_directory))
        except OSError:
            return None
        for entry in entries:
            name = entry.name
            if name in self.MONITOR_OWNED or name.startswith('.') or name.endswith('.tmp'):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            count += 1
            newest = max(newest, stat.st_mtime_ns)
            if name.lower().endswith('.log'):
                log_bytes += stat.st_size
        return (count, newest, log_bytes) if count else None

    def _idle_expired(self):
        """True (and the session ended) when SerialEM has gone quiet too long.

        Not while `busy_elsewhere` is set: a tick with work in flight outside
        this process is not idle, and the idle clock starts afresh after it.
        """
        if not self.idle_timeout or self.state not in self.idle_states \
                or self.busy_elsewhere:
            self._idle_clock = None
            return False
        signature = self._arrival_signature()
        if signature is None:
            self._idle_clock = None
            return False
        now = time.monotonic()
        if self._idle_clock is None or self._idle_clock[0] != signature \
                or self._idle_clock[2] != self.state:
            self._idle_clock = (signature, now, self.state)
            return False
        idle = now - self._idle_clock[1]
        if idle < self.idle_timeout:
            return False
        detail = (f"nothing new arrived from the microscope for {idle / 3600:.1f} h "
                  f"(--idle-timeout {self.idle_timeout:g} s) while in state "
                  f"{self.state.name}")
        self.logger.error("=" * 80)
        self.logger.error(f"IDLE TIMEOUT: {detail}")
        self.logger.error("=" * 80)
        self._write_session_status('interrupted', detail, reason='idle_timeout')
        return True

    def monitor(self):
        """Main monitoring loop with state machine.

        The session says how it ended, in `finally`, because the cases that
        most need a verdict on disk are the ones that do not reach a tidy
        ending: a cancelled job, a crash in a handler, an operator's Ctrl-C.
        """
        self.logger.info(f"Starting monitoring in working directory: {self.working_directory}")
        handlers = self.state_handlers()

        try:
            while True:
                handler = handlers.get(self.state)
                self.busy_elsewhere = False
                before = self.state
                # A state with no handler (COMPLETED) simply polls.
                outcome = handler() if handler is not None else None

                if outcome is STOP:
                    return
                if outcome is None and self.state == before \
                        and self.state in self.abort_states:
                    found = self._check_serialem_abort()
                    if found:
                        self._halt(serialem.describe_abort(found),
                                   f"The monitor was waiting in {before.name}; "
                                   f"nothing more will arrive for this grid.",
                                   reason='serialem_aborted')
                        return
                if self._idle_expired():
                    return
                if outcome is not SKIP_DELAY:
                    time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.logger.info("Shutting down monitor..."
                             + (f" ({self.stop_reason})" if self.stop_reason else ""))
        except Exception as e:
            self.logger.error(f"Unexpected error in monitor loop: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self._final_status()

def get_microscope_config_choices():
    """
    Get available microscope configurations from pyautoprocess.
    Falls back to hardcoded list if import fails.

    Returns:
        tuple: (list of config names, bool indicating if loaded from pyautoprocess)
    """
    # Fallback list matching pyautoprocess microscope_configs.json
    fallback_choices = ['Arctica-CETA-ser-SM', 'Arctica-CETA-mrc-SM', 'Arctica-EMcore-ser-SM',
                        'Talos-Apollo-SM', 'Talos-CETA-ser-SM', 'Talos-Apollo-P', 'default']

    try:
        import pyautoprocess
        import json

        # Get the package path
        package_path = pyautoprocess.__path__[0]
        config_file = os.path.join(package_path, 'data', 'microscope_configs.json')

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                configs = json.load(f)
                choices = sorted(list(configs.keys()))
                return choices, True
        else:
            return fallback_choices, False

    except ImportError:
        # pyautoprocess not installed - use fallback silently
        return fallback_choices, False
    except Exception as e:
        # pyautoprocess installed but has errors - log warning and use fallback
        logging.warning(f"Could not load microscope configs from pyautoprocess: {e}. Using default list.")
        return fallback_choices, False

def main():
    """
    Args:
        args (argparse.Namespace): Must contain:
            - working_directory (str): Path to working directory
            - microscope (str): Microscope type
            - filtering (str): Filtering type
    Returns:
        int: 0 on success, non-zero on failure
    """
    print_banner()
    parser = argparse.ArgumentParser(
        description='REyes Processing Monitor Settings',
        formatter_class=CustomHelpFormatter
    )

    state_descriptions = {
        'WAITING_FOR_MONTAGE': 'Wait for montage image acquisition',
        'WAITING_FOR_GRID_SQUARES': 'Wait for grid squares processing and eucentricity',
        'WAITING_FOR_MANUAL_SQUARES': 'Wait for manual grid squares selection',
        'WAITING_FOR_DIFFRACTION_MAP': 'Wait for diffraction mapping',
        'GENERATING_TARGETS': 'Generate target lists',
        'GENERATING_FINAL_MAP': 'Generate final diffraction atlas',
        'WAITING_FOR_MOVIES': 'Wait for movie acquisition',
        'RUNNING_AUTOSOLVE': 'Run AutoSolve for structure solution',
        'COMPLETED': 'Processing completed'
    }

    states_help_text = 'States:\n' + '\n'.join(f'  {state}: {desc}' 
                                              for state, desc in state_descriptions.items())
    
    # Create argument groups
    reyes_group = parser.add_argument_group('REyes arguments', 'Arguments for REyes processing')
    autoprocess_group = parser.add_argument_group('Autoprocess arguments', 'Arguments passed to autoprocess.py')
    autosolve_group = parser.add_argument_group('Autosolve arguments', 'Arguments passed to autosolve.py')

    # Microscope-specific arguments
    parser.add_argument('--microscope',
                       type=str,
                       choices=list(MICROSCOPE_CONFIGS.keys()),
                       default="Arctica-CETA",
                       help='Microscope configuration to use')
    
    # REyes-specific arguments
    reyes_group.add_argument('--filtering',
                            choices=['default', '1', '4', '9', '16', '25', '36', '49', '64', '81', '100', '121', '144', '169', '196', '96', 'None'],
                            default='default',
                            help='Filtering type to use (default: default)')
    reyes_group.add_argument('--manual-squares',
                            action='store_true',
                            help='Enable manual squares processing step')

    reyes_group.add_argument('--camera-length',
                            type=float,
                            default=None,
                            help='Optional override for camera length in mm (default: based on microscope config)')
    
    reyes_group.add_argument('--top-target-per-category',
                            type=int,
                            default=2,
                            help='Number of top targets to select per category (used by create-final-targets)')

    reyes_group.add_argument('--top-target-per-block',
                            type=int,
                            default=None,
                            help='Number of top targets to select per block (used by create-final-targets)')

    reyes_group.add_argument('--target-lists',
                            nargs='+',
                            choices=['sum', 'peaks', 'dqi'],
                            default=['sum', 'peaks', 'dqi'],
                            help='Which target lists to use for block-based selection: sum (intensity), peaks (diffraction peaks), dqi (quality). Can specify 1-3 values (default: all three). Applies only with --top-target-per-block; category-based selection always uses the nxds, quality, spots and sum navigation files')

    # Add state selection argument
    reyes_group.add_argument('--starting-state',
                            choices=[state.name for state in ProcessingState],
                            help=states_help_text)
    
    reyes_group.add_argument('--current-block',
                            type=int,
                            default=1,
                            help='Current block number (for diffraction map state)')
    
    reyes_group.add_argument('--stepscan-only',
                            action='store_true',
                            help='Do not collect tilt series movies for targets')

    reyes_group.add_argument('--montage-only',
                            action='store_true',
                            help='Process only the initial montage and grid squares, then exit')

    reyes_group.add_argument('--subprocess-timeout',
                            type=float,
                            default=DEFAULT_SUBPROCESS_TIMEOUT,
                            help=f'Seconds before a processing subprocess is treated as hung '
                                 f'and killed (default: {DEFAULT_SUBPROCESS_TIMEOUT:g}). A stall '
                                 f'detector, not a work limit')

    reyes_group.add_argument('--movie-timeout',
                            type=float,
                            default=None,
                            help='Seconds before autoprocess of ONE movie is abandoned, '
                                 'the movie marked as finished with, and the session '
                                 'moves on (default: the --subprocess-timeout value)')

    reyes_group.add_argument('--autosolve-timeout',
                            type=float,
                            default=None,
                            help='Seconds before AutoSolve is killed (default: no limit)')

    reyes_group.add_argument('--idle-timeout',
                            type=float,
                            default=DEFAULT_IDLE_TIMEOUT,
                            help=f'Seconds with nothing new from the microscope, while '
                                 f'waiting on SerialEM, before the monitor stops and '
                                 f'writes session_status.json "interrupted" '
                                 f'(default: {DEFAULT_IDLE_TIMEOUT:g} = never). A '
                                 f'backstop: a SerialEM abort is read from its logs')

    reyes_group.add_argument('--autoprocess',
                            action='store_true',
                            help='Run AutoProcess after movie acquisition')
    
    # Autoprocess arguments (parsed individually like autosolve)
    # Get microscope config choices and source info
    microscope_configs, is_dynamic = get_microscope_config_choices()
    config_source = "(from pyautoprocess)" if is_dynamic else "(fallback list)"

    autoprocess_group.add_argument('--microscope-config',
                                type=str,
                                choices=microscope_configs,
                                required='--autoprocess' in sys.argv,
                                help=f'Microscope configuration preset {config_source}')

    autoprocess_group.add_argument('--config-file',
                                type=str,
                                help='Path to microscope configuration file (default: microscope_configs.json)')

    autoprocess_group.add_argument('--rotation-axis',
                                type=str,
                                help='Rotation axis (e.g., "-1 0 0")')

    autoprocess_group.add_argument('--frame-size',
                                type=int,
                                help='Frame size of the movie (e.g., 2048)')

    autoprocess_group.add_argument('--signal-pixel',
                                type=int,
                                help='Signal pixel radius')

    autoprocess_group.add_argument('--min-pixel',
                                type=int,
                                help='Minimum pixel radius')

    autoprocess_group.add_argument('--background-pixel',
                                type=int,
                                help='Background pixel radius')

    autoprocess_group.add_argument('--pixel-size',
                                type=float,
                                help='Pixel size in microns (e.g., 0.028)')

    autoprocess_group.add_argument('--wavelength',
                                type=float,
                                help='Beam wavelength (e.g., 0.0251)')

    autoprocess_group.add_argument('--beam-center-x',
                                type=int,
                                help='Beam center X coordinate')

    autoprocess_group.add_argument('--beam-center-y',
                                type=int,
                                help='Beam center Y coordinate')

    autoprocess_group.add_argument('--file-extension',
                                type=str,
                                help='File extension for movies (e.g., ".mrc" or ".ser")')

    autoprocess_group.add_argument('--detector-distance',
                                type=float,
                                help='Detector distance in mm')

    autoprocess_group.add_argument('--exposure',
                                type=float,
                                help='Exposure time')

    autoprocess_group.add_argument('--rotation',
                                type=float,
                                help='Rotation value')

    autoprocess_group.add_argument('--pointless',
                                action='store_true',
                                help='Run pointless for space group analysis')

    autoprocess_group.add_argument('--parallel',
                                action='store_true',
                                help='Use parallel XDS (xds_par) instead of serial XDS')

    autoprocess_group.add_argument('--dqa',
                                action='store_true',
                                help='Enable diffraction quality analysis and frame selection')

    autoprocess_group.add_argument('--verbose',
                                action='store_true',
                                help='Enable verbose logging for detailed conversion validation')

    autoprocess_group.add_argument('--res-range',
                                type=str,
                                help='Manual resolution range in Angstroms (overrides calculated values)')

    autoprocess_group.add_argument('--friedel',
                                type=str,
                                help='Set Friedel\'s law for XDS (true or false, default: true)')

    autosolve_group.add_argument('--autosolve',
                                action='store_true',
                                help='Run AutoSolve after AutoProcess')

    autosolve_group.add_argument('--shelx',
                                choices=['t', 'd', 'td'],
                                default='td',
                                help='Which SHELX programs to run: t=SHELXT only, d=SHELXD only, td=both (default)')

    autosolve_group.add_argument('--ntryt',
                                type=int,
                                default=1000,
                                help='Number of trials for initial SHELXD screening (default: 1000)')

    autosolve_group.add_argument('--ntryf',
                                type=int,
                                default=10000,
                                help='Number of trials for final SHELXD runs (default: 10000)')

    autosolve_group.add_argument('--space-group',
                                type=str,
                                help='Target specific space group (1-230) or by name without spaces (e.g., P212121)')

    autosolve_group.add_argument('--atoms',
                                type=str,
                                help='Comma-separated list of atoms for SFAC (default: C,H,N,O,S). Examples: "C,H,N,O,P" or "C,H,O,Cl"')

    autosolve_group.add_argument('--type',
                                type=str,
                                choices=['organic', 'inorganic'],
                                dest='atom_type',
                                help='Type of compound (organic or inorganic)')

    autosolve_group.add_argument('--composition',
                                type=str,
                                help='Chemical formula (overrides --atoms). Examples: "CaTiO3", "Ca3Al2Si3O12", "NaCl"')
            
    args = parser.parse_args()

    # Validate autoprocess dependencies
    if args.autoprocess and not args.microscope_config:
        parser.error("--microscope-config is required when using --autoprocess")

    
   
    working_dir = os.path.abspath('.')

    processor_kwargs = dict(
        working_directory=working_dir,
        microscope=args.microscope,
        filtering=args.filtering,
        camera_length=args.camera_length,
        manual_squares=args.manual_squares,
        start_state=args.starting_state,
        current_block=args.current_block,
        autoprocess=args.autoprocess,
        top_target_per_category=args.top_target_per_category,
        top_target_per_block=args.top_target_per_block,
        target_lists=args.target_lists,
        stepscan_only=args.stepscan_only,
        montage_only=args.montage_only,
        subprocess_timeout=args.subprocess_timeout,
        movie_timeout=args.movie_timeout,
        autosolve_timeout=args.autosolve_timeout,
        idle_timeout=args.idle_timeout,
    )

    # Only include AutoProcess args if the flag is set
    if args.autoprocess:
        processor_kwargs.update({
            'microscope_config': args.microscope_config,
            'config_file': args.config_file,
            'rotation_axis': args.rotation_axis,
            'frame_size': args.frame_size,
            'signal_pixel': args.signal_pixel,
            'min_pixel': args.min_pixel,
            'background_pixel': args.background_pixel,
            'pixel_size': args.pixel_size,
            'wavelength': args.wavelength,
            'beam_center_x': args.beam_center_x,
            'beam_center_y': args.beam_center_y,
            'file_extension': args.file_extension,
            'detector_distance': args.detector_distance,
            'exposure': args.exposure,
            'rotation': args.rotation,
            'pointless': args.pointless,
            'parallel': args.parallel,
            'dqa': args.dqa,
            'verbose': args.verbose,
            'res_range': args.res_range,
            'friedel': args.friedel,
        })

    # Only include AutoSolve args if the flag is set
    if args.autosolve:
        processor_kwargs.update({
            'autosolve': True,
            'shelx': args.shelx,
            'ntryt': args.ntryt,
            'ntryf': args.ntryf,
            'space_group': args.space_group,
            'atoms': args.atoms,
            'atom_type': args.atom_type,
            'composition': args.composition,
        })

    processor = REyesProcessor(**processor_kwargs)
    install_sigterm_handler(processor)

    processor.monitor()
    if processor.stop_reason:
        # Stopped by SIGTERM: session_status.json says "interrupted"; the exit
        # status says so too, as a process killed by the signal would.
        return 128 + signal.SIGTERM
    return 0

if __name__ == "__main__":
    sys.exit(main())
