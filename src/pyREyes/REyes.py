"""
REyes Monitor
Monitoring tool for REyes processing scripts to automate the workflow.
"""

import os
import time
import logging
from subprocess import run, PIPE, Popen
import re
import sys
from enum import Enum, auto
import argparse
import shutil
import threading


from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_microscope_configurations import MicroscopeConfig, load_microscope_configs

MICROSCOPE_CONFIGS = load_microscope_configs()

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

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

class REyesProcessor:
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
             stepscan_only=False,
             # Autoprocess parameters
             autoprocess=False,
             microscope_config=None,
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
             # Autosolve parameters
             autosolve=False,
             shelx=None,
             ntryt=None,
             ntryf=None):

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
        self.processed_logs = set()
        self.state = ProcessingState[start_state] if start_state else ProcessingState.WAITING_FOR_MONTAGE

        self.nav_pattern = re.compile(r'(.+)_(\d+)LM_(\d+)x(\d+)_nav\.nav$')
        
        self.current_block = current_block
        self.microscope = microscope
        self.filtering = filtering
        self.manual_squares = manual_squares
        self.camera_length = camera_length
        self.top_target_per_category = top_target_per_category
        self.top_target_per_block = top_target_per_block

        self.stepscan_only = stepscan_only

        self.autoprocess = autoprocess
        # Save autoprocess arguments individually
        if autoprocess:
            self.microscope_config = microscope_config
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

        
        self.autosolve = autosolve

        if autosolve:
            self.shelx = shelx
            self.ntryt = ntryt
            self.ntryf = ntryf

        
        self.processed_movies = set()
        self.movies_count = 0
        self.movies_directory = os.path.join(self.working_directory, 'movies')
        
        self.logger.info(f"Initializing with state: {self.state.name}")
        self.logger.info(f"Current block: {self.current_block}")
    
    def run_module(self, module_name, args=None):
        """Run a package module as a subprocess command with arguments"""
        try:
            self.logger.info(f"Running module: {module_name}")
            
            # Map module names to their corresponding command line names
            module_commands = {
                'grid_squares': 'grid-squares-0',
                'manual_squares': 'manual-squares-0-1',
                'eucentricity': 'eucentricity-1',
                'dif_map': 'dif-map-2',
                'write_targets': 'write-targets-3',
                'append_targets': 'append-targets-3-1',
                'create_final_targets': 'create-final-targets-4',
                'mnt_maps_targets': 'mnt-maps-targets-5'
            }
            
            if module_name not in module_commands:
                self.logger.error(f"Module {module_name} not found")
                return False

            # Construct the command with the entry point name
            cmd = [module_commands[module_name]]
            
            # Add arguments if provided
            if args:
                for key, value in args.items():
                    if isinstance(value, bool):
                        if value:  # Only add flag if True
                            cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--{key.replace('_', '-')}")
                        cmd.append(str(value))

            self.logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command and capture output
            result = run(cmd, stdout=PIPE, stderr=PIPE, text=True, cwd=self.working_directory)
            
            # Log the complete output to the main log file
            if result.stdout:
                self.logger.info(f"Process output:\n{result.stdout}")
            
            # Log stderr as well, but as a warning or error
            if result.stderr:
                self.logger.warning(f"Process stderr:\n{result.stderr}")
            
            if result.returncode == 0:
                self.logger.info(f"Successfully executed {module_name}")
                return True
            else:
                self.logger.error(f"{module_name} failed with return code {result.returncode}")
                return False
                    
        except Exception as e:
            self.logger.error(f"Error running module {module_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

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
        """Find diffraction map log files for current block"""
        block_log = f"{nav_info['prefix']}_{self.current_block}.log"
        log_path = os.path.join(self.working_directory, block_log)
        
        if os.path.exists(log_path) and block_log not in self.processed_logs:
            return [block_log]
        return []
    
    def find_movie_logs(self):
        """Find movie acquisition log file"""
        movie_logs = [f for f in os.listdir(self.working_directory) 
                    if f.endswith('_targets.log') and 
                    f not in self.processed_logs]
        #self.logger.info(f"Found movie log files: {movie_logs}")
        return movie_logs
    
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
                                'suffix': '_integrated_movie.mrc'  # Generic filename suffix
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
            time.sleep(5)
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
        def find_movie_file(root_dir, target_name, suffix):
            for dirpath, _, filenames in os.walk(root_dir):
                for f in filenames:
                    if f == target_name or f.endswith(suffix):
                        return os.path.join(dirpath, f)
            return None

        """Process a single movie using autoprocess.py"""
        try:
            # Construct the full path to the movie directory
            movie_dir = os.path.join(self.movies_directory, movie_info['folder'])
            
            # Recursively search for the movie file
            movie_path = find_movie_file(movie_dir, movie_info['acquired_name'], movie_info['suffix'])
            if not movie_path:
                self.logger.error(f"No integrated movie file found in directory or subdirectories: {movie_dir}")
                return False

            target_path = os.path.join(movie_dir, movie_info['acquired_name'])

            # Rename if needed
            if movie_path != target_path:
                try:
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
                    if value is not None:
                        cmd.append(f"--{flag}")
                        if not isinstance(value, bool):
                            cmd.append(str(value))

                add_arg("microscope-config", self.microscope_config)
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

                
                self.logger.info(f"Running autoprocess command: {' '.join(cmd)}")
                
                # Run the command and capture output
                try:
                    result = run(cmd, stdout=PIPE, stderr=PIPE, text=True, cwd=movie_dir)
                    
                    # Print stdout directly without logger formatting
                    if result.stdout:
                        self.logger.info(f"Autoprocess stdout:\n{result.stdout}")
                    
                    # Print stderr directly without logger formatting
                    if result.stderr:
                        self.logger.warning(f"Autoprocess stderr:\n{result.stderr}")
                    
                    if result.returncode == 0:
                        self.logger.info(f"Successfully processed movie {movie_info['acquired_name']}")
                        return True
                    else:
                        self.logger.error(f"Autoprocess failed with return code {result.returncode}")
                        return False
                except Exception as e:
                    self.logger.error(f"Error running autoprocess: {str(e)}")
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

    def run_autosolve(self):
        def stream_reader(pipe, log_func):
            for line in iter(pipe.readline, ''):
                log_func(line.strip())

        """Run the AutoSolve pipeline after REyes is completed."""
        try:
            autosolve_cmd = [
                "autosolve",
                "--shelx", self.shelx,
                "--ntryt", str(self.ntryt),
                "--ntryf", str(self.ntryf)
            ]

            self.logger.info(f"Running AutoSolve command: {' '.join(autosolve_cmd)}")

            process = Popen(autosolve_cmd, cwd=self.working_directory, stdout=PIPE, stderr=PIPE, text=True)

            # Start threads to read stdout and stderr concurrently
            stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, self.logger.info))
            stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, self.logger.warning))
            stdout_thread.start()
            stderr_thread.start()

            # Wait for process and threads to complete
            returncode = process.wait()
            stdout_thread.join()
            stderr_thread.join()

            if returncode == 0:
                self.logger.info("AutoSolve completed successfully!")
                return True
            else:
                # Fixed: changed result.returncode to returncode
                self.logger.error(f"AutoSolve failed with return code {returncode}")
                return False

        except Exception as e:
            self.logger.error(f"Error running AutoSolve: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def monitor(self):
        """Main monitoring loop with state machine"""
        self.logger.info(f"Starting monitoring in working directory: {self.working_directory}")
        
        try:
            while True:
                if self.state == ProcessingState.WAITING_FOR_MONTAGE:
                    self.logger.debug("Waiting for montage completion...")
                    montage_logs = self.find_montage_logs()
                    
                    for log_file in montage_logs:
                        if self.check_montage_completion(log_file):
                            self.processed_logs.add(log_file)
                            if self.run_module('grid_squares', {
                                'microscope': self.microscope,
                                'filtering': self.filtering
                            }):
                                if self.manual_squares:
                                    self.state = ProcessingState.WAITING_FOR_MANUAL_SQUARES
                                    self.logger.info("Moving to manual squares monitoring state")
                                else:
                                    self.state = ProcessingState.WAITING_FOR_GRID_SQUARES
                                    self.logger.info("Moving to grid squares monitoring state")

                elif self.state == ProcessingState.WAITING_FOR_MANUAL_SQUARES:
                    if self.handle_manual_squares_state():
                        continue
                    time.sleep(5)  # Wait before checking again

                elif self.state == ProcessingState.WAITING_FOR_GRID_SQUARES:
                    self.logger.debug("Waiting for eucentricity adjustment...")
                    grid_logs = self.find_grid_squares_logs()
                    
                    for log_file in grid_logs:
                        if self.check_grid_squares_completion(log_file):
                            self.processed_logs.add(log_file)
                            if self.run_module('eucentricity', {}):
                                self.state = ProcessingState.WAITING_FOR_DIFFRACTION_MAP
                                self.logger.info("Moving to diffraction map monitoring state")
                                
                elif self.state == ProcessingState.WAITING_FOR_DIFFRACTION_MAP:
                    self.logger.debug(f"Waiting for diffraction map block {self.current_block} completion...")
                    nav_info_list = self.find_nav_files()

                    for nav_info in nav_info_list:
                        diffraction_logs = self.find_diffraction_map_logs(nav_info)

                        for log_file in diffraction_logs:
                            # First check if current block is complete and process it
                            if self.check_diffraction_map_completion(log_file):
                                self.processed_logs.add(log_file)
                                folder_name = log_file.replace(".log", "")

                                success = self.run_module('dif_map', {
                                    'microscope': self.microscope,
                                    'folder': folder_name,
                                    'skip_processed': True,
                                    'proc_blocks': 1
                                })

                                if success:
                                    if self.check_all_maps_completion(log_file):
                                        self.state = ProcessingState.GENERATING_TARGETS
                                        self.logger.info("All diffraction map blocks completed, moving to target generation")
                                    else:
                                        self.current_block += 1
                                        self.logger.info(f"Moving to next block: {self.current_block}")
                                else:
                                    self.logger.warning(f"Failed to process {folder_name}, skipping to next block")
                                    self.current_block += 1

                elif self.state == ProcessingState.GENERATING_TARGETS:
                    self.logger.debug("Generating target lists...")
                    write_targets_args = {
                        'microscope': self.microscope
                    }
                    if self.camera_length is not None:
                        write_targets_args['camera_length'] = self.camera_length

                    if self.run_module('write_targets', write_targets_args):
                        self.logger.info("All targets identified")
                        self.logger.debug("Generating spatialy aware targets navigator...")
                        create_targets_args = {
                            'top-target-per-category': self.top_target_per_category
                        }
                        if self.top_target_per_block is not None:
                            create_targets_args['top-target-per-block'] = self.top_target_per_block

                        if self.run_module('create_final_targets', create_targets_args):
                            self.state = ProcessingState.GENERATING_FINAL_MAP
                            self.logger.info("Moving to final map generation state")

                elif self.state == ProcessingState.GENERATING_FINAL_MAP:
                    self.logger.debug("Generating final diffraction map...")
                    if self.run_module('mnt_maps_targets', {
                            'microscope': self.microscope
                        }):
                        if self.stepscan_only:
                            self.state = ProcessingState.COMPLETED
                            self.logger.info("REyes processing completed! (Step scan only)")
                            return
                        else:
                            self.state = ProcessingState.WAITING_FOR_MOVIES
                            self.logger.info("REyes processing completed!")
                            self.logger.debug("Monitoring for new movies...")
                
                elif self.state == ProcessingState.WAITING_FOR_MOVIES:
                    self.logger.debug("Waiting for the next tilt dataset...")
                    movie_logs = self.find_movie_logs()
                    
                    for log_file in movie_logs:
                        # Find any new movies
                        new_movies = self.find_new_movies(log_file)
                        
                        # Process each new movie
                        for movie_info in new_movies:
                            if self.process_movie(movie_info):
                                self.logger.info(f"Successfully processed movie {self.movies_count}")
                            else:
                                self.logger.error(f"Failed to process movie: {movie_info['acquired_name']}")
                        
                        # Check if all movies are completed
                        if self.check_all_targets_completion(log_file):
                            if self.autosolve:
                                self.state = ProcessingState.RUNNING_AUTOSOLVE
                                self.logger.info(f"All movies processed ({self.movies_count} total). Starting AutoSolve.")
                            else:
                                self.state = ProcessingState.COMPLETED
                                self.logger.info(f"All movies processed ({self.movies_count} total). Sample to Indexing completed.")
                                return

                elif self.state == ProcessingState.RUNNING_AUTOSOLVE:
                    self.logger.debug("Running AutoSolve...")
                    if self.run_autosolve():
                        self.state = ProcessingState.COMPLETED
                        self.logger.info("AutoSolve completed. Marking processing as COMPLETED.")
                    else:
                        self.logger.error("AutoSolve failed. Still marking processing as COMPLETED.")
                        self.state = ProcessingState.COMPLETED
                    return


                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down monitor...")
        except Exception as e:
            self.logger.error(f"Unexpected error in monitor loop: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

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
    
    reyes_group.add_argument('--autoprocess',
                            action='store_true',
                            help='Run AutoProcess after movie acquisition')
    
    # Autoprocess arguments (parsed individually like autosolve)
    autoprocess_group.add_argument('--microscope-config',
                                type=str,
                                required='--autoprocess' in sys.argv,
                                help='Microscope config name (e.g., Arctica-CETA-mrc-SM)')

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

    autosolve_group.add_argument('--autosolve', 
                                action='store_true', 
                                help='Run AutoSolve after AutoProcess')

    autosolve_group.add_argument('--shelx', 
                                choices=['t', 'd', 'td'], 
                                default='td', 
                                help='Which SHELX programs to run (t: SHELXT, d: SHELXD, td: both)')
    
    autosolve_group.add_argument('--ntryt', 
                                type=int, 
                                default=1000, 
                                help='Number of trials for initial SHELXD screening (default: 1000)')
    
    autosolve_group.add_argument('--ntryf', 
                                type=int, 
                                default=10000, 
                                help='Number of trials for extended SHELXD runs (default: 10000)')
            
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
        stepscan_only=args.stepscan_only
    )

    # Only include AutoProcess args if the flag is set
    if args.autoprocess:
        processor_kwargs.update({
            'microscope_config': args.microscope_config,
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
        })

    # Only include AutoSolve args if the flag is set
    if args.autosolve:
        processor_kwargs.update({
            'autosolve': True,
            'shelx': args.shelx,
            'ntryt': args.ntryt,
            'ntryf': args.ntryf,
        })

    processor = REyesProcessor(**processor_kwargs)

    processor.monitor()
    return 0

if __name__ == "__main__":
    main()
