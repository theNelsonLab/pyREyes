"""
REyes MGSA (Manual Grid Squres Adder)
A tool to process NAV files for manually selected grid squares.
"""

import logging
import os
import re
import sys
from typing import Optional
import pickle
import numpy as np
from pathlib import Path
import shutil

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.image_processing.PlottingManager import PlottingManager
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs
from pyREyes.lib.REyes_utils import find_nav_file


__version__ = '3.4.0'
__min_required_version__ = '3.4.0'



def process_nav_file(input_file: str, output_dir: str) -> bool:
    """Process the NAV file and create grid_squares.nav."""
    try:
        output_file = os.path.join(output_dir, 'grid_squares.nav')
        item_pattern = re.compile(r'\[Item\s*=\s*(\d+)\]')
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Read first line (header) and write it
            first_line = infile.readline().strip()
            if not first_line:
                log_print("Input file is empty", logging.ERROR)
                return False
            
            # Write header and empty line
            outfile.write(first_line + '\n\n')
            
            # Skip content until we find second item block
            item_count = 0
            for line in infile:
                if item_pattern.match(line.strip()):
                    item_count += 1
                    if item_count == 2:
                        # Found second item - write it and all remaining content
                        outfile.write(line)  # Write the [Item = X] line
                        for remaining_line in infile:
                            outfile.write(remaining_line)
                        log_print(f"Successfully created {output_file}")
                        return True
            
            log_print("Could not find second item block in input file", logging.ERROR)
            return False
            
    except Exception as e:
        log_print(f"Error processing NAV file: {str(e)}", logging.ERROR)
        return False

def extract_stage_coordinates_from_nav(nav_path):
    coordinates = []
    with open(nav_path, 'r') as f:
        lines = f.readlines()

    item_pattern = re.compile(r"\[Item\s*=\s*(\d+)\]")
    stage_pattern = re.compile(r"StageXYZ\s*=\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")

    current_item = None
    for line in lines:
        item_match = item_pattern.match(line)
        if item_match:
            current_item = int(item_match.group(1))
        elif current_item is not None and 2 <= current_item <= 99:
            stage_match = stage_pattern.match(line)
            if stage_match:
                x = float(stage_match.group(1))
                y = float(stage_match.group(2))
                coordinates.append((x, y))
                current_item = None  # reset to avoid accidental reuse

    return coordinates

def plot_manual_selections(coords):
    try:
        # Load extent and montage image from temp dir
        temp_dir = Path.cwd() / "reyes_temp"
        extent = pickle.load(open(temp_dir / "extent.pkl", "rb"))
        image = np.load(temp_dir / "montage_image.npy")

        # Use default microscope config to init plotting manager
        microscope_config = load_microscope_configs()["Arctica-CETA"]
        plotter = PlottingManager(Path.cwd(), microscope_config)

        # Plot manually selected centroids
        plotter.plot_montage_from_image(image, extent, coords, save_path="grid_squares/manual_grid_overlay")

        log_print("Saved manual overlay plot as manual_grid_overlay.png")
    except Exception as e:
        log_print(f"Failed to plot manual selections: {e}", logging.ERROR)

def main() -> int:
    """Main execution function."""
    try:
        # Initialize logging and create output directory
        output_dir = 'REyes_logs'
        setup_logging('manual_squares.log', output_dir)
        print_banner()
        
        log_print("\nREyes NAV File Processor will help you create a grid squares file\n")
        
        # Find matching .nav file
        nav_file = find_nav_file()
        if not nav_file:
            return 1
        
        # Process the NAV file
        if not process_nav_file(nav_file, 'grid_squares'):
            log_print("Failed to process NAV file", logging.ERROR)
            return 1
        
        # Extract stage coordinates
        coordinates = extract_stage_coordinates_from_nav(nav_file)
        plot_manual_selections(coordinates)

        # Clean up temp directory
        temp_dir = Path.cwd() / "reyes_temp"
        if temp_dir.exists() and temp_dir.is_dir():
            shutil.rmtree(temp_dir)
            log_print("Temporary directory 'reyes_temp' deleted.")

        log_print("\nProcessing completed successfully!")
        log_print("Manual squares processing completed")
        return 0
        
    except KeyboardInterrupt:
        log_print("\nOperation cancelled by user", logging.WARNING)
        return 1
    except Exception as e:
        log_print(f"Processing failed: {str(e)}", logging.ERROR)
        logging.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())