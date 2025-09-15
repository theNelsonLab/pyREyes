"""
REyes STA (Selected Targets Adder)
A tool to append targets list with custom items, e.g. from nXDS.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import glob

import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs, MicroscopeConfig

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

MICROSCOPE_CONFIGS = load_microscope_configs()

class NavProcessingError(Exception):
    """Custom exception for NAV processing errors."""
    pass

def process_diffraction_data(
    targets_file: str,
    dif_map_file: str
) -> Optional[pd.DataFrame]:
    """Process target filenames and corresponding diffraction data."""
    try:
        if not os.path.exists(targets_file):
            log_print(f"Error: Targets file not found: {targets_file}", logging.ERROR)
            return None
            
        with open(targets_file, 'r') as f:
            targets = [line.strip() for line in f.readlines()]
            
        if not targets:
            log_print("Error: No target filenames found", logging.ERROR)
            return None
            
        df = pd.read_csv(dif_map_file)
        log_print(f"Found {len(targets)} targets to process")
        
        # Filter and rank entries
        quality_ranks = {
            'Good diffraction': 0,
            'Bad diffraction': 1,
            'Poor diffraction': 2,
            'No diffraction': 3,
            'Grid': 4
        }
        
        filtered_df = df[df['Path'].apply(lambda x: os.path.basename(x) in targets)].copy()
        filtered_df['QualityRank'] = filtered_df['DifQuality'].map(quality_ranks)
        
        sorted_df = filtered_df.sort_values(
            by=['QualityRank', 'FTPeaks', 'FilteredPeaks', 'Sum'],
            ascending=[True, False, False, False]
        )
        
        if sorted_df.empty:
            log_print("Error: No matching entries found in diffraction data", logging.ERROR)
            return None
            
        log_print(f"Successfully processed {len(sorted_df)} entries")
        return sorted_df
        
    except Exception as e:
        log_print(f"Error processing diffraction data: {str(e)}", logging.ERROR)
        return None

def generate_nav_entry(
    item_number: int,
    coordinates: Tuple[float, float, float],
    map_id: int
) -> str:
    """Generate a single NAV file entry."""
    stage_x, stage_y, stage_z = coordinates
    return f"""
[Item = {item_number}]
Color = 0
StageXYZ = {stage_x} {stage_y} {stage_z}
NumPts = 1
Regis = 1
Type = 0
RawStageXY = {stage_x} {stage_y}
MapID = {map_id}
PtsX = {stage_x}
PtsY = {stage_y}"""

def create_nav_file(
    df: pd.DataFrame,
    output_file: str,
    start_index: int,
    mode: str = 'new'
) -> bool:
    """Create or append to a NAV file."""
    try:
        entries = []
        for idx, row in enumerate(df.itertuples(), start=start_index):
            coordinates = eval(row.Coordinates)
            map_id = int(row.Sum)
            entry = generate_nav_entry(idx, coordinates, map_id)
            entries.append(entry)
            
        write_mode = 'w' if mode == 'new' else 'a'
        with open(output_file, write_mode) as f:
            if mode == 'new':
                f.write('AdocVersion = 2.00\n')
            else:
                f.write('\n')
            f.write('\n'.join(entries) + '\n')
            
        action = 'Created new' if mode == 'new' else 'Updated'
        log_print(f"{action} NAV file: {output_file}")
        return True
        
    except Exception as e:
        log_print(f"Error writing NAV file: {str(e)}", logging.ERROR)
        return False

def plot_diffraction_pattern(
    frame: np.ndarray,
    config: MicroscopeConfig,
    spots: int,
    sum_value: int,
    filename: str
) -> plt.Figure:
    """Create a diffraction pattern plot with resolution rings."""
    # Adjust pixel size based on image dimensions
    pixel_size = config.pixel_size
    if frame.shape[0] == 4096:
        pixel_size /= 2
    elif frame.shape[0] == 8192:
        pixel_size /= 4
    elif frame.shape[0] == 1024:
        pixel_size *= 2
        
    center = (frame.shape[0] / 2, frame.shape[1] / 2)
    max_radius = min(center[0], center[1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    vmin, vmax = np.percentile(frame, (1, 99))
    ax.imshow(frame, cmap='Greys', vmin=vmin, vmax=vmax)
    
    # Add resolution rings
    num_rings = 4
    ring_radii = np.linspace(max_radius / (num_rings + 1), max_radius, num_rings)
    
    for radius in ring_radii:
        circle = plt.Circle(
            center, radius,
            color='dimgrey',
            fill=False,
            linestyle='--'
        )
        ax.add_patch(circle)
        
        resolution = (config.wavelength * config.camera_length) / (radius * pixel_size)
        label_pos = (
            center[0] + radius / np.sqrt(2),
            center[1] - radius / np.sqrt(2)
        )
        ax.text(
            label_pos[0],
            label_pos[1],
            f"{resolution:.2f} Å",
            color='dimgrey',
            fontsize=10
        )
    
    ax.set_title(f"{filename} | Spots: {spots} | Sum: {sum_value}")
    ax.axis('off')
    return fig

def save_diffraction_snapshots(
    df: pd.DataFrame,
    output_folder: str,
    start_index: int,
    microscope_config: MicroscopeConfig
) -> None:
    """Save diffraction pattern snapshots for each entry."""
    os.makedirs(output_folder, exist_ok=True)
    log_print(f"\nGenerating diffraction snapshots in {output_folder}")
    
    for idx, row in enumerate(df.itertuples(), start=start_index):
        try:
            signal = hs.load(row.Path)
            spots = int(row.FilteredPeaks)
            sum_value = int(row.Sum)
            
            fig = plot_diffraction_pattern(
                signal.data,
                microscope_config,
                spots,
                sum_value,
                os.path.basename(row.Path)
            )
            
            output_filename = f"{idx}_{os.path.splitext(os.path.basename(row.Path))[0]}_sum{sum_value}.png"
            output_file = os.path.join(output_folder, output_filename)
            fig.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            log_print(f"Saved: {os.path.basename(output_file)}")
            
        except Exception as e:
            log_print(f"Error processing {row.Path}: {str(e)}", logging.ERROR)

def prompt_for_nav_file(create_nxds_nav: Optional[str] = None) -> Tuple[str, str, int]:
    """
    Determines whether to create a new .nav file or append to an existing one.
    
    Args:
        create_nxds_nav: Optional command-line answer ('yes' or 'no'). If None, prompts user.
    
    Returns:
        tuple: (nav_filename, snapshot_folder, start_item_number)
    """
    if create_nxds_nav is not None:
        answer = create_nxds_nav.lower()
    else:
        while True:
            answer = input("\nDo you want to create 'targets_nxds.nav'? Reply 'yes' or 'no': ").strip().lower()
            if answer in ('yes', 'no'):
                break
            log_print("I am sorry, I did not understand your answer. Please, enter 'yes' or 'no'.")

    if answer == 'yes':
        log_print("Proceeding with creating 'targets_nxds.nav'.")
        return 'targets_nxds.nav', 'targets_nxds_snapshots', 401
    else:
        log_print("OK. I will append the existing 'targets.nav' file.")
        return 'targets.nav', 'targets_diff_snapshots', 401

def create_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process diffraction data and generate NAV files'
    )
    
    parser.add_argument(
        '--targets',
        default='add_targets.txt',
        help='File containing target filenames'
    )
    parser.add_argument(
        '--microscope',
        default='Arctica-CETA',
        choices=MICROSCOPE_CONFIGS.keys(),
        help='Microscope configuration to use (default: Arctica-CETA)'
    )
    parser.add_argument(
        '--output-folder',
        help='Override default folder for diffraction snapshots'
    )
    parser.add_argument(
        '--camera-length',
        type=float,
        help='Override camera length in mm (default: based on microscope choice)'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        help='Override pixel size in mm/pixel for 2k images (default: based on microscope choice)'
    )
    parser.add_argument(
        '--create-nxds-nav',
        type=str,
        choices=['yes', 'no'],
        help='Create new targets_nxds.nav file (yes) or append to existing targets.nav (no)'
    )
    
    return parser

def main() -> int:
    """Main execution function."""
    try:
        parser = create_parser()
        args = parser.parse_args()

        setup_logging('append_targets.log', 'targets')
        print_banner()
        
        log_print(f"\nREyes STA v{__version__} will append .nav file with selected targets\n")
        log_print(f"Using {args.microscope} configuration")
        
        # Get microscope config and apply overrides if provided
        microscope_config = MICROSCOPE_CONFIGS[args.microscope]
        if args.camera_length is not None:
            log_print(f"Overriding camera length: {args.camera_length} mm")
            microscope_config.camera_length = args.camera_length
        if args.pixel_size is not None:
            log_print(f"Overriding pixel size: {args.pixel_size} mm/pixel")
            microscope_config.pixel_size = args.pixel_size
        
        # Get user preference for NAV file (from args or prompt)
        nav_file, default_output_folder, start_index = prompt_for_nav_file(args.create_nxds_nav)
        
        # Use command-line output folder if provided, otherwise use default
        output_folder = args.output_folder if args.output_folder else default_output_folder
        
        # Find diffraction data file
        dif_map_files = glob.glob('dif_map_sums*.csv')
        if not dif_map_files:
            log_print("No diffraction mapping data files found. Please run diffraction mapping first.", logging.ERROR)
            return 1
            
        # Process diffraction data
        df = process_diffraction_data(args.targets, dif_map_files[0])
        if df is None:
            return 1
            
        # Create or update NAV file based on user choice
        mode = 'new' if nav_file == 'targets_nxds.nav' else 'append'
        if not create_nav_file(df, nav_file, start_index, mode):
            return 1
            
        # Save diffraction snapshots
        save_diffraction_snapshots(
            df,
            output_folder,
            start_index,
            microscope_config
        )
        
        log_print("\nProcessing completed successfully!\n")
        return 0
        
    except Exception as e:
        log_print(f"An error occurred: {str(e)}", logging.ERROR)
        return 1

if __name__ == "__main__":
    sys.exit(main())