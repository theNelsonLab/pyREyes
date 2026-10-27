"""
REyes NFG (Navigation File Generator)
A tool for processing diffraction data and generating navigation files with quality metrics.
"""

import logging
import os
import sys
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # files only; never a GUI backend, even with X11 forwarding
import pandas as pd

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs
from pyREyes.lib.diffraction import beam_center_search
from pyREyes.lib.diffraction.DiffractionDataProcessor import DiffractionDataProcessor

__version__ = '3.6.2'
__min_required_version__ = '3.4.0'

MICROSCOPE_CONFIGS = load_microscope_configs()

def create_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='REyes NFG - Navigation File Generator for diffraction data processing'
    )
    parser.add_argument(
        '--microscope',
        type=str,
        choices=list(MICROSCOPE_CONFIGS.keys()),
        default="Arctica-CETA",
        help='Microscope configuration to use (default: Arctica-CETA)'
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
        '--beam-center',
        type=beam_center_search.parse_mode,
        default=beam_center_search.MODE_AUTO,
        help=beam_center_search.ARGUMENT_HELP
    )
    return parser

def main() -> int:
    """Main function to process all CSV files.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    
    
    try:
        parser = create_parser()
        args = parser.parse_args()

        setup_logging('targets_creation.log')
        print_banner()

        log_print(f"\nREyes NFG v{__version__} will generate individual .nav files\n")
        log_print(f"Using {args.microscope} configuration")
        
        processor = DiffractionDataProcessor(
            args.microscope,
            camera_length=args.camera_length,
            pixel_size=args.pixel_size,
            beam_center=args.beam_center
        )

        csv_files = glob.glob('targets/targets*.csv')
        if not csv_files:
            log_print("No CSV files found. Please process diffraction mapping data first.", logging.ERROR)
            return 1
        
        # Define output folder
        target_folder = Path("targets")
        target_folder.mkdir(parents=True, exist_ok=True)

        # One beam centre for every snapshot of the session, found from all
        # the frames that are about to be plotted rather than list by list
        snapshot_paths = []
        for csv_file in csv_files:
            try:
                snapshot_paths.extend(pd.read_csv(csv_file, usecols=['Path'])['Path'].astype(str))
            except Exception as e:
                log_print(f"Beam centre: no frame paths in {csv_file}: {str(e)}", logging.WARNING)
        processor.find_beam_center(snapshot_paths, str(target_folder))


        for csv_file in csv_files:
            try:
                # Determine start item number based on file type
                file_base = os.path.splitext(os.path.basename(csv_file))[0]
                start_item_number = {
                    'quality': 101,
                    'spots': 201,
                    'sum': 301
                }.get(file_base.split('_')[-1], 1)

                # Process CSV and generate nav file
                df = processor.process_csv(csv_file)
                
                # Add target_number column to DataFrame
                df['target_number'] = range(start_item_number, start_item_number + len(df))
                
                # Save updated CSV with target numbers
                df.to_csv(csv_file, index=False)
                log_print(f"Updated {csv_file} with target numbers")
                
                nav_entries = processor.generate_nav_entries(df, start_item_number)
                
                
                # Save nav file
                nav_file_old = os.path.splitext(csv_file)[0] + '.nav'
                nav_file = target_folder / (Path(csv_file).stem + '.nav')

                print(nav_file_old)
                print(nav_file)


                nav_content = 'AdocVersion = 2.00\n' + '\n'.join(nav_entries)
                with open(nav_file, 'w') as f:
                    f.write(nav_content)
                log_print(f"Created nav file: {nav_file}")

                # Create diffraction snapshots
                output_folder = os.path.splitext(csv_file)[0] + '_diff_snapshots'
                os.makedirs(output_folder, exist_ok=True)
                
                log_print(f"\nGenerating diffraction snapshots in {output_folder}")
                
                for index, row in enumerate(df.itertuples(), start=start_item_number):
                    mrc_path = row.Path
                    
                    if not os.path.exists(mrc_path):
                        log_print(f"MRC file not found: {mrc_path}", logging.ERROR)
                        continue

                    output_filename = f"{index}_{os.path.splitext(os.path.basename(mrc_path))[0]}_sum{int(row.Sum)}.png"
                    output_path = os.path.join(output_folder, output_filename)
                    processor.plot_diffraction_snapshot(
                        str(mrc_path),
                        str(output_path),
                        index,
                        row.FilteredPeaks,
                        row.Sum
                    )

            except Exception as e:
                log_print(f"Error processing {csv_file}: {str(e)}", logging.ERROR)
                continue

        log_print("\nProcessing completed successfully!\n")
        return 0

    except Exception as e:
        log_print(f"Fatal error: {str(e)}", logging.ERROR)
        return 1

if __name__ == "__main__":
    sys.exit(main())