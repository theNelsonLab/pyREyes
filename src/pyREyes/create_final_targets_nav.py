"""
REyes FTP (Final Targets Processor)
A tool for processing and combining multiple .nav files while
maintaining spatial separation between targets.
"""

import argparse
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.diffraction.DiffractionDataProcessor import DiffractionDataProcessor, DiffractionProcessingError
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs

# Configuration Constants
MICROSCOPE_CONFIGS = load_microscope_configs()

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

class NavFileError(Exception):
    """Custom exception for NAV file processing errors."""
    pass

def parse_nav_file_correctly(nav_file: str) -> List[str]:
    """
    Parses a .nav file to extract individual item entries.
    
    Args:
        nav_file (str): Path to the .nav file
        
    Returns:
        List[str]: List of item entries from the nav file
        
    Raises:
        NavFileError: If file cannot be read or is malformed
    """
    try:
        with open(nav_file, 'r') as f:
            content = f.read().split('\n\n')
        items = [item for item in content if '[Item =' in item]
        
        if not items:
            raise NavFileError(f"No valid items found in {nav_file}")
            
        log_print(f"Successfully parsed {len(items)} items from {nav_file}")
        return items
        
    except FileNotFoundError:
        raise NavFileError(f"Nav file not found: {nav_file}")
    except Exception as e:
        raise NavFileError(f"Error parsing {nav_file}: {str(e)}")

def select_top_items(
    items: List[str], 
    top_target_per_category: int, 
    existing_coords: List[Tuple[float, float]], 
    tolerance: float,
) -> List[str]:
    """
    Selects top N items ensuring minimum distance between coordinates.
    Logs detailed skip information to file while keeping console output clean.
    
    Args:
        items: List of item strings to filter
        top_target_per_category: Number of top items to select
        existing_coords: List of existing coordinate tuples
        tolerance: Minimum allowed distance between coordinates
        
    Returns:
        List[str]: Selected items meeting the criteria
    """
    selected_items = []
    skipped_count = 0
    
    for item in items:
        # Parse coordinates
        pts_x_match = re.search(r'PtsX = (-?[\d.]+)', item)
        pts_y_match = re.search(r'PtsY = (-?[\d.]+)', item)
        
        if not pts_x_match or not pts_y_match:
            log_print(f"Skipped item - Missing coordinates:\n{item.strip()}\n", logging.DEBUG)
            skipped_count += 1
            continue
        
        pts_x = float(pts_x_match.group(1))
        pts_y = float(pts_y_match.group(1))
        
        # Check distance from existing coordinates
        too_close = False
        for ex_index, ex in enumerate(existing_coords):
            distance = math.sqrt((pts_x - ex[0])**2 + (pts_y - ex[1])**2)
            if distance < tolerance:
                log_print(
                    f"Skipped item - Too close to existing target:\n"
                    f"Item details:\n{item.strip()}\n"
                    f"Distance: {distance:.2f} microns\n"
                    f"Existing target index: {ex_index + 1}\n"
                    f"Existing coordinates: ({ex[0]}, {ex[1]})\n",
                    logging.DEBUG
                )
                too_close = True
                skipped_count += 1
                break
        
        if not too_close:
            selected_items.append(item.strip())
            existing_coords.append((pts_x, pts_y))
            log_print(f"Selected item:\n{item.strip()}\n", logging.DEBUG)
            log_print(f"Selected item at coordinates ({pts_x:.2f}, {pts_y:.2f})")
        
        if len(selected_items) >= top_target_per_category:
            break
    
    # Log summary information
    log_print(f"Selected {len(selected_items)} items, skipped {skipped_count} items")
    log_print(f"\nDetailed selection summary:", logging.DEBUG)
    log_print(f"Total items processed: {len(selected_items) + skipped_count}", logging.DEBUG)
    log_print(f"Items selected: {len(selected_items)}", logging.DEBUG)
    log_print(f"Items skipped: {skipped_count}", logging.DEBUG)
    
    return selected_items

def block_based_target_selection(
    input_csv_path: str,
    output_file: str,
    top_n_per_block: int,
    tolerance: float,
    processor_instance,
    snapshot_output_dir: str = "targets/targets_per_block_diff_snapshots"
):
    import os
    import math
    from pathlib import Path
    from typing import List, Tuple

    def euclidean_distance(coord1, coord2):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    df = processor_instance.process_csv(input_csv_path)

    if "Block" not in df.columns:
        raise ValueError("Missing 'Block' column in input CSV")

    df = df[df["DifQuality"].isin(["Good diffraction", "Bad diffraction"])]
    blocks = sorted(df["Block"].unique())

    Path(snapshot_output_dir).mkdir(parents=True, exist_ok=True)

    all_entries = []
    existing_coords: List[Tuple[float, float]] = []
    total_selected = 0
    total_skipped = 0

    for block_index, block_id in enumerate(blocks):
        block_prefix = 4 + block_index
        block_df = df[df["Block"] == block_id].copy()
        sorted_block = block_df.sort_values(
            by=["QualityRank", "FTPeaks", "FilteredPeaks", "Sum"],
            ascending=[True, False, False, False]
        )
        used_indices = set()
        counters = {"quality": 0, "peaks": 0, "sum": 0}

        def select_best_unique(n, sort_by, category: str, quality_only=False):
            nonlocal total_selected, total_skipped
            selected = []
            skipped = 0

            candidate_rows = (
                sorted_block[sorted_block["DifQuality"] == "Good diffraction"]
                if quality_only else sorted_block
            )

            for idx, row in candidate_rows.sort_values(by=sort_by, ascending=False).iterrows():
                coord = eval(row["Coordinates"])
                coord_xy = (coord[0], coord[1])
                if idx in used_indices or any(euclidean_distance(coord_xy, ec) < tolerance for ec in existing_coords):
                    skipped += 1
                    continue

                category_digit = {"quality": 0, "peaks": 1, "sum": 2}[category]
                item_number = int(f"{block_prefix}{category_digit}{counters[category]}")
                counters[category] += 1

                row["CustomItemNumber"] = item_number
                selected.append(row)
                used_indices.add(idx)
                existing_coords.append(coord_xy)

                # Snapshot generation
                try:
                    mrc_file = row["Path"]
                    mrc_filename = os.path.splitext(os.path.basename(row["Path"]))[0]
                    snapshot_filename = f"{item_number}_{mrc_filename}_sum{int(row['Sum'])}.png"
                    snapshot_path = os.path.join(snapshot_output_dir, snapshot_filename)

                    processor_instance.plot_diffraction_snapshot(
                        mrc_file=mrc_file,
                        output_path=snapshot_path,
                        index=idx,
                        spots_value=row["FTPeaks"],
                        sum_value=row["Sum"]
                    )
                except Exception as e:
                    log_print(f"Snapshot failed for idx {idx}: {str(e)}", level=logging.WARNING)

                if len(selected) >= n:
                    break

            total_selected += len(selected)
            total_skipped += skipped
            log_print(f"Block {block_id}: Selected {len(selected)} items by {sort_by}, skipped {skipped}")
            return pd.DataFrame(selected)

        q_sel = select_best_unique(top_n_per_block, "QualityRank", "quality", quality_only=True)
        p_sel = select_best_unique(top_n_per_block, "FTPeaks", "peaks")
        s_sel = select_best_unique(top_n_per_block, "Sum", "sum")
        combined = pd.concat([q_sel, p_sel, s_sel]).drop_duplicates()

        entries = processor_instance.generate_nav_entries(combined, start_item_number=None)
        all_entries.extend(entries)
        
        # Add to master list for CSV export
        if 'all_selected_df' not in locals():
            all_selected_df = combined.copy()
        else:
            all_selected_df = pd.concat([all_selected_df, combined], ignore_index=True)

    log_print(f"\nFinal summary:")
    log_print(f"Total selected entries: {total_selected}")
    log_print(f"Total skipped entries due to duplication or proximity: {total_skipped}")

    nav_content = 'AdocVersion = 2.00\n' + '\n\n'.join(all_entries) + '\n'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(nav_content)

    log_print(f"Successfully created block-based .nav file: {output_file} with {len(all_entries)} entries")
    
    # Create corresponding CSV file
    csv_output_file = str(Path(output_file).parent / "targets.csv")
    if 'all_selected_df' in locals():
        # Ensure target_number column exists (use CustomItemNumber as target_number)
        if 'CustomItemNumber' in all_selected_df.columns:
            all_selected_df['target_number'] = all_selected_df['CustomItemNumber']
        all_selected_df.to_csv(csv_output_file, index=False)
        log_print(f"Successfully created corresponding CSV file: {csv_output_file} with {len(all_selected_df)} entries")


def create_targets_nav(
    output_file: str, 
    top_target_per_category: int, 
    tolerance: float, 
    input_files: Optional[List[str]] = None,
    processor_instance=None
) -> None:
    """
    Creates a new targets.nav file from multiple input files.
    
    Args:
        output_file: Path for the output file
        top_target_per_category: Number of top items to select from each file
        tolerance: Minimum distance between targets
        input_files: Optional list of input nav files
    """
    if input_files is None:
        input_files = [
            'targets/targets_nxds.nav',
            'targets/targets_quality.nav',
            'targets/targets_spots.nav',
            'targets/targets_sum.nav'
        ]
    
    all_selected_items = []
    existing_coords = []
    
    for nav_file in input_files:
        if not os.path.exists(nav_file):
            log_print(f"Input file not found, skipping: {nav_file}", logging.WARNING)
            continue
            
        try:
            items = parse_nav_file_correctly(nav_file)
            top_items = select_top_items(items, top_target_per_category, existing_coords, tolerance)
            all_selected_items.extend(top_items)
        except NavFileError as e:
            log_print(f"Error processing {nav_file}: {str(e)}", logging.ERROR)
            continue
    
    # Create output file
    try:
        with open(output_file, 'w') as f:
            f.write('AdocVersion = 2.00\n\n')
            f.write('\n\n'.join(all_selected_items) + '\n')
        log_print(f"Successfully created {output_file} with {len(all_selected_items)} items")
        
        # Create corresponding CSV file if we have selected items and processor
        if all_selected_items and processor_instance:
            csv_output_file = str(Path(output_file).parent / "targets.csv")
            create_targets_csv_from_nav_entries(all_selected_items, csv_output_file, processor_instance)
            
    except Exception as e:
        log_print(f"Error writing output file: {str(e)}", logging.ERROR)
        raise

def create_targets_csv_from_nav_entries(nav_entries: List[str], csv_output_file: str, processor_instance) -> None:
    """
    Create targets.csv from selected navigation entries by matching coordinates.
    
    Args:
        nav_entries: List of selected navigation entry strings
        csv_output_file: Path for the output CSV file
        processor_instance: DiffractionDataProcessor instance
    """
    try:
        # Load the original dif_map CSV data
        dif_map_csv = "dif_maps/dif_map_sums.csv"
        if not os.path.exists(dif_map_csv):
            log_print(f"Original CSV not found: {dif_map_csv}", logging.ERROR)
            return
            
        original_df = pd.read_csv(dif_map_csv)
        
        # Extract target information from nav entries
        selected_rows = []
        target_number = 1
        
        for nav_entry in nav_entries:
            # Extract coordinates from nav entry
            pts_x_match = re.search(r'PtsX = (-?[\d.]+)', nav_entry)
            pts_y_match = re.search(r'PtsY = (-?[\d.]+)', nav_entry)
            item_match = re.search(r'\[Item = (\d+)', nav_entry)
            
            if pts_x_match and pts_y_match:
                nav_x = float(pts_x_match.group(1))
                nav_y = float(pts_y_match.group(1))
                nav_item = int(item_match.group(1)) if item_match else target_number
                
                # Find matching row in original data by coordinates
                for idx, row in original_df.iterrows():
                    if pd.isna(row['Coordinates']) or row['Coordinates'] == '[None, None, None]':
                        continue
                        
                    try:
                        coords = eval(row['Coordinates'])
                        if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                            # Compare coordinates (allow small tolerance for floating point differences)
                            if abs(coords[0] - nav_x) < 0.1 and abs(coords[1] - nav_y) < 0.1:
                                # Found matching row
                                match_row = row.copy()
                                match_row['target_number'] = nav_item
                                selected_rows.append(match_row)
                                break
                    except:
                        continue
                        
            target_number += 1
        
        if selected_rows:
            targets_df = pd.DataFrame(selected_rows)
            targets_df.to_csv(csv_output_file, index=False)
            log_print(f"Successfully created corresponding CSV file: {csv_output_file} with {len(targets_df)} entries")
        else:
            log_print("No matching coordinates found between nav entries and original CSV data", logging.WARNING)
            
    except Exception as e:
        log_print(f"Error creating targets CSV: {str(e)}", logging.ERROR)

def create_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process NAV files and create combined targets file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Microscope configuration
    parser.add_argument(
        '--microscope',
        type=str,
        choices=list(MICROSCOPE_CONFIGS.keys()),
        default="Arctica-CETA",
        help='Microscope configuration to use (default: Arctica-CETA)'
    )
    parser.add_argument(
        '--output', 
        default='targets.nav',
        help='Output file path'
    )
    parser.add_argument(
        '--top-target-per-category', 
        type=int, 
        default=2,
        help='Number of top targets to select from each category (replaces old --top-n)'
    )
    parser.add_argument(
        '--top-target-per-block',
        type=int,
        default=None,
        help='If set, overrides category selection and applies top N filtering per block (alternative mode)'
    )

    parser.add_argument(
        '--tolerance', 
        type=float, 
        default=10.1,
        help='Minimum distance between targets (microns)'
    )
    parser.add_argument(
        '--input-files',
        nargs='+',
        help='List of input NAV files (optional)'
    )
    return parser

def main() -> Optional[int]:
    """
    Main function that orchestrates the NAV file processing.
    
    Returns:
        Optional[int]: Return code (0 for success, non-zero for error)
    """
    
    try:
        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Setup logging with specific directory
        setup_logging('processing.log', 'REyes_logs')
        print_banner()

        log_print(f"\nREyes FTP v{__version__} will generate final .nav file\n")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if args.top_target_per_block is not None:
            log_print(f"Running block-based target selection (top {args.top_target_per_block} per block)")
            processor = DiffractionDataProcessor(args.microscope) 
            block_based_target_selection(
                input_csv_path="dif_maps/dif_map_sums.csv",
                output_file=str(Path('targets') / args.output),
                top_n_per_block=args.top_target_per_block,
                tolerance=args.tolerance,
                processor_instance=processor
            )
        else:
            log_print(f"Running category-based target selection (top {args.top_target_per_category} per category)")
            processor = DiffractionDataProcessor(args.microscope) 
            create_targets_nav(
                output_file=str(Path('targets') / args.output),
                top_target_per_category=args.top_target_per_category,
                tolerance=args.tolerance,
                input_files=args.input_files,
                processor_instance=processor
            )

        
        log_print("\nProcessing completed successfully!\n")
        return 0
        
    except Exception as e:
        log_print(f"An error occurred: {str(e)}", logging.ERROR)
        return 1

if __name__ == "__main__":
    sys.exit(main())