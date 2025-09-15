"""
REyes GSS (Grid Squares Searcher)
A tool to process a montage of images and detect grid squares' centers.
"""

import sys
import logging
import argparse
import dataclasses
import numpy as np
import pickle
from pathlib import Path
import hyperspy.api as hs
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_microscope_configurations import MicroscopeConfig, load_microscope_configs
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_errors import GridSquareError, CentroidDetectionError, FileNotFoundError
from pyREyes.lib.image_processing.CartImage import CartImage
from pyREyes.lib.image_processing.CartImage_utils import solve_scaling_magnitude
from pyREyes.lib.image_processing.image_processor import ImageProcessor
from pyREyes.lib.image_processing.PlottingManager import PlottingManager
from pyREyes.lib.grid_processing.GridProcessor import GridProcessor
from pyREyes.lib.grid_processing.CentroidProcessor import CentroidProcessor
from pyREyes.lib.REyes_utils import find_mrc_and_mdoc_files

MICROSCOPE_CONFIGS = load_microscope_configs()


def get_mrc_resolution(mrc_file):
    """Load MRC file and return resolution information."""
    s = hs.load(mrc_file)
    resolution = s.data.shape[1:3]  # Assuming (n_frames, height, width)
    logging.info(f"Frame resolution: {resolution[0]} x {resolution[1]} pixels")
    return s

class GridSquareProcessor:
    """Main class for processing grid square images and detecting centroids."""
    
    def __init__(self, microscope_type: str, filtering_type: str = "default", debug: bool = False):
        """
        Initialize GridSquareProcessor with microscope and filtering configuration.
        
        Args:
            microscope_type: Type of microscope ("Arctica-CETA" or "F200-Apollo" or "F30-TVIPS", or "Arctica-Apollo")
            filtering_type: Type of filtering to apply ("default", "narrow", or "well-plate")
            
        Raises:
            ValueError: If microscope_type or filtering_type is invalid
        """
        if microscope_type not in MICROSCOPE_CONFIGS:
            raise ValueError(f"Unknown microscope type: {microscope_type}")
        
        self.microscope_type = microscope_type
        self.config = MICROSCOPE_CONFIGS[microscope_type]
        self.filtering_type = filtering_type
        
        # Create output directory
        self.output_dir = Path('grid_squares')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize service classes with new output directory
        self.centroid_processor = CentroidProcessor(self.config)
        self.plotting_manager = PlottingManager(self.output_dir, self.config)
        self.nav_generator = NavFileGenerator(self.output_dir)

        # Set debug mode
        self.debug = debug
    
    def _generate_nav_file(self,
                          centroids: List[Tuple[float, float]],
                          stage_z: float,
                          filename: str) -> None:
        """Generate navigation file using the nav generator."""
        self.nav_generator.generate_nav_file(centroids, stage_z, filename)
    
    def _generate_eucentricity_nav(self,
                                  centroids: List[Tuple[float, float]],
                                  stage_z: float) -> None:
        """Generate eucentricity nav file using the nav generator."""
        self.nav_generator.generate_eucentricity_nav(centroids, stage_z)
        
    def process_montage(self, mrc_path: Path, mdoc_path: Path) -> None:
        """Main processing pipeline for grid square detection."""
        try:
            logging.info(f"Processing montage with {self.microscope_type} configuration...")  # Use microscope_type instead of config class
            
            rotation_angle, coordinates, stage_z = self._calculate_rotation_angle(mdoc_path)
            
            # Load MRC file once and reuse it
            mrc_data = self._load_mrc_file(mrc_path)
            centroids, extent, montage_image = self._detect_centroids(mrc_data, rotation_angle, coordinates)
            if not centroids:
                raise CentroidDetectionError("No centroids found after processing")

            # Generate outputs using the already loaded MRC data
            self._generate_outputs(mrc_path, montage_image, rotation_angle, coordinates, extent,
                                 centroids, stage_z, mrc_data)
                                 
        except Exception as e:
            logging.error(f"Error processing montage: {str(e)}")
            raise
    
    def _calculate_rotation_angle(self, mdoc_path: Path) -> Tuple[float, np.ndarray, float]:
        """
        Calculate rotation angle and extract stage coordinates from mdoc file.
        
        Args:Can
            mdoc_path: Path to the .mdoc file
            
        Returns:
            Tuple containing:
            - rotation_angle (float): Calculated rotation angle in degrees
            - coordinates (np.ndarray): Array of stage coordinates
            - stage_z (float): Z-stage position
            
        Raises:
            GridSquareError: If mdoc file cannot be read or processed
        """
        try:
            coordinates = []
            stage_z = None
            
            with mdoc_path.open('r') as file:
                for line in file:
                    if line.startswith('StagePosition ='):
                        try:
                            parts = line.strip().split()
                            x, y = float(parts[2]), float(parts[3])
                            coordinates.append((x, y))
                        except (IndexError, ValueError) as e:
                            logging.warning(f"Skipping malformed stage position line: {line.strip()}")
                            continue
                            
                    elif line.startswith('StageZ =') and stage_z is None:
                        try:
                            stage_z = float(line.strip().split()[-1])
                        except (IndexError, ValueError) as e:
                            logging.warning(f"Could not parse StageZ value: {line.strip()}")
                            continue

            if not coordinates:
                raise GridSquareError("No valid stage coordinates found in mdoc file")
                
            if stage_z is None:
                logging.warning("No StageZ value found, using default of 0.0")
                stage_z = 0.0

            coordinates = np.array(coordinates)
            
            # Calculate rotation angle using first two coordinates
            if len(coordinates) > 1:
                p1, p2 = coordinates[0], coordinates[1]
                delta_x, delta_y = p2[0] - p1[0], p2[1] - p1[1]
                angle = np.degrees(np.arctan2(delta_x, delta_y))
            else:
                logging.warning("Not enough coordinates for angle calculation, using 0.0")
                angle = 0.0

            # Apply user-defined rotation offset
            user_offset = 0.0  # Could be made configurable
            rotation_angle = angle + user_offset
            
            logging.info(f"Calculated rotation angle: {rotation_angle:.2f}°")
            logging.info(f"Eucentric height: {stage_z:.2f} µm")
            logging.debug(f"Found {len(coordinates)} stage positions")

            return rotation_angle, coordinates, stage_z
            
        except Exception as e:
            raise GridSquareError(f"Error processing mdoc file: {str(e)}") from e

    def _load_mrc_file(self, mrc_path: Path) -> hs.signals.Signal2D:
        """Load and validate MRC file data."""
        try:
            # Load the MRC file using hyperspy
            signal = hs.load(str(mrc_path))
            
            # Validate the loaded data
            if signal is None:
                raise GridSquareError("Failed to load MRC file")
                
            if len(signal.data.shape) != 3:
                raise GridSquareError(
                    f"Invalid MRC data shape: {signal.data.shape}. Expected 3 dimensions."
                )
                
            n_frames, height, width = signal.data.shape
            if n_frames < 1:
                raise GridSquareError("MRC file contains no frames")
                
            # Log successful load
            logging.info(f"Loaded MRC file with {n_frames} frames")
            
            # Also log frame resolution if you want to keep that information
            logging.debug(f"Frame resolution: {height} x {width} pixels")
            
            return signal
            
        except Exception as e:
            raise GridSquareError(f"Error loading MRC file: {str(e)}") from e
    
    def _detect_centroids(
        self,
        mrc_data: hs.signals.Signal2D,
        rotation_angle: float,
        coordinates: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Process frames to detect and locate centroids.
        
        Args:
            mrc_data: Hyperspy signal containing frame data
            rotation_angle: Angle to rotate frames
            coordinates: Array of stage coordinates
            
        Returns:
            List of detected centroids in Cartesian coordinates
        """
        try:
            nearest_distances = [
                np.linalg.norm(coordinates[i] - coordinates[i-1])
                for i in range(1, len(coordinates))
            ]
            base_image_size = min(nearest_distances)
            scaling_coefficient = self.config.scaling

            # ----------------- Creating polygons for the images -----------------
            polygons = []
            for i in range(len(mrc_data)):
                frame = np.zeros((1, 1))                
                scaling_para = solve_scaling_magnitude(scaling_coefficient, rotation_angle)
                cartimage = CartImage(frame, coordinates[i], base_image_size, base_image_size, -rotation_angle, scaling_para)
                polygon = cartimage.get_shapely_polygon()
                polygons.append(polygon)

            merged = unary_union(polygons)
            min_x, min_y, max_x, max_y = merged.bounds
            extent = (min_x, max_x, min_y, max_y)

            # ----------------- Using Matplot lib to interpolate image and extract the data of the image then lable them -----------------
            image_from_matplotlib = self.plotting_manager.plot_return_image(
                mrc_data, rotation_angle, extent, coordinates
            )
            masked_image = ImageProcessor.mask_image_outside_polygon(image_from_matplotlib, extent, merged) # Get the image in polygon
            dark_area, bright_area, binary_image = ImageProcessor.calculate_normalized_areas(masked_image, dark_threshold=self.config.dark_threshold)  # Get the binary image 
            labeled_image, num_features, region_sizes = ImageProcessor.find_connected_regions(binary_image) # Get the number of connected regions in the entire image
            
            if self.debug:
                PlottingManager.plot_image(binary_image, "Binary Image") # Plot the binary image
                ImageProcessor.plot_labeled_regions_random_colors(labeled_image, num_features, save_path = "colored regions") # Plot the labeled regions
                
            # ----------------- Calculating the edge strip -----------------
            space_width, bar_width, strip = ImageProcessor.compute_strip_and_spacing(
                merged_polygon=merged,
                num_regions=num_features,
                bright_area=bright_area,
                dark_area=dark_area
            )

            clearly_labeled_image, new_numer_features = ImageProcessor.remove_edge_touching_regions(labeled_image, extent, region_sizes, strip)
            centroids = ImageProcessor.merge_small_regions_centroids(clearly_labeled_image, extent, std_merge_threshold = 1.5, space_width = space_width, binary_image = binary_image, debug = self.debug, plotting_manager=self.plotting_manager)
            centroid_list = [(round(x, 3), round(y, 3)) for x, y in centroids.values()]

            if not centroid_list:
                raise CentroidDetectionError("No centroids detected in any frame")
                
            logging.info(f"Detected {len(centroid_list)} centroids across {len(mrc_data)} frames")
            return centroid_list, extent, image_from_matplotlib
            
        except Exception as e:
            raise CentroidDetectionError(f"Error detecting centroids: {str(e)}") from e
    

    def _validate_coordinates(self, coordinates: np.ndarray) -> bool:
        """
        Validate stage coordinates array.
        
        Args:
            coordinates: Array of stage positions
            
        Returns:
            bool: True if coordinates are valid
            
        Raises:
            GridSquareError: If coordinates are invalid
        """
        if coordinates.size == 0:
            raise GridSquareError("Empty coordinates array")
            
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise GridSquareError(
                f"Invalid coordinates shape: {coordinates.shape}. Expected (n, 2)."
            )
            
        if np.any(~np.isfinite(coordinates)):
            raise GridSquareError("Coordinates contain non-finite values")
            
        return True

    def _preprocess_frame(self, 
                         frame: np.ndarray, 
                         rotation_angle: float,
                         border_margin: int = 20) -> np.ndarray:
        """
        Preprocess a single frame before centroid detection.
        
        Args:
            frame: Input frame data
            rotation_angle: Angle to rotate frame
            border_margin: Margin size for border mask
            
        Returns:
            Preprocessed frame data
        """
        # Clip intensity values using ImageProcessor
        frame_clipped = ImageProcessor.clip_image(frame)
        
        # Apply threshold mask
        masked_frame = np.where(frame >= self.config.threshold, frame, 0)
        
        # Create and apply border mask
        border_mask = np.zeros_like(masked_frame)
        border_mask[border_margin:-border_margin, border_margin:-border_margin] = 1
        masked_frame = masked_frame * border_mask
        
        # Rotate frame using ImageProcessor
        rotated_frame = ImageProcessor.rotate_with_nan(masked_frame, rotation_angle)
        
        return rotated_frame
    
    def _generate_outputs(self, 
                         mrc_path: Path, 
                         montage_image: np.ndarray,
                         rotation_angle: float,
                         coordinates: np.ndarray, 
                         extent,
                         centroids: List[Tuple[float, float]], 
                         stage_z: float,
                         mrc_data: Optional[hs.signals.Signal2D] = None) -> None:
        """Generate all output files (plot, nav files)."""
        try:
            if not isinstance(stage_z, (int, float)) or not np.isfinite(stage_z):
                raise GridSquareError(f"Invalid stage_z value: {stage_z}")
                
            filtered_centroids, eucentricity_centroids = self.centroid_processor.filter_centroids(centroids, self.filtering_type)
            sorted_centroids = self.centroid_processor.sort_centroids(filtered_centroids)
            
            # Generate nav files first (less memory intensive)
            self.nav_generator.generate_nav_file(sorted_centroids, stage_z, "grid_squares.nav")
            self.nav_generator.generate_eucentricity_nav(eucentricity_centroids, stage_z)                
            self.plotting_manager.plot_montage_from_image(montage_image, extent, sorted_centroids, save_path = "combined_grid_montage")
            self.plotting_manager.plot_montage_from_image(montage_image, extent, eucentricity_centroids, save_path = "eucentricity_points")

            # Save extent and montage image for manual squares later
            temp_dir = Path.cwd() / "reyes_temp"
            temp_dir.mkdir(exist_ok=True)

            # Save extent
            with open(temp_dir / "extent.pkl", "wb") as f:
                pickle.dump(extent, f)

            # Save montage image
            np.save(temp_dir / "montage_image.npy", montage_image)

        except Exception as e:
            raise GridSquareError(f"Error generating outputs: {str(e)}") from e



class NavFileGenerator:
    """Handles generation of navigation files."""
    
    NAV_TEMPLATE = """
        [Item = {item_number}]
        Color = 0
        StageXYZ = {stage_x} {stage_y} {stage_z}
        NumPts = 1
        Regis = 1
        Type = 0
        RawStageXY = {stage_x} {stage_y}
        MapID = {map_id}
        PtsX = {stage_x}
        PtsY = {stage_y}
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def generate_nav_file(self,
                         centroids: List[Tuple[float, float]],
                         stage_z: float,
                         filename: str = 'grid_squares.nav',
                         log_output: bool = True) -> None:
        """
        Generate navigation file from centroids.
        
        Args:
            centroids: List of centroid coordinates
            stage_z: Z-stage position
            filename: Output filename
            log_output: Whether to log the generation message
        """
        try:
            nav_entries = []
            
            for index, (stage_x, stage_y) in enumerate(centroids):
                item_number = index + 2  # Start from 2
                nav_entry = self.NAV_TEMPLATE.format(
                    item_number=item_number,
                    stage_x=stage_x,
                    stage_y=stage_y,
                    stage_z=stage_z,
                    map_id=item_number
                )
                nav_entries.append(nav_entry)
            
            nav_content = 'AdocVersion = 2.00\n' + '\n'.join(nav_entries)
            
            nav_path = self.output_dir / filename
            nav_path.write_text(nav_content)
            
            if log_output:
                logging.info(f"Generated {filename} with {len(centroids)} entries")
            
        except Exception as e:
            raise GridSquareError(f"Error generating nav file: {str(e)}") from e
    
    def generate_eucentricity_nav(self,
                                 eucentricity_centroids: List[Tuple[float, float]],
                                 stage_z: float,
                                 filename: str = 'eucentricity.nav') -> None:
        """Generate eucentricity navigation file."""
        try:
            # Pass log_output=False to avoid duplicate logging
            self.generate_nav_file(eucentricity_centroids, stage_z, filename, log_output=False)
            logging.info(f"Generated {filename} with {len(eucentricity_centroids)} entries")
            
        except Exception as e:
            raise GridSquareError(
                f"Error generating eucentricity nav file: {str(e)}"
            ) from e


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Process grid squares with custom configuration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--microscope',
        type=str,
        choices=list(MICROSCOPE_CONFIGS.keys()),
        default="Arctica-CETA",
        help='Microscope configuration to use'
    )
    
    parser.add_argument(
        '--filtering',
        choices=['default', '1', '4', '9', '16', '25', '36', '49', '64', '81', '100', '121', '144', '169', '196', '96', "None"],
        default='default',
        help='Type of filtering to apply'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser

def main() -> int:
    """Main execution function with command line argument support.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command line arguments
        parser = create_parser()
        args = parser.parse_args()
        
        # Initialize logging and display banner
        setup_logging('grid_squares.log', 'REyes_logs')
        print_banner()
        
        log_print(f"\nREyes GSS v{__version__} will find grid squares "
                 f"and prepare navigator files\n")
        
        # Find input files
        mrc_path, mdoc_path = find_mrc_and_mdoc_files()
        if not (mrc_path and mdoc_path):
            log_print("No matching .mrc and .mdoc files found", logging.ERROR)
            return 1
        
        # Initialize and run processor
        processor = GridSquareProcessor(
            microscope_type=args.microscope,
            filtering_type=args.filtering,
            debug = args.debug
        )
        processor.process_montage(mrc_path, mdoc_path)
        
        log_print("\nProcessing completed successfully!\n")
        return 0
        
    except FileNotFoundError as e:
        log_print(f"File error: {str(e)}", logging.ERROR)
        return 1
    except Exception as e:
        log_print(f"Processing failed: {str(e)}", logging.ERROR)
        logging.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())