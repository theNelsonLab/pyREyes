"""
REyes FMP (Final Map Plotter)
A tool to generate final diffraction maps over montage.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import hyperspy.api as hs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.image_processing.PlottingManager import PlottingManager
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

MICROSCOPE_CONFIGS = load_microscope_configs()

def find_mrc_and_mdoc_files() -> Tuple[Optional[str], Optional[str]]:
    """Search for .mrc and corresponding .mdoc files in current directory."""
    current_dir = os.getcwd()
    for file in os.listdir(current_dir):
        if file.endswith('.mrc'):
            mrc_path = os.path.join(current_dir, file)
            mdoc_path = f"{mrc_path}.mdoc"
            if os.path.exists(mdoc_path):
                return mrc_path, mdoc_path
    return None, None

class DiffractionQuality(Enum):
    """Enumeration for diffraction quality labels.
    
    Note: These quality classifications are based on DQI (Diffraction Quality Index),
    which is the ratio of LQP (Lattice Quality Peaks) to diffraction peaks.
    The DQI calculation and classification is performed in dif_map.py.
    """
    GRID = 0
    NO_DIFFRACTION = 1
    POOR_DIFFRACTION = 2
    BAD_DIFFRACTION = 3
    GOOD_DIFFRACTION = 4

    @classmethod
    def get_label_mapping(cls) -> Dict[str, int]:
        """Returns mapping of quality labels to integer values."""
        return {
            'Grid': cls.GRID.value,
            'No diffraction': cls.NO_DIFFRACTION.value,
            'Poor diffraction': cls.POOR_DIFFRACTION.value,
            'Bad diffraction': cls.BAD_DIFFRACTION.value,
            'Good diffraction': cls.GOOD_DIFFRACTION.value
        }

@dataclass
class PlotConfig:
    """Configuration settings for plot generation."""
    figure_size: Tuple[int, int] = (8, 8)
    padding_factor: float = 0.1
    dpix_size: int = 30
    target_marker_size: int = 15
    font_size: int = 8
    alpha: float = 0.5
    target_alpha: float = 0.8
    colorbar_pad: float = 0.07
    colorbar_fraction: float = 0.05
    
class Coordinates:
    """Stores and manages coordinate data."""
    def __init__(self, positions: np.ndarray, mapping: Dict[int, int], 
                 image_size: float, rotation_angle: float):
        self.positions = positions
        self.mapping = mapping
        self.image_size = image_size
        self.rotation_angle = rotation_angle

    @classmethod
    def from_mdoc(cls, mdoc_path: str, scaling_factor: float = 1.2) -> 'Coordinates':
        """Creates Coordinates instance from .mdoc file data."""
        positions = cls._parse_mdoc_coordinates(mdoc_path)
        mapping = cls._create_mapping(positions)
        image_size = cls._calculate_base_image_size(positions, scaling_factor)
        rotation_angle = cls._calculate_rotation_angle(positions)
        return cls(positions, mapping, image_size, rotation_angle)

    @staticmethod
    def _parse_mdoc_coordinates(mdoc_path: str) -> np.ndarray:
        """Parses coordinates from .mdoc file."""
        coordinates = []
        try:
            with open(mdoc_path, 'r') as file:
                for line in file:
                    if line.startswith('StagePosition ='):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y = float(parts[2]), float(parts[3])
                            if x != 0 or y != 0:  # Skip origin points
                                coordinates.append((x, y))
            
            if not coordinates:
                raise ValueError("No valid coordinates found in .mdoc file")
            
            return np.array(coordinates)
        except Exception as e:
            log_print(f"Error parsing .mdoc coordinates: {e}", logging.ERROR)
            raise

    @staticmethod
    def _create_mapping(coordinates: np.ndarray) -> Dict[int, int]:
        """Creates mapping between image indices and coordinate indices."""
        return {i + 1: i + 1 for i in range(len(coordinates))}

    @staticmethod
    def _calculate_base_image_size(coordinates: np.ndarray, scaling_factor: float = 1.2) -> float:
        """Calculates base image size from coordinate spacing."""
        if len(coordinates) < 2:
            raise ValueError("At least two coordinates required")
        distances = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
        return np.median(distances) * scaling_factor

    @staticmethod
    def _calculate_rotation_angle(coordinates: np.ndarray) -> float:
        """Calculates rotation angle from coordinate orientation."""
        if len(coordinates) < 2:
            raise ValueError("At least two coordinates required")
        delta_x = coordinates[1][0] - coordinates[0][0]
        delta_y = coordinates[1][1] - coordinates[0][1]
        return np.degrees(np.arctan2(delta_x, delta_y))

    def get_frame_position(self, coord_index: int) -> Tuple[float, float]:
        """Gets the position for a given coordinate index."""
        if coord_index < 1 or coord_index > len(self.positions):
            raise ValueError(f"Invalid coordinate index: {coord_index}")
        return tuple(self.positions[coord_index - 1])

def parse_nav_file(nav_path: str) -> Dict[int, Tuple[float, float]]:
    """Parse navigation file to extract target coordinates."""
    if not os.path.exists(nav_path):
        log_print(f"NAV file not found: {nav_path}", logging.WARNING)
        return {}

    coordinates = {}
    current_item = None
    
    try:
        with open(nav_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()  # Handle indented lines
                
                if stripped_line.startswith('[Item ='):
                    current_item = int(stripped_line.split('=')[1].strip().strip(']'))
                elif stripped_line.startswith('RawStageXY') and current_item is not None:
                    x, y = map(float, stripped_line.split('=')[1].strip().split())
                    coordinates[current_item] = (x, y)
                    current_item = None
        return coordinates
    except Exception as e:
        log_print(f"Error parsing NAV file: {e}", logging.ERROR)
        raise

class DiffractionMapProcessor:
    """Processes diffraction map data."""
    
    @staticmethod
    def process_map(
        input_csv: str,
        output_csv: str,
        heatmap_column: str,
        use_log_scale: bool = True,
        dif_quality: bool = False
    ) -> pd.DataFrame:
        """Processes diffraction map data from CSV."""
        try:
            log_print(f"Processing map from {input_csv}")
            df = pd.read_csv(input_csv)
            
            if 'Coordinates' not in df.columns:
                raise ValueError("Missing 'Coordinates' column in the CSV file")
            
            if df['Coordinates'].isna().all():
                raise ValueError("No coordinates found in diffraction maps")
            
            # Process coordinates
            coordinates_list = []
            for coord_str in df['Coordinates']:
                if pd.isna(coord_str):
                    coordinates_list.append([0, 0])
                else:
                    try:
                        coords = json.loads(coord_str)
                        coordinates_list.append(coords if isinstance(coords, list) else [0, 0])
                    except json.JSONDecodeError:
                        coordinates_list.append([0, 0])
            
            # Create coordinate columns safely
            coord_array = np.array(coordinates_list)
            df['x'] = coord_array[:, 0]
            df['y'] = coord_array[:, 1]

            # Process heatmap values
            if dif_quality:
                df[f'{heatmap_column}_original'] = df[heatmap_column]
                df['plot_value'] = df[heatmap_column].map(DiffractionQuality.get_label_mapping())
            else:
                df['heatmap_value'] = df[heatmap_column]
                df['plot_value'] = np.log10(df['heatmap_value'] + 1) if use_log_scale else df['heatmap_value']

            # Save processed data
            df[['x', 'y', 'plot_value', heatmap_column]].to_csv(output_csv, index=False)
            return df

        except Exception as e:
            log_print(f"Error processing diffraction map: {str(e)}", logging.ERROR)
            raise

class Plotter:
    """Handles plot generation and customization."""
    
    def __init__(self, config: PlotConfig):
        self.config = config
        self.manual_limits = None
        self.show_diff_map = True
        self.show_montage = True
        self.use_fixed_size = False
    
    def set_manual_limits(self, limits: Optional[Tuple[float, float, float, float]]) -> None:
        """Sets manual plot limits if provided."""
        self.manual_limits = limits
    
    def set_diff_map_visibility(self, show_diff_map: bool) -> None:
        """Sets whether diffraction map should be shown."""
        self.show_diff_map = show_diff_map

    def set_montage_visibility(self, show_montage: bool) -> None:
        """Sets whether montage should be shown."""
        self.show_montage = show_montage
    
    def set_fixed_size(self, use_fixed: bool) -> None:
        """Sets whether to use fixed point size or auto-calculate."""
        self.use_fixed_size = use_fixed

    def create_plot(
        self,
        df: pd.DataFrame,
        coordinates_nav: Dict[int, Tuple[float, float]],
        mnt: hs.signals.Signal2D,
        title: str,
        output_path: str,
        cbar_label: str,
        target_color: str,
        quality_map: bool = False,
        coordinates: Optional[Coordinates] = None
    ) -> None:
        """Creates and saves the plot with all overlays."""
        # Create figure first
        plt.figure(figsize=self.config.figure_size)
        
        # Set plot limits
        self._set_plot_limits(df)
        
        # Auto-size points unless fixed size is explicitly requested
        if not self.use_fixed_size and self.show_diff_map:
            optimal_size = self.calculate_optimal_point_size(df)
            log_print(f"Calculated optimal point size: {optimal_size:.2f}", logging.INFO)
            self.config.dpix_size = optimal_size
        else:
            log_print(f"Using fixed point size: {self.config.dpix_size}", logging.INFO)
        
        # Plot montage background (if enabled)
        if coordinates is not None and self.show_montage:
            self._plot_montage_frames(mnt, coordinates)
        
        # Plot diffraction map (if enabled)
        scatter = None
        if self.show_diff_map:
            scatter = self._plot_diffraction_map(df, quality_map)
            if scatter is not None:
                self._add_colorbar(scatter, cbar_label, quality_map)
        
        # Plot targets (always, if they exist)
        if coordinates_nav:
            self._plot_targets(coordinates_nav, target_color)
        
        # Finalize and save
        self._finalize_plot(title, output_path)
    
    def calculate_optimal_point_size(self, df: pd.DataFrame) -> float:
        """
        Calculates optimal scatter point size based on plot dimensions,
        with larger coordinate spaces getting smaller points.
        """
        if len(df) <= 1:
            return self.config.dpix_size
        
        # Calculate the bounds of the plot
        if self.manual_limits is not None:
            x_min, x_max, y_min, y_max = self.manual_limits
        else:
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
        
        # Calculate plot dimensions
        plot_width = max(x_max - x_min, 0.001)
        plot_height = max(y_max - y_min, 0.001)
        max_dimension = max(plot_width, plot_height)
        
        # Apply sizing algorithm
        if max_dimension <= 150:
            base_size = 1000.0
        elif max_dimension <= 300:
            base_size = 18000 / max_dimension
        elif max_dimension <= 1000:
            base_size = 5000 / max_dimension
        else:
            base_size = 1500 / max_dimension
        
        # Round and apply bounds
        base_size = round(base_size)
        return np.clip(base_size, 1.0, 1500.0)

    def _set_plot_limits(self, df: pd.DataFrame) -> None:
        if self.manual_limits is not None:
            x_min, x_max, y_min, y_max = self.manual_limits
            log_print(f"Using manual plot limits: x[{x_min}, {x_max}], y[{y_min}, {y_max}]")
        else:
            x_min, x_max = df['x'].min(), df['x'].max()
            y_min, y_max = df['y'].min(), df['y'].max()
            padding_x = self.config.padding_factor * (x_max - x_min)
            padding_y = self.config.padding_factor * (y_max - y_min)
            x_min -= padding_x
            x_max += padding_x
            y_min -= padding_y
            y_max += padding_y
            log_print(f"Using automatic plot limits: x[{x_min:.2f}, {x_max:.2f}], y[{y_min:.2f}, {y_max:.2f}]")
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', adjustable='box')

    def _plot_montage_frames(self, mnt: hs.signals.Signal2D, coordinates: Coordinates) -> None:
        for image_index, coord_index in coordinates.mapping.items():
            if image_index - 1 < len(mnt):
                frame = PlottingManager.clip_image(mnt.inav[image_index - 1].data)
                frame_rotated = PlottingManager.rotate_with_nan(frame, coordinates.rotation_angle)
                
                x, y = coordinates.get_frame_position(coord_index)
                half_size = coordinates.image_size / 2
                extent = [x - half_size, x + half_size, 
                         y - half_size, y + half_size]
                
                plt.imshow(frame_rotated, extent=extent, origin='lower',
                          cmap='gray', alpha=1.0)

    def _plot_diffraction_map(self, df: pd.DataFrame, quality_map: bool):
        if quality_map:
            cmap = mcolors.ListedColormap(['#4d4d4d', '#8da0cb', '#ffd92f',
                                         '#fc8d62', '#66c2a5'])
            boundaries = [0, 1, 2, 3, 4, 5]
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        else:
            cmap = plt.get_cmap('gist_heat')
            norm = None

        return plt.scatter(
            df['x'], df['y'],
            c=df['plot_value'],
            cmap=cmap,
            norm=norm,
            s=self.config.dpix_size,
            marker='s',
            edgecolors='none',
            alpha=self.config.alpha
        )

    def _plot_targets(self, coordinates_nav: Dict[int, Tuple[float, float]], target_color: str) -> None:
        if not coordinates_nav:
            return
        
        # Get current plot limits
        xlim = plt.xlim()
        ylim = plt.ylim()
        
        # Filter targets to only those within plot bounds
        filtered_targets = {}
        for item, (x, y) in coordinates_nav.items():
            if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
                filtered_targets[item] = (x, y)
        
        if not filtered_targets:
            return
        
        # Plot only the filtered targets
        x_coords, y_coords = zip(*filtered_targets.values())
        
        plt.scatter(
            x_coords, y_coords,
            color=target_color,
            s=self.config.target_marker_size,
            marker='x',
            label='targets.nav',
            alpha=self.config.target_alpha
        )

        # Add text labels only for filtered targets
        for item, (x, y) in filtered_targets.items():
            plt.text(
                x, y,
                str(item),
                fontsize=self.config.font_size,
                ha='right',
                color=target_color,
                fontweight='bold',
                alpha=self.config.target_alpha
            )

    def _add_colorbar(self, scatter, cbar_label: str, quality_map: bool) -> None:
        cbar = plt.colorbar(
            scatter,
            orientation='horizontal',
            pad=self.config.colorbar_pad,
            fraction=self.config.colorbar_fraction
        )

        if quality_map:
            cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
            cbar.set_ticklabels([q.name.replace('_', ' ').title()
                               for q in DiffractionQuality])
        cbar.set_label(cbar_label)

    def _finalize_plot(self, title: str, output_path: str) -> None:
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(output_path, transparent=True, dpi=300, bbox_inches='tight')
        plt.close()

def create_parser() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='REyes FMP - Final Map Plotter')
    parser.add_argument(
        '--microscope',
        type=str,
        choices=list(MICROSCOPE_CONFIGS.keys()),
        default="Arctica-CETA",
        help='Microscope configuration to use'
    )
    parser.add_argument(
        '--dpix-size',
        type=int,
        help='Override automatic sizing with fixed size for scatter plot diffraction pixel'
    )
    parser.add_argument(
        '--no-targets',
        action='store_true',
        help='Disable plotting of targets from targets.nav file'
    )
    parser.add_argument(
        '--no-diff-map',
        action='store_true',
        help='Disable diffraction map overlay (shows only montage frames)'
    )
    parser.add_argument(
        '--no-mnt',
        action='store_true',
        help='Disable montage background (shows only diffraction map and targets)'
    )
    parser.add_argument(
        '--plot-limits',
        type=float,
        nargs=4,
        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help='Override plot limits: xmin xmax ymin ymax'
    )
    return parser

def main() -> int:
    """Main execution function."""
    try:
        parser = create_parser()
        args = parser.parse_args()
                
        # Setup
        work_dir = os.getcwd()
        output_dir = os.path.join(work_dir, 'dif_maps')
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        setup_logging('diffraction_maps.log', 'REyes_logs')
        print_banner()

        # Validate argument combinations
        if args.no_diff_map and args.no_mnt:
            log_print("Error: Cannot use --no-diff-map and --no-mnt together as nothing would be plotted", logging.ERROR)
            return 1
        
        # Get scaling factor from microscope config
        microscope_config = MICROSCOPE_CONFIGS[args.microscope]
        scaling_factor = microscope_config.scaling
        log_print(f"Using {args.microscope} configuration with scaling factor {scaling_factor}")

        log_print(f"\nREyes FMP v{__version__} will plot:")
        if not args.no_mnt:
            log_print("- Montage background")
        if not args.no_diff_map:
            log_print("- Diffraction map overlay")
        if not args.no_targets:
            log_print("- Selected targets")
            
        # Point size information
        if args.dpix_size is not None:
            log_print(f"- Using fixed point size: {args.dpix_size}")
        else:
            log_print("- Using automatic point sizing optimization")
        
        log_print("")
        
        # Find required files
        mrc_path, mdoc_path = find_mrc_and_mdoc_files()
        if not mrc_path or not mdoc_path:
            log_print("Required .mrc and .mdoc files not found", logging.ERROR)
            return 1

        # Load files
        mnt = hs.load(mrc_path)
        coordinates = Coordinates.from_mdoc(mdoc_path, scaling_factor)
        
        # Load nav coordinates if targets are to be plotted
        coordinates_nav = {}
        if not args.no_targets:
            nav_path = os.path.join(work_dir, 'targets', 'targets.nav')
            coordinates_nav = parse_nav_file(nav_path)
        
        # Initialize plot configuration
        plot_config = PlotConfig()
        if args.dpix_size is not None:
            plot_config.dpix_size = args.dpix_size
            
        plotter = Plotter(plot_config)
        
        # Set configuration flags
        plotter.set_fixed_size(args.dpix_size is not None)
        
        if args.plot_limits:
            plotter.set_manual_limits(args.plot_limits)
        
        plotter.set_diff_map_visibility(not args.no_diff_map)
        plotter.set_montage_visibility(not args.no_mnt)
        
        # Build title components
        components = []
        if not args.no_mnt:
            components.append("Montage")
        if not args.no_diff_map:
            components.append("Diffraction Map")
        if not args.no_targets:
            components.append("Targets")
        
        map_configs = [
            {
                'input_csv': 'dif_map_sums.csv',
                'output_csv': 'diff_map_plot.csv',
                'column': 'Sum',
                'title': ' w/ '.join(components + (['Sum Map'] if not args.no_diff_map else [])),
                'output_file': 'mnt_sum_dif_map_w_targets.png',
                'cbar_label': 'Log(Sum)',
                'use_log': True,
                'quality_map': False
            },
            {
                'input_csv': 'dif_map_sums.csv',
                'output_csv': 'diff_map_peaks_plot.csv',
                'column': 'FilteredPeaks',
                'title': ' w/ '.join(components + (['Diffraction Peaks Map'] if not args.no_diff_map else [])),
                'output_file': 'mnt_dif_peaks_map_w_targets.png',
                'cbar_label': 'Diffraction Peaks',
                'use_log': False,
                'quality_map': False
            },
            {
                'input_csv': 'dif_map_sums.csv',
                'output_csv': 'diff_map_lqp_plot.csv',
                'column': 'FTPeaks',
                'title': ' w/ '.join(components + (['LQP Map'] if not args.no_diff_map else [])),
                'output_file': 'mnt_lqp_map_w_targets.png',
                'cbar_label': 'LQP',
                'use_log': False,
                'quality_map': False
            },
            {
                'input_csv': 'dif_map_sums.csv',
                'output_csv': 'diff_qual_map_plot.csv',
                'column': 'DifQuality',
                'title': ' w/ '.join(components + (['Quality Map'] if not args.no_diff_map else [])),
                'output_file': 'mnt_quality_map_w_targets.png',
                'cbar_label': 'Diffraction Quality',
                'use_log': False,
                'quality_map': True
            }
        ]
        
        # Process maps
        for config in map_configs:
            try:
                input_path = os.path.join(work_dir, 'dif_maps', config['input_csv'])
                output_path = os.path.join(output_dir, config['output_csv'])
                
                df = DiffractionMapProcessor.process_map(
                    input_path,
                    output_path,
                    config['column'],
                    config['use_log'],
                    config['quality_map']
                )
                
                plotter.create_plot(
                    df,
                    coordinates_nav,
                    mnt,
                    config['title'],
                    os.path.join(output_dir, config['output_file']),
                    config['cbar_label'],
                    'white',
                    config['quality_map'],
                    coordinates
                )
            except Exception as e:
                log_print(f"Error processing {config['column']}: {str(e)}", logging.ERROR)
                continue
        
        log_print("\nProcessing completed successfully!\n")
        return 0
        
    except Exception as e:
        log_print(f"Error in main execution: {str(e)}", logging.ERROR)
        return 1

if __name__ == "__main__":
    sys.exit(main())