"""
REyes NEC (Navigator Eucentricity Corrector)
A tool for processing electron microscope stage positions and correcting eucentricity tilt.
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import LinearRegression

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_errors import ProcessingError

__version__ = '3.4.0'
__min_required_version__ = '3.4.0'

@dataclass
class RegressionResult:
    """Store regression calculation results."""
    a: float
    b: float
    c: float
    df: pd.DataFrame

def locate_files() -> Tuple[str, str]:
    """Locate required log and nav files."""
    try:
        current_dir = os.getcwd()
        grid_squares_dir = os.path.join(current_dir, 'grid_squares')
        
        # Look for log file in current directory (original location)
        log_file = next((file for file in os.listdir(current_dir) if file.endswith('_grid_squares.log')), None)
        if not log_file:
            log_print("No SerialEM log file found in the current directory.", logging.ERROR)  # Changed to ERROR level
            exit(1)
        log_file = os.path.join(current_dir, log_file)
            
        # Look for nav file in grid_squares directory
        nav_file = os.path.join(grid_squares_dir, 'grid_squares.nav')
        if not os.path.exists(nav_file):
            log_print(f"Nav file not found: {nav_file}", logging.ERROR)  # This was already at ERROR level
            exit(1)
            
        return log_file, nav_file
    except Exception as e:
        log_print(f"Error locating files: {e}", logging.ERROR)
        exit(1)

def parse_log_file(log_file: str) -> Tuple[List[List[float]], Optional[int]]:
    """Parse coordinates and expected items count from log file."""
    try:
        coordinates: List[List[float]] = []
        expected_num_items: Optional[int] = None
        
        with open(log_file, 'r') as file:
            for line in file:
                if line.startswith('Stage '):
                    match = re.findall(r'-?\d+\.\d+', line)
                    if match and len(match) == 3:
                        coordinates.append([float(x) for x in match])
                elif 'Current number of items is' in line:
                    expected_num_items = int(re.search(r'\d+', line).group())
                    
        if not coordinates:
            log_print("No coordinates found in log file.")
            exit(1)
            
        return coordinates, expected_num_items
    except Exception as e:
        log_print(f"Error parsing log file: {e}", logging.ERROR)
        exit(1)

def perform_regression(coordinates: List[List[float]]) -> RegressionResult:
    """Perform linear regression on coordinates.
    
    Args:
        coordinates: List of coordinate lists [x, y, z]
        
    Returns:
        RegressionResult object containing coefficients and DataFrame
        
    Raises:
        ProcessingError: If regression fails
    """
    try:
        df = pd.DataFrame(coordinates, columns=['x', 'y', 'z'])
        X, y = df[['x', 'y']], df['z']
        reg = LinearRegression().fit(X, y)
        a, b, c = reg.coef_[0], reg.coef_[1], reg.intercept_

        # Calculate standard errors
        n = len(df)
        mse = np.sum((y - reg.predict(X)) ** 2) / (n - 3)  # 3 parameters: a, b, c
        X_with_intercept = np.column_stack([X, np.ones(n)])
        covariance_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        std_errors = np.sqrt(np.diag(covariance_matrix))

        # Format coefficient strings with their standard errors
        # First term needs special handling since it starts the equation
        a_term = f"{'-' if a < 0 else ''}{abs(a):.3f}"
        # Other terms include the operator with proper spacing
        b_term = f"{' + ' if b >= 0 else ' - '}{abs(b):.3f}"
        c_term = f"{' + ' if c >= 0 else ' - '}{abs(c):.2f}"
        
        # Format standard errors with specific decimal places
        # For c_delta, pad based on number of digits before decimal
        c_digits_before_decimal = len(str(int(abs(std_errors[2]))))
        c_padding = ' ' * (1 if c_digits_before_decimal == 1 else 0)
        
        delta_vals = [
            f"±{std_errors[0]:.3f}",
            f"±{std_errors[1]:.3f}",
            f"±{c_padding}{std_errors[2]:.2f}"
        ]
        
        # Create output template with exact spacing
        output_template = (
            "Equation of the plane: {equation}\n"
            "Delta:                {deltas}"
        )
        
        # Build equation and deltas with proper alignment
        equation_str = f"z = {a_term}x{b_term}y{c_term}"
        deltas_str = "     " + "   ".join(delta_vals)
        
        # Format complete output
        output = output_template.format(
            equation=equation_str,
            deltas=deltas_str
        )
        
        # Print formatted output
        log_print(output)
        log_print(f"\nR^2: {reg.score(X, y):.4f}")
        
        return RegressionResult(a=a, b=b, c=c, df=df)
        
    except Exception as e:
        raise ProcessingError(f"Error performing regression: {str(e)}") from e

def extract_nav_coordinates(nav_file: str) -> List[List[float]]:
    """Extract coordinates from NAV file."""
    try:
        original_coordinates: List[List[float]] = []
        with open(nav_file, 'r') as file:
            for line in file:
                if line.strip().startswith('StageXYZ ='):
                    # Changed from r'-?\d+\.\d+' to r'-?\d+\.?\d*' to match update_nav_file
                    xyz = list(map(float, re.findall(r'-?\d+\.?\d*', line)))
                    if len(xyz) == 3:
                        original_coordinates.append(xyz)
                        
        if not original_coordinates:
            log_print("No coordinates found in NAV file.")
            exit(1)
            
        return original_coordinates
    except Exception as e:
        log_print(f"Error extracting NAV coordinates: {e}", logging.ERROR)
        exit(1)

def update_nav_file(nav_file: str, a: float, b: float, c: float) -> List[List[float]]:
    """Update NAV file with corrected coordinates."""
    try:
        updated_coordinates: List[List[float]] = []
        updated_lines: List[str] = []
        
        with open(nav_file, 'r') as file:
            for line in file:
                if line.strip().startswith('StageXYZ ='):
                    x, y, _ = map(float, re.findall(r'-?\d+\.?\d*', line)[:3])
                    new_z = a * x + b * y + c
                    updated_coordinates.append([x, y, new_z])
                    updated_line = re.sub(r'(-?\d+\.?\d*)$', f"{new_z:.3f}", line)
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
                    
        with open(nav_file, 'w') as out_file:
            out_file.writelines(updated_lines)
            
        return updated_coordinates
    except Exception as e:
        log_print(f"Error updating NAV file: {e}", logging.ERROR)
        exit(1)

def plot_results(orig_df: pd.DataFrame, updated_df: pd.DataFrame, df: pd.DataFrame, 
                a: float, b: float, c: float) -> None:
    """Create visualization plots."""
    try:
        # Calculate axis limits for consistency across plots
        x_min = min(orig_df['x'].min(), updated_df['x'].min(), df['x'].min())
        x_max = max(orig_df['x'].max(), updated_df['x'].max(), df['x'].max())
        y_min = min(orig_df['y'].min(), updated_df['y'].min(), df['y'].min())
        y_max = max(orig_df['y'].max(), updated_df['y'].max(), df['y'].max())
        z_min = min(orig_df['z'].min(), updated_df['z'].min(), df['z'].min())
        z_max = max(orig_df['z'].max(), updated_df['z'].max(), df['z'].max())

        fig = plt.figure(figsize=(15, 10))

        # Subplot 1: Original 3D grid squares
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(orig_df['x'], orig_df['y'], orig_df['z'], color='blue', marker='o', label='Grid squares')
        ax1.set_title('Original Grid Squares')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.legend()

        # Subplot 3: Empty
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.axis('off')

        # Subplot 2: Updated 3D grid squares after correction
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.scatter(updated_df['x'], updated_df['y'], updated_df['z'], color='blue', marker='o', label='Grid squares')
        ax3.set_title('Grid Squares after Eucentricity Correction')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_zlim(z_min, z_max)
        ax3.legend()

        # Subplot 4: 3D plot with fitted plane
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        zz = a * xx + b * yy + c
        ax4.scatter(df['x'], df['y'], df['z'], color='blue', marker='o', label='Grid squares')
        ax4.plot_surface(xx, yy, zz, color='cyan', alpha=0.5, label='Fitted plane projection')
        ax4.set_title('Eucentricity Plane Fit')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Z')
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        ax4.set_zlim(z_min, z_max)
        ax4.legend()

        # Subplot 5: Y-Z projection holding X constant (2D)
        ax5 = fig.add_subplot(2, 3, 5)
        y_vals = np.linspace(df['y'].min(), df['y'].max(), 100)
        z_plane_x = a * df['x'].mean() + b * y_vals + c
        z_perpendicular_x = df['z'] - (a * df['x'] + c)
        ax5.scatter(df['y'], z_perpendicular_x, color='blue', marker='o', label='Grid squares')
        ax5.plot(y_vals, z_plane_x - (a * df['x'].mean() + c), 'b--', label='Fitted plane projection')
        ax5.set_xlabel('Y')
        ax5.set_ylabel('Z* (perpendicular to X)')
        ax5.set_title('Perpendicular to X Direction')
        ax5.legend()

        # Subplot 6: X-Z projection holding Y constant (2D)
        ax6 = fig.add_subplot(2, 3, 6)
        x_vals = np.linspace(df['x'].min(), df['x'].max(), 100)
        z_plane_y = a * x_vals + b * df['y'].mean() + c
        z_perpendicular_y = df['z'] - (b * df['y'] + c)
        ax6.scatter(df['x'], z_perpendicular_y, color='blue', marker='o', label='Grid squares')
        ax6.plot(x_vals, z_plane_y - (b * df['y'].mean() + c), 'b--', label='Fitted plane projection')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Z** (perpendicular to Y)')
        ax6.set_title('Perpendicular to Y Direction')
        ax6.legend()

        # Adjust scales for projections on vertical axis
        z_min_perp = min(z_perpendicular_x.min(), z_perpendicular_y.min())
        z_max_perp = max(z_perpendicular_x.max(), z_perpendicular_y.max())
        ax5.set_ylim(1.05 * z_min_perp, 1.05 * z_max_perp)
        ax6.set_ylim(1.05 * z_min_perp, 1.05 * z_max_perp)

        plt.tight_layout(pad=4)
        
        try:
            grid_squares_dir = os.path.join(os.getcwd(), 'grid_squares')
            plot_path = os.path.join(grid_squares_dir, 'eucentricity_correction.png')
            plt.savefig(plot_path)
        except Exception as e:
            log_print(f"Error saving plot: {e}", logging.ERROR)
            
    except Exception as e:
        log_print(f"Error creating plots: {e}", logging.ERROR)

def process_coordinates(coordinates: List[dict], expected_num_items: Optional[int]) -> None:
    """Process coordinates and validate count.
    
    Args:
        coordinates: List of coordinate dictionaries
        expected_num_items: Expected number of coordinates
        
    Raises:
        ProcessingError: If coordinate validation fails
    """
    num_coordinates = len(coordinates)
    if not coordinates:
        raise ProcessingError("No coordinates found in log file")
        
    if expected_num_items and expected_num_items != num_coordinates:
        log_print(
            f"Warning: Number of grid square data points ({num_coordinates}) "
            f"does not match expected number ({expected_num_items}).",
            logging.WARNING
        )

def main() -> int:
    """Main execution function for eucentricity correction.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Display banner and initialize logging
        setup_logging('eucentricity_correction.log', 'REyes_logs')
        print_banner()
        
        # Log version information
        log_print(f"\nREyes NEC v{__version__} will find eucentricity plane "
                 f"and correct navigator file\n")
        
        # Locate and validate input files
        try:
            log_file, nav_file = locate_files()
        except FileNotFoundError as e:
            log_print(f"Input file error: {str(e)}", logging.ERROR)
            return 1
            
        # Parse and validate input data
        try:
            coordinates, expected_num_items = parse_log_file(log_file)
            process_coordinates(coordinates, expected_num_items)
        except ProcessingError as e:
            log_print(f"Data validation error: {str(e)}", logging.ERROR)
            return 1
            
        # Perform regression calculations
        try:
            regression_result = perform_regression(coordinates)  # Now returns RegressionResult
            orig_coordinates = extract_nav_coordinates(nav_file)
            updated_coordinates = update_nav_file(
                nav_file,
                regression_result.a,
                regression_result.b,
                regression_result.c
            )
        except Exception as e:
            log_print(f"Calculation error: {str(e)}", logging.ERROR)
            return 1
            
        # Create and save visualizations
        try:
            orig_df = pd.DataFrame(orig_coordinates, columns=['x', 'y', 'z'])
            updated_df = pd.DataFrame(updated_coordinates, columns=['x', 'y', 'z'])
            plot_results(
                orig_df,
                updated_df,
                regression_result.df,
                regression_result.a,
                regression_result.b,
                regression_result.c
            )
        except Exception as e:
            log_print(f"Visualization error: {str(e)}", logging.ERROR)
            log_print("Processing completed with visualization errors")
            return 1
            
        log_print("\nProcessing completed successfully!\n")
        return 0
        
    except Exception as e:
        log_print(f"An unexpected error occurred: {str(e)}", logging.ERROR)
        logging.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())
    



