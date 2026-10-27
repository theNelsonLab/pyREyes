"""
REyes NEC (Navigator Eucentricity Corrector)
A tool for processing electron microscope stage positions and correcting eucentricity tilt.
"""

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # files only; never a GUI backend, even with X11 forwarding
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)
import pandas as pd
from sklearn.linear_model import LinearRegression

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_errors import ProcessingError

__version__ = '3.6.1'
__min_required_version__ = '3.4.0'

# A fitted plane is recorded as `trusted` in eucentricity_plane.json only when
# the fit that produced it is good enough to be worth applying elsewhere. The
# fit quality that counts is the one against the raw fine-eucentricity
# measurements (a handful of points): refitting from the corrected navigator,
# whose Z values were computed from the plane, returns R^2 = 1 tautologically.
PLANE_MIN_R2 = 0.65
PLANE_MIN_POINTS = 4

@dataclass
class RegressionResult:
    """Store regression calculation results."""
    a: float
    b: float
    c: float
    df: pd.DataFrame
    # Fit quality. se_* are None when there are too few points to estimate them
    # (the residual variance needs n > 3 for three parameters).
    se_a: Optional[float] = None
    se_b: Optional[float] = None
    se_c: Optional[float] = None
    r2: Optional[float] = None
    n_points: int = 0
    # Full 3x3 covariance of (a, b, c), row-major. The standard errors above
    # are only its diagonal; se_c is the uncertainty at the ORIGIN, and stage
    # positions sit hundreds of microns out, so the slope terms contribute.
    # With the full matrix the prediction standard error at any (x, y) is
    #     sd(z) = sqrt( [x y 1] . cov . [x y 1]^T )
    covariance: Optional[List[List[float]]] = None

    @property
    def trusted(self) -> bool:
        """Whether this plane is well enough determined to be applied elsewhere."""
        return (self.r2 is not None
                and self.r2 >= PLANE_MIN_R2
                and self.n_points >= PLANE_MIN_POINTS)

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
        r2 = float(reg.score(X, y))
        log_print(f"\nR^2: {r2:.4f}")

        # The standard errors are only meaningful once there are more points
        # than parameters; below that the residual variance divides by zero and
        # std_errors comes back non-finite. Report them as absent rather than
        # as infinity, and let `trusted` fail on n_points.
        se_is_usable = n > 3 and bool(np.all(np.isfinite(std_errors)))

        return RegressionResult(
            a=a, b=b, c=c, df=df,
            se_a=float(std_errors[0]) if se_is_usable else None,
            se_b=float(std_errors[1]) if se_is_usable else None,
            se_c=float(std_errors[2]) if se_is_usable else None,
            r2=r2,
            n_points=n,
            covariance=[[float(v) for v in row] for row in covariance_matrix]
                       if se_is_usable else None,
        )

    except Exception as e:
        raise ProcessingError(f"Error performing regression: {str(e)}") from e


def write_plane_json(result: RegressionResult, output_dir: str) -> str:
    """Write the eucentric plane to `<output_dir>/eucentricity_plane.json`.

    The log line is lossy -- `z = 0.004x + 0.008y + 49.77`, three decimals on
    the slopes and two on the intercept, where a slope may itself be
    0.004 +/- 0.003. The file carries the coefficients at full precision AND
    the fit quality that says whether they may be applied at all. A tool that
    uses the plane should check `trusted` rather than read a, b, c and hope.
    """
    plane = {
        'a': result.a,
        'b': result.b,
        'c': result.c,
        'se_a': result.se_a,
        'se_b': result.se_b,
        'se_c': result.se_c,
        'r2': result.r2,
        'n_points': result.n_points,
        'covariance': result.covariance,
        'trusted': result.trusted,
        'min_r2': PLANE_MIN_R2,
        'min_points': PLANE_MIN_POINTS,
        'equation': 'z = a*x + b*y + c',
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'eucentricity_plane.json')
    with open(path, 'w') as handle:
        json.dump(plane, handle, indent=2, allow_nan=False)
        handle.write('\n')

    log_print(f"\nEucentric plane written to {path}")
    if not result.trusted:
        log_print(
            f"WARNING: plane fit is NOT trusted "
            f"(R^2={result.r2:.4f} needs >= {PLANE_MIN_R2}, "
            f"n={result.n_points} needs >= {PLANE_MIN_POINTS}). "
            f"Tools that use the plane should fall back to the uncorrected Z.",
            logging.WARNING
        )
    return path

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

        # Record the plane for later use. Deliberately NOT fatal: the
        # navigator is already corrected, and a reader of the file falls
        # back to the uncorrected Z when it is absent.
        try:
            write_plane_json(regression_result, os.path.dirname(nav_file))
        except Exception as e:
            log_print(f"Could not write eucentricity plane JSON: {str(e)}",
                      logging.ERROR)

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
    



