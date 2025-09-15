import numpy as np
from typing import List, Tuple, Dict
import re
import os
import matplotlib.pyplot as plt
from pyREyes.lib.REyes_logging import log_print
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


from pyREyes.lib.grid_processing.filtering_options import load_filtering_options
from pyREyes.lib.REyes_microscope_configurations import MicroscopeConfig
from pyREyes.lib.REyes_errors import GridSquareError
from pyREyes.lib.REyes_utils import find_nav_file


FILTERING_OPTIONS = load_filtering_options()

# Helper function for 96-well-plate filtering
def filter_rotated_points_by_box(rotated_coords: np.ndarray, mean_dist: float) -> np.ndarray:
    """
    Filters the rotated coordinates using the bounding box:
    x ∈ [-0.5a, 12.5a], y ∈ [1.5a, 9.5a]
    where a = mean_dist

    Returns:
        Filtered coordinates as a NumPy array.
    """
    a = mean_dist
    x_min, x_max = -0.5 * a, 11.5 * a
    y_min, y_max = 1.5 * a, 9.5 * a

    mask = (
        (rotated_coords[:, 0] >= x_min) & (rotated_coords[:, 0] <= x_max) &
        (rotated_coords[:, 1] >= y_min) & (rotated_coords[:, 1] <= y_max)
    )
    return rotated_coords[mask]

def extract_stage_coordinates_from_item_2(nav_path):
    with open(nav_path, 'r') as f:
        lines = f.readlines()

    item_pattern = re.compile(r"\[Item\s*=\s*(\d+)\]")
    stage_pattern = re.compile(r"StageXYZ\s*=\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")

    current_item = None
    for line in lines:
        item_match = item_pattern.match(line)
        if item_match:
            current_item = int(item_match.group(1))
        elif current_item == 2:
            stage_match = stage_pattern.match(line)
            if stage_match:
                x = float(stage_match.group(1))
                y = float(stage_match.group(2))
                return (x, y)

    return None  # If [Item = 2] or StageXYZ not found


class CentroidProcessor:
    """Handles centroid detection and processing operations."""
    
    def __init__(self, config: MicroscopeConfig):
        self.config = config
    
    def filter_centroids(self, 
                        centroids: List[Tuple[float, float]], 
                        filtering_type: str = "default") -> List[Tuple[float, float]]:
        """
        Filter centroids based on either bounding box constraints or target count.
        
        Args:
            centroids: List of (x, y) coordinate tuples
            filtering_type: Type of filtering to apply ("default", "narrow", or "well-plate", or a numeric target)
            
        Returns:
            Filtered list of centroids
        """
        try:
            if filtering_type not in FILTERING_OPTIONS:
                raise ValueError(f"Unknown filtering type: {filtering_type}")
            
            filtering_value = FILTERING_OPTIONS.get(filtering_type, FILTERING_OPTIONS["default"])
            
            if isinstance(filtering_value, str):
                if filtering_value == "None":
                    center_corner_eucentricity_points = self.find_eucentricity_points(centroids)
                    if len(centroids) < 9:
                        eucentricity_points = center_corner_eucentricity_points
                    else: 
                        additional_points = CentroidProcessor.find_inner_triangle_points(
                            centroids, center_corner_eucentricity_points
                        )
                        eucentricity_points = center_corner_eucentricity_points + additional_points
                    return centroids, center_corner_eucentricity_points
                elif filtering_value == "96-Well-Plate":
                    # Apply any special logic for 96-well plates here (placeholder = use all centroids)
                    centroids, eucentricity_points = self._filter_by_well_plate(centroids)
                    return centroids, eucentricity_points

            
            elif isinstance(filtering_value, int):
                if filtering_value < 3: 
                    # Use 3x3 grid for eucentricity points when it is 1x1 or 2x2 grid
                    three_by_three_centroids, eucentricity_points = self._filter_by_grid_selection(centroids, 3)
                    centroids, _ = self._filter_by_grid_selection(centroids, filtering_value)
                else: 
                    centroids, eucentricity_points = self._filter_by_grid_selection(centroids, filtering_value)
                    # Get eucentricity points normally
                    if filtering_value > 8:
                        additional_points = CentroidProcessor.find_inner_triangle_points(
                            centroids, eucentricity_points
                        )
                        eucentricity_points = eucentricity_points + additional_points
                return centroids, eucentricity_points

            else:
                raise GridSquareError(
                    f"Unsupported filtering value type for '{filtering_type}': {type(filtering_value)}"
                )
            
        except Exception as e:
            raise GridSquareError(f"Error filtering centroids: {str(e)}") from e

    def _filter_by_well_plate(self, centroids: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        For 96-Well-Plate filtering:
        - Use the user-selected point (Item = 2) as the reference.
        - Align, rotate, and filter the centroids using KDE and neighbor distance analysis.
        - Extract 9 eucentricity points from rotated space and map them to original coordinates.
        
        Returns:
            filtered_original: All centroids inside bounding box (original coordinates)
            eucentricity_original: 9 eucentricity points (original coordinates)
        """
        try:
            # 1. Load reference point from NAV file
            nav_file = find_nav_file()
            if nav_file is None:
                raise GridSquareError("No .nav file found in current directory")

            reference_point = extract_stage_coordinates_from_item_2(nav_file)
            if reference_point is None:
                raise GridSquareError("StageXYZ from [Item = 2] not found in NAV file")

            # 2. Find the closest centroid to the reference
            closest = min(centroids, key=lambda pt: (pt[0] - reference_point[0])**2 + (pt[1] - reference_point[1])**2)

            # 3. Analyze nearest neighbor angles
            angle_dict = self.compute_neighbor_distances_and_angles(centroids)
            mean_distance = self.robust_mean_combined_distances(angle_dict)

            theta = self._estimate_statistical_rotation_angle(centroids)

            dx, dy = closest[0], closest[1]

            if theta <= np.pi / 4:
                if dx >= 0 and dy >= 0:
                    quadrant_adjust = np.pi           # 2 * π/2
                elif dx < 0 and dy >= 0:
                    quadrant_adjust = np.pi / 2       # 1 * π/2
                elif dx < 0 and dy < 0:
                    quadrant_adjust = 0               # 0 * π/2
                else:  # dx >= 0 and dy < 0
                    quadrant_adjust = 3 * np.pi / 2   # 3 * π/2
            else:
                if dx >= 0 and dy >= 0:
                    quadrant_adjust = np.pi / 2
                elif dx < 0 and dy >= 0:
                    quadrant_adjust = 0
                elif dx < 0 and dy < 0:
                    quadrant_adjust = 3 * np.pi / 2
                else:  # dx >= 0 and dy < 0
                    quadrant_adjust = np.pi

            rotation_angle = theta + quadrant_adjust

            # 5. Translate and rotate all centroids
            coords = np.array(centroids)
            shifted = coords - np.array(closest)
            rot_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle)],
                [np.sin(rotation_angle),  np.cos(rotation_angle)]
            ])
            rotated = shifted @ rot_matrix.T

            # 6. Filter rotated centroids within bounding box
            filtered_rotated = filter_rotated_points_by_box(rotated, mean_distance)

            # 7. Map filtered rotated centroids back to original space
            filtered_original = [
                tuple(orig) for orig, rot in zip(coords, rotated)
                if any(np.allclose(rot, fr, atol=1e-6) for fr in filtered_rotated)
            ]

            # 8. Get 9 rotated eucentricity points and map them back to original space
            eucentricity_rotated = self._get_eucentricity_points_from_rotated(filtered_rotated)
            eucentricity_original = [
                tuple(orig) for orig, rot in zip(coords, rotated)
                if any(np.allclose(rot, ep, atol=1e-6) for ep in eucentricity_rotated)
            ]

            return filtered_original, eucentricity_original

        except Exception as e:
            raise GridSquareError(f"Error in 96-Well-Plate filtering: {str(e)}") from e


    def _get_eucentricity_points_from_rotated(self, filtered_rotated: np.ndarray) -> List[np.ndarray]:
        """
        Used in 96-Well-Plate filtering.
        From rotated filtered coordinates, return 9 points:
        - Center point (closest to center of mass of 4 corners)
        - 4 corners (top-left, top-right, bottom-left, bottom-right)
        - 4 edge-center triangle points (top, bottom, left, right)
        """
        x_vals = filtered_rotated[:, 0]
        y_vals = filtered_rotated[:, 1]

        # 1. Find 4 corners
        top_left = filtered_rotated[np.argmin(x_vals + y_vals)]
        top_right = filtered_rotated[np.argmin(-x_vals + y_vals)]
        bottom_left = filtered_rotated[np.argmin(x_vals - y_vals)]
        bottom_right = filtered_rotated[np.argmin(-x_vals - y_vals)]
        corners = [top_left, top_right, bottom_left, bottom_right]

        # 2. Compute center of mass of corners → center point
        center_mass = np.mean(corners, axis=0)
        center_point = filtered_rotated[np.argmin(np.linalg.norm(filtered_rotated - center_mass, axis=1))]

        # 3. Triangle centers
        triangle_centers = [
            np.mean([top_left, top_right], axis=0),                    # top
            np.mean([bottom_left, bottom_right], axis=0),              # bottom
            np.mean([top_left, center_point, bottom_left], axis=0),    # left
            np.mean([top_right, center_point, bottom_right], axis=0)   # right
        ]

        triangle_points = []
        for tri_center in triangle_centers:
            dists = np.linalg.norm(filtered_rotated - tri_center, axis=1)
            closest = filtered_rotated[np.argmin(dists)]
            triangle_points.append(closest)

        return [center_point] + corners + triangle_points

    def robust_mean_combined_distances(self, angle_dict: Dict[Tuple[float, float], List[Tuple[float, float]]]) -> float:
        """
        Combine all neighbor distances (first and second closest) into one list,
        then iteratively remove values more than 1 standard deviation from the mean.
        
        Args:
            angle_dict: Output from compute_neighbor_distances_and_angles

        Returns:
            Final mean distance after iterative 1-std filtering
        """

        # Flatten all distances from both neighbors
        all_dists = np.array([d for pair in angle_dict.values() for d, _ in pair])

        prev_len = -1
        while prev_len != len(all_dists):
            prev_len = len(all_dists)
            mean = np.mean(all_dists)
            std = np.std(all_dists)
            all_dists = all_dists[np.abs(all_dists - mean) <= std]

        final_mean = np.mean(all_dists)
        return final_mean


    def find_two_distinct_peaks(self, x_vals, y_vals, tolerance=5):
        """
        Finds two distinct peaks in a KDE curve, accounting for 180° symmetry.

        Args:
            x_vals: np.ndarray of x-axis (angle in degrees).
            y_vals: np.ndarray of KDE values.
            tolerance: angular threshold (in degrees) to treat peaks as duplicates modulo 180.

        Returns:
            dominant_angles: list of two distinct peak angles (in degrees)
        """

        def _ang_dist_180(a, b):
            """Shortest distance between angles a,b on a 180° circle (degrees)."""
            d = abs(a - b) % 180.0
            return min(d, 180.0 - d)

        # Find all peaks
        peak_indices, _ = find_peaks(y_vals, prominence=0.001)
        peak_angles = x_vals[peak_indices]
        peak_heights = y_vals[peak_indices]

        # Fold to [0, 180) to avoid edge duplication 
        peak_angles = peak_angles % 180 


        # Sort peaks by descending height
        sorted_indices = np.argsort(-peak_heights)
        distinct_peaks = []

        for idx in sorted_indices:
            angle = peak_angles[idx]

            # Check for duplicates modulo 180 within tolerance
            is_duplicate = any(_ang_dist_180(angle, prev) < tolerance for prev in distinct_peaks)
            if not is_duplicate:
                distinct_peaks.append(angle)
            if len(distinct_peaks) == 2:
                break

        if len(distinct_peaks) < 2:
            raise GridSquareError(f"Failed to detect two distinct peaks. Found: {distinct_peaks}")

        # Fold final output into [0, 180)
        folded_peaks = [angle % 180 for angle in distinct_peaks]

        return sorted(folded_peaks)


    def compute_neighbor_distances_and_angles(
        self, centroids: List[Tuple[float, float]]
    ) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
        """
        For each centroid, find its 2 closest neighbors and compute:
        - Distance to each neighbor
        - Angle (in degrees) between the connecting line and the Y-axis, normalized to [0°, 180°]

        Returns:
            A dictionary mapping each centroid to a list of two (distance, angle) tuples.
        """
        import math
        distances_and_angles = {}

        for i, c0 in enumerate(centroids):
            dists = []
            for j, c1 in enumerate(centroids):
                if i == j:
                    continue
                dx = c1[0] - c0[0]
                dy = c1[1] - c0[1]
                dist = math.hypot(dx, dy)

                # Compute angle relative to y-axis using atan2 and wrap to [0, 180)
                angle_rad = math.atan2(dx, dy)  # Note: dx, dy order for y-axis reference
                angle_deg = math.degrees(angle_rad) % 180

                dists.append((dist, angle_deg, c1))

            # Sort by distance and take 2 closest neighbors
            dists.sort(key=lambda tup: tup[0])
            closest_two = dists[:2]

            distances_and_angles[c0] = [(d, a) for d, a, _ in closest_two]

        return distances_and_angles

    def _estimate_statistical_rotation_angle(
        self,
        centroids: List[Tuple[float, float]],
        plot: bool = False,
        fig: plt.Figure = None,
        bins: int = 45
    ) -> float:
        """
        Estimate a global rotation angle (radians) from centroid neighbor orientations
        using KDE + two dominant peaks. No quadrant adjustment applied here.
        """
        # Reuse existing utilities
        angle_dict = self.compute_neighbor_distances_and_angles(centroids)

        # Pool angles and extend by ±180 for periodicity
        all_angles = [angle for pair in angle_dict.values() for _, angle in pair]
        extended_angles = all_angles + [a + 180 for a in all_angles] + [a - 180 for a in all_angles]

        kde = gaussian_kde(extended_angles, bw_method=0.01)
        x_vals = np.linspace(-90, 270, 1000)
        y_vals = kde(x_vals)

        small_angle, large_angle = self.find_two_distinct_peaks(x_vals, y_vals)
        if plot:
            print("the large and small angles are", large_angle, small_angle)

        # Same formula as well-plate (without quadrant correction)
        rotation_angle = np.radians((large_angle - 90 + small_angle) / 2)

        # To debug to see if the anles make sense
        if plot:
            if fig is None:
                fig = plt.figure(figsize=(12, 4))
            axs = fig.subplots(1, 2)

            # 1) Histogram on [0, 180)
            ax0 = axs[0]
            ax0.hist(all_angles, bins=bins, range=(0, 180), edgecolor="black")
            ax0.set_title("Neighbor angle distribution (folded to [0°, 180°))")
            ax0.set_xlabel("Angle (degrees, vs Y-axis)")
            ax0.set_ylabel("Count")
            ax0.set_xlim(0, 180)
            ax0.grid(True, alpha=0.3)

            # 2) KDE over [-90, 270] with peaks and rotation annotation
            ax1 = axs[1]
            ax1.plot(x_vals, y_vals, linewidth=2)
            ax1.set_title("KDE of angles (extended for periodicity)")
            ax1.set_xlabel("Angle (degrees)")
            ax1.set_ylabel("Density")
            ax1.set_xlim(-90, 270)
            ax1.grid(True, alpha=0.3)

            # mark peaks
            ax1.axvline(small_angle, linestyle="--", linewidth=1.5)
            ax1.axvline(large_angle, linestyle="--", linewidth=1.5)
            ax1.text(small_angle, max(y_vals)*0.9, f"small={small_angle:.1f}°", rotation=90,
                    va="top", ha="right")
            ax1.text(large_angle, max(y_vals)*0.9, f"large={large_angle:.1f}°", rotation=90,
                    va="top", ha="right")

            # annotate rotation
            ax1.text(0.02, 0.95,
                    f"rotation = {(np.degrees(rotation_angle)):.2f}°",
                    transform=ax1.transAxes, ha="left", va="top",
                    bbox=dict(boxstyle="round", alpha=0.1, ec="none"))

            fig.tight_layout()
            fig.savefig("angle_kde_debug.png", dpi=200, bbox_inches="tight")

        return rotation_angle   


    ### Grid Selection Filtering ###

    def _filter_by_grid_selection(self, centroids: List[Tuple[float, float]], target_count: int) -> List[Tuple[float, float]]:
        coordinates = np.array(centroids, dtype=np.float64)

        # Translate the coordinates to center the point closest to (0,0) at (0,0)
        distances = np.linalg.norm(coordinates, axis=1)
        closest_index = np.argmin(distances)
        closest_point = coordinates[closest_index]
        translated_coordinates = coordinates - closest_point
        origin = np.array([0.0, 0.0])

        # Step 1: compute distances (excluding the origin point)
        nonzero_mask = np.any(translated_coordinates != origin, axis=1)
        nonzero_points = translated_coordinates[nonzero_mask]
        nonzero_dists = np.linalg.norm(nonzero_points, axis=1)

        # Step 2: find 4 closest non-origin neighbors and get distances of 4 closest neighbors 
        sorted_indices = np.argsort(nonzero_dists)
        nearest4 = nonzero_points[sorted_indices[:4]]
        neighbor_distances = np.linalg.norm(nearest4, axis=1)
        avg_neighbor_dist = np.mean(neighbor_distances)  # average spacing unit

        # Step 3 (new): estimate a GLOBAL rotation angle statistically (KDE + two peaks)
        rotation_angle = self._estimate_statistical_rotation_angle(centroids)

        # Step 4 (unchanged interface): build rotation matrix (helper returns R(-θ); we apply .T later → net rotation +θ)
        rotation_matrix = self._compute_rotation_matrix(rotation_angle)

        # Step 5: select points depending on odd/even target_count
        if target_count % 2 != 0: # odd target count (e.g., 3x3, 5x5)
            # Rorate the already centered coodinates so that the orientation alisng with the axises
            rotated_coordinates = translated_coordinates @ rotation_matrix.T
            # Since the coordinates are already centered, the distance from origin
            # to the farthest point is (n - 1)/2 * avg_neighbor_dist
            # So if we do n/2 * avg_neighbor_dist it goes to the
            # middle between the farthest point and the next point
            radius = (target_count / 2) * avg_neighbor_dist

            # Select points inside square of width `radius * 2` centered at (0, 0)
            in_square_mask = np.logical_and.reduce((
                np.abs(rotated_coordinates[:, 0]) <= radius,
                np.abs(rotated_coordinates[:, 1]) <= radius
            ))

            # Map square points back to original coordinates
            square_points_original = coordinates[in_square_mask]

        else: # even target count (e.g., 2x2, 4x4)
            rotated_coordinates = coordinates @ rotation_matrix.T

            if rotated_coordinates.shape[0] < 4:
                raise ValueError("Need at least 4 coordinates to proceed, but got fewer.")

            # Iteratively expand square until we get enough points (e.g., 4x4 = 16 points)
            # Find 4 closest points by iteratively expanding radius
            step = avg_neighbor_dist / 10  # small increment step
            margin = avg_neighbor_dist     # initial guess

            initial_4_mask = self._expand_until_four(rotated_coordinates, margin, step)

            first4_points = rotated_coordinates[initial_4_mask][:4]
            center_of_mass = np.mean(first4_points, axis=0)

            # Translate all rotated coordinates to move center_of_mass to origin
            recentered_coordinates = rotated_coordinates - center_of_mass
            radius = (target_count / 2) * avg_neighbor_dist

            in_square_mask = np.logical_and.reduce((
                np.abs(recentered_coordinates[:, 0]) <= radius,
                np.abs(recentered_coordinates[:, 1]) <= radius
            ))

            # Grab original coordinates of selected points
            square_points_original = coordinates[in_square_mask]

        # selected originals
        sel_orig = coordinates[in_square_mask]

        # --- pick corners from the ROTATED subset (not recentered) ---
        rotated_subset = rotated_coordinates[in_square_mask]
        x, y = rotated_subset[:, 0], rotated_subset[:, 1]


        idx_center = int(np.argmin(np.linalg.norm(rotated_subset, axis=1)))  # closest to origin
        idx_tl = int(np.argmin(x + y))
        idx_tr = int(np.argmin(-x + y))
        idx_bl = int(np.argmin(x - y))
        idx_br = int(np.argmin(-x - y))

        eucentricity_points = [
            tuple(sel_orig[idx_center].tolist()),  # center
            tuple(sel_orig[idx_tl].tolist()),      # top-left
            tuple(sel_orig[idx_tr].tolist()),      # top-right
            tuple(sel_orig[idx_bl].tolist()),      # bottom-left
            tuple(sel_orig[idx_br].tolist()),      # bottom-right
        ]

        centroids = [tuple(row) for row in square_points_original]
        return centroids, eucentricity_points

    def _compute_rotation_matrix(self, mean_angle: float) -> np.ndarray:
        cos_theta = np.cos(mean_angle)
        sin_theta = np.sin(mean_angle)
        return np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])

    def _expand_until_four(self, coords: np.ndarray, margin: float, step: float, max_iters: int = 1000) -> np.ndarray:
            """Iteratively expand the radius until at least 4 points are found inside square bounds."""
            for i in range(max_iters):
                radius = margin + step * i
                in_square_mask = np.logical_and.reduce((
                    np.abs(coords[:, 0]) <= radius,
                    np.abs(coords[:, 1]) <= radius
                ))
                if np.sum(in_square_mask) >= 4:
                    return in_square_mask
            raise RuntimeError("Could not find at least 4 points after expanding search area.")

    ### Sorting Centroids ###
    def sort_centroids(self, 
                      centroids: List[Tuple[float, float]], 
                      y_tolerance: float = 25) -> List[Tuple[float, float]]:
        """
        Sort centroids by Y-bands and X-position.
        
        Args:
            centroids: List of (x,y) coordinate tuples
            y_tolerance: Tolerance for grouping Y coordinates
            
        Returns:
            Sorted list of centroids
        """
        try:
            def sort_key(coord: Tuple[float, float]) -> Tuple[int, float]:
                y_band = round(coord[1] / y_tolerance)
                return (-y_band, coord[0])
            
            return sorted(centroids, key=sort_key)
            
        except Exception as e:
            raise GridSquareError(f"Error sorting centroids: {str(e)}") from e

    ### Eucentricity Points ###

    def find_eucentricity_points(
        self, centroids: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Find eucentricity points: the centroid closest to origin and the farthest points in each quadrant.

        Args:
            centroids: List of (x, y) centroid coordinates

        Returns:
            List of unique eucentricity points (1 closest + up to 4 farthest from each quadrant)
        """
        try:
            closest = self._find_closest_to_origin(centroids)
            farthest_quadrant_points = self._find_farthest_in_quadrants(centroids)

            # Avoid duplicates (e.g., if closest is also one of the farthest)
            eucentricity_points = [closest]
            for pt in farthest_quadrant_points:
                if pt != closest:
                    eucentricity_points.append(pt)

            return eucentricity_points

        except Exception as e:
            raise GridSquareError("Error computing eucentricity points") from e


    @staticmethod
    def _find_closest_to_origin(centroids: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Find centroid closest to origin (0,0)."""
        try:
            if not centroids:
                raise GridSquareError("Cannot find closest point from empty centroids list")
                
            return min(
                centroids,
                key=lambda point: (point[0] * point[0] + point[1] * point[1])
            )
            
        except Exception as e:
            raise GridSquareError("Error finding closest point to origin") from e

    @staticmethod
    def _find_farthest_in_quadrants(
        centroids: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Find farthest centroid in each quadrant from origin (0,0)."""
        try:
            quadrants: Dict[str, List[Tuple[float, float]]] = {
                "Q1": [],  # (+x, +y)
                "Q2": [],  # (-x, +y)
                "Q3": [],  # (-x, -y)
                "Q4": []   # (+x, -y)
            }
            
            for x, y in centroids:
                if x >= 0 and y >= 0:
                    quadrants["Q1"].append((x, y))
                elif x < 0 and y >= 0:
                    quadrants["Q2"].append((x, y))
                elif x < 0 and y < 0:
                    quadrants["Q3"].append((x, y))
                elif x >= 0 and y < 0:
                    quadrants["Q4"].append((x, y))
            
            farthest_points = []
            for points in quadrants.values():
                if points:
                    farthest = max(
                        points,
                        key=lambda point: (point[0] * point[0] + point[1] * point[1])
                    )
                    farthest_points.append(farthest)
            
            return farthest_points
            
        except Exception as e:
            raise GridSquareError("Error finding farthest points in quadrants") from e


    def find_inner_triangle_points(
        centroids: List[Tuple[float, float]],
        eucentricity_points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Given 5 eucentricity points (center + 4 corners), find 4 triangle centers,
        then find the closest actual centroids to those triangle centers.
        
        Args:
            centroids: List of all centroids.
            eucentricity_points: List of 5 points [p0 (center), p1, p2, p3, p4]
        
        Returns:
            List of 4 centroids closest to triangle centers.
        """
        if len(eucentricity_points) != 5:
            raise ValueError("Exactly 5 eucentricity points are required: [p0, p1, p2, p3, p4]")
        
        p0, p1, p2, p3, p4 = [np.array(p) for p in eucentricity_points]

        triangle_centers = [
            np.mean([p0, p1, p2], axis=0),
            np.mean([p0, p2, p3], axis=0),
            np.mean([p0, p3, p4], axis=0),
            np.mean([p0, p4, p1], axis=0)
        ]

        def closest_point(center: np.ndarray) -> Tuple[float, float]:
            return min(centroids, key=lambda pt: np.sum((np.array(pt) - center)**2))

        return [closest_point(center) for center in triangle_centers]