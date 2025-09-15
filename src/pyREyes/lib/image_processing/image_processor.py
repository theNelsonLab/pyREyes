import os
import random
import numpy as np
from pyREyes.lib.REyes_errors import GridSquareError
from pyREyes.lib.grid_processing.GridProcessor import GridProcessor
from pyREyes.lib.image_processing.PlottingManager import PlottingManager
from scipy.ndimage import rotate, label
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Tuple, List
from shapely.geometry import Polygon, MultiPolygon, Point



class ImageProcessor:
    """Handles image processing operations for grid square detection."""
    def __init__(self, debug: bool = False):
        self.debug = False

    @staticmethod
    def calculate_normalized_areas(image: np.ndarray, dark_threshold=0.3):
        """
        Computes the normalized area of dark regions in an image and 
        returns a binary image where dark regions are 0 (black), bright regions are 255 (white),
        and NaNs are preserved.

        Args:
            image (np.ndarray): Input image array (can contain NaNs).
            dark_threshold (float): Intensity threshold for dark regions (default = 0.3).

        Returns:
            tuple: (normalized_dark_area, normalized_bright_area, binary_image)
        """
        try:
            # Normalize the image to [0, 1], ignoring NaNs
            image_min, image_max = np.nanmin(image), np.nanmax(image)
            normalized_image = (image - image_min) / (image_max - image_min)

            # Mask valid (non-NaN) pixels
            valid_mask = ~np.isnan(normalized_image)

            # Classify dark and bright pixels (only where valid)
            dark_pixels = (normalized_image < dark_threshold) & valid_mask
            bright_pixels = valid_mask & ~dark_pixels

            # Compute normalized areas
            dark_area = np.sum(dark_pixels)
            bright_area = np.sum(bright_pixels)

            # Create binary image (0 = dark, 255 = bright, NaN stays NaN)
            binary_image = np.full(image.shape, np.nan, dtype=np.float32)
            binary_image[dark_pixels] = 0.0
            binary_image[bright_pixels] = 255.0

            return dark_area, bright_area, binary_image.astype(np.float32)

        except Exception as e:
            raise GridSquareError(f"Error calculating normalized areas: {str(e)}") from e

    @staticmethod
    def find_connected_regions(image: np.ndarray):
        """
        Find connected regions in a binary image, removing all regions smaller than 10% of the largest.

        Args:
            image (np.ndarray): Binary image where 0 = background, 1 = foreground (regions).

        Returns:
            tuple:
                - np.ndarray: Labeled image with small regions removed.
                - int: Number of detected regions above threshold.
                - np.ndarray: Array of region sizes (index = region label).
        """
        try:
            # Ensure binary format: 1 for regions, 0 for background
            binary_image = (image != 0) & np.isfinite(image)

            # Label connected components
            labeled_image, num_features = label(binary_image)

            # Count pixel size of each region
            region_sizes = np.bincount(labeled_image.ravel())

            if len(region_sizes) <= 1:
                return labeled_image, 0, region_sizes  # Only background

            # Determine threshold as 1% of the largest region (excluding background)
            max_region_size = np.max(region_sizes[1:])  # Skip background label 0
            dynamic_threshold = max_region_size * 0.01

            # Remove small regions by setting them to 0 (background)
            small_region_labels = np.where(region_sizes < dynamic_threshold)[0]
            remove_mask = np.isin(labeled_image, small_region_labels)
            labeled_image[remove_mask] = 0

            # Relabel after removing small regions
            new_labeled_image, new_num_features = label(labeled_image > 0)

            # Get updated region sizes
            new_region_sizes = np.bincount(new_labeled_image.ravel())

            # --- NEW: Remove overly large regions (> mean + 2 * std), excluding background ---
            if len(new_region_sizes) > 1:
                nonzero_sizes = new_region_sizes[1:]  # Exclude background label 0
                mean_size = np.mean(nonzero_sizes)
                std_size = np.std(nonzero_sizes)
                upper_threshold = mean_size + 2 * std_size

                large_region_labels = np.where(new_region_sizes > upper_threshold)[0]
                large_region_labels = large_region_labels[large_region_labels != 0]  # Avoid label 0
                remove_mask = np.isin(new_labeled_image, large_region_labels)
                new_labeled_image[remove_mask] = 0

                # Final relabeling
                new_labeled_image, new_num_features = label(new_labeled_image > 0)
                new_region_sizes = np.bincount(new_labeled_image.ravel())


                return new_labeled_image, new_num_features, new_region_sizes

        except Exception as e:
            raise GridSquareError(f"Error finding connected regions: {str(e)}") from e


    @staticmethod
    def compute_strip_and_spacing(
        merged_polygon,
        num_regions: int,
        bright_area: float,
        dark_area: float
    ) -> Tuple[float, float, Polygon]:
        """
        Compute bar width, space width, and strip region from a merged grid polygon.

        Args:
            merged_polygon (shapely.geometry.Polygon): The union of all frame polygons.
            num_regions (int): Number of detected grid regions.
            bright_area (float): Total bright area from binary image.
            dark_area (float): Total dark area from binary image.

        Returns:
            Tuple containing:
                - space_width (float): Estimated spacing between grid bars.
                - bar_width (float): Estimated width of grid bars.
                - strip (Polygon): Shapely polygon representing the edge strip region.
        """
        if num_regions == 0:
            raise ValueError("Number of regions must be > 0 to compute unit width.")

        total_area = merged_polygon.area
        unit_width = np.sqrt(total_area / num_regions)  # Average width of a grid square

        ratio = GridProcessor.compute_grid_bar_ratio(bright_area / dark_area)
        bar_width = unit_width - (unit_width / (ratio + 1))
        space_width = unit_width - bar_width

        inner_region = merged_polygon.buffer(-bar_width)
        strip = merged_polygon.difference(inner_region)

        return space_width, bar_width, strip


    @staticmethod
    def plot_labeled_regions_random_colors(labeled_image: np.ndarray, num_features: int, save_path="labeled_regions.png"):
        """
        Plot labeled regions with distinct colors and save the image.

        Args:
            labeled_image (np.ndarray): Image with labeled regions.
            num_features (int): Number of detected regions.
            save_path (str): Path to save the labeled image.
        """
        try:
            # Generate a unique color map for every region (0 is always black for background)
            colors = [(0, 0, 0)] + [plt.cm.jet(random.random()) for _ in range(num_features)]  
            cmap = ListedColormap(colors)

            plt.figure(figsize=(6, 6))
            plt.imshow(labeled_image, cmap=cmap, origin="upper")
            plt.colorbar(label="Region Label")
            plt.title(f"Detected Connected Regions ({num_features} regions)")
            plt.axis("off")

            # Save the image
            save_path = os.path.join(os.getcwd(), save_path)
            plt.savefig(save_path, dpi=300)
            plt.close()
            
            print(f"Labeled region image saved at: {save_path}")

        except Exception as e:
            raise GridSquareError(f"Error plotting labeled regions: {str(e)}") from e
        
    staticmethod
    def plot_labeled_regions(labeled_image: np.ndarray, num_features: int, save_path="labeled_regions.png"):
        """
        Plot labeled regions with custom coloring:
        - Background (0) → black
        - Normal labels (1–999) → red
        - Edge labels (1000+) → white

        Args:
            labeled_image (np.ndarray): Image with labeled regions.
            num_features (int): Number of detected regions.
            save_path (str): Path to save the labeled image.
        """
        try:
            # Create an RGB image to manually set color per pixel
            height, width = labeled_image.shape[:2]
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Assign colors
            rgb_image[labeled_image == 0] = [0, 0, 0]             # Background → black
            rgb_image[(labeled_image > 0) & (labeled_image < 1000)] = [255, 0, 0]   # Regular → red
            rgb_image[labeled_image >= 1000] = [255, 255, 255]     # Edge regions → white

            # Plot
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb_image, origin="upper")
            plt.title(f"Custom Labeled Regions ({num_features} regions)")
            plt.axis("off")

            save_path = os.path.join(os.getcwd(), save_path)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Labeled region image saved at: {save_path}")

        except Exception as e:
            raise GridSquareError(f"Error plotting labeled regions: {str(e)}") from e
    
    @staticmethod
    def get_physical_coordinates(image: np.ndarray, extent: Tuple[float, float, float, float]):
        H, W = image.shape[:2]
        xmin, xmax, ymin, ymax = extent

        # Create pixel grids (each value corresponds to a pixel center)
        x = np.linspace(xmin, xmax, W)
        y = np.linspace(ymax, ymin, H)

        # Create meshgrid of coordinates
        X, Y = np.meshgrid(x, y)

        return X, Y  # X and Y are each (H, W), physical coordinates

    @staticmethod
    def remove_edge_touching_regions(
        labeled_image: np.ndarray,
        extent,
        region_sizes: np.ndarray,
        strip: Polygon,
        distinct_label_offset: int = 1000
    ):
        """
        Removes edge-touching regions from the labeled image if their size is less than half
        the size of the largest region. Large edge-touching regions are assigned distinct labels
        using an offset.

        Args:
            labeled_image (np.ndarray): Labeled image where each region has a unique label > 0.
            extent (tuple): (xmin, xmax, ymin, ymax) for coordinate mapping.
            region_sizes (np.ndarray): Array of region sizes (index = region label).
            strip (Polygon): Shapely polygon defining the edge region.
            distinct_label_offset (int): Label offset for large edge-touching regions.

        Returns:
            tuple:
                - np.ndarray: Labeled image with updated regions.
                - int: Number of compactly labeled (normal) regions.
        """
        try:
            output = labeled_image.copy()
            height, width = output.shape

            # Map pixels to physical coordinates
            X, Y = ImageProcessor.get_physical_coordinates(output, extent)
            coords = np.stack([X.ravel(), Y.ravel()], axis=1)
            edge_mask = np.array([strip.contains(Point(x, y)) for x, y in coords], dtype=bool).reshape(height, width)

            # Get edge-touching labels
            edge_labels = np.unique(output[edge_mask])
            edge_labels = edge_labels[edge_labels != 0]

            # Compute size threshold
            if len(region_sizes) > 1:
                max_region_size = np.max(region_sizes[1:])
            else:
                max_region_size = 0

            size_threshold = max_region_size / 2.0

            # Relabel edge regions based on size
            current_offset = distinct_label_offset
            for label_val in edge_labels:
                if label_val >= len(region_sizes):
                    continue  # Skip out-of-bounds (shouldn't happen, but safe)
                if region_sizes[label_val] >= size_threshold:
                    output[output == label_val] = current_offset
                    current_offset += 1
                else:
                    output[output == label_val] = 0  # Remove small edge-touching regions

            # Relabel normal regions (label < offset) compactly
            normal_mask = (output > 0) & (output < distinct_label_offset)
            relabeled, num_features = label(normal_mask)
            output[normal_mask] = relabeled[normal_mask]

            total_labeled = num_features + (current_offset - distinct_label_offset)
            return output, total_labeled

        except Exception as e:
            raise GridSquareError(f"Error removing edge-touching regions: {str(e)}") from e


    def mask_image_outside_polygon(image: np.ndarray, extent, polygon: Polygon) -> np.ndarray:
        """
        Sets pixels outside the given polygon to NaN based on physical coordinates.

        Args:
            image (np.ndarray): 2D input image (grayscale or labeled).
            extent (tuple): (xmin, xmax, ymin, ymax) of the image.
            polygon (Polygon): Shapely polygon in physical coordinates.

        Returns:
            np.ndarray: Image with pixels outside the polygon set to NaN.
        """
        try:
            height, width = image.shape
            X, Y = ImageProcessor.get_physical_coordinates(image, extent)  # Shape: (H, W)

            coords = np.stack([X.ravel(), Y.ravel()], axis=1)
            inside_mask = np.array([polygon.contains(Point(x, y)) for x, y in coords], dtype=bool)
            inside_mask = inside_mask.reshape((height, width))

            masked_image = image.copy()
            masked_image[~inside_mask] = np.nan

            return masked_image

        except Exception as e:
            raise GridSquareError(f"Error masking image with polygon: {str(e)}") from e


    def merge_small_regions_centroids(image, 
                extent, 
                space_width = None, 
                area_threshold_ratio=0.1, 
                std_merge_threshold = 2.5, 
                distinct_label_offset: int = 1000, 
                binary_image=None, 
                debug = False,
                plotting_manager=None):
        # Works when the img is already labeled 
        X_grid, Y_grid = ImageProcessor.get_physical_coordinates(image, extent)
        X = X_grid.ravel()
        Y = Y_grid.ravel()
        labels = image.ravel().copy()

        if debug:
            centroids = {}
            areas = {}
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]  # Exclude background

            # Compute centroids and area
            for label_val in unique_labels:
                mask = labels == label_val
                if np.any(mask):
                    x_mean = np.mean(X[mask])
                    y_mean = np.mean(Y[mask])
                    centroids[label_val] = (x_mean, y_mean)
                    areas[label_val] = np.sum(mask)


            centroid_list = list(centroids.values())
            plotting_manager.plot_montage_from_image(binary_image, extent, centroid_list, save_path="centroids_pre_size_merge.png")
            

        while True:
            centroids = {}
            areas = {}
            
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels != 0]  # Exclude background

            # Compute centroids and area
            for label_val in unique_labels:
                mask = labels == label_val
                if np.any(mask):
                    x_mean = np.mean(X[mask])
                    y_mean = np.mean(Y[mask])
                    centroids[label_val] = (x_mean, y_mean)
                    areas[label_val] = np.sum(mask)

            if not areas:
                break

            max_area = max(areas.values())
            small_labels = [lbl for lbl, area in areas.items() if area < area_threshold_ratio * max_area]

            if not small_labels:
                break  # No more small regions to merge

            centroid_coords = {lbl: centroids[lbl] for lbl in small_labels}
            all_other_coords = {lbl: centroids[lbl] for lbl in centroids if lbl != 0}

            for small_label in small_labels:
                if small_label not in centroid_coords:
                    continue  # This label may have been merged already

                small_pt = np.array(centroid_coords[small_label]).reshape(1, -1)
                other_labels = [lbl for lbl in all_other_coords if lbl != small_label]
                if not other_labels:
                    continue

                other_pts = np.array([all_other_coords[lbl] for lbl in other_labels])
                dists = cdist(small_pt, other_pts)[0]
                min_idx = np.argmin(dists)
                closest_label = other_labels[min_idx]

                # Merge the two labels
                mask_small = labels == small_label
                mask_other = labels == closest_label
                labels[mask_small | mask_other] = closest_label



        if debug:
            centroid_list = list(centroids.values())
            plotting_manager.plot_montage_from_image(binary_image, extent, centroid_list, save_path="centroids_pre_std_merge.png")

        
        centroid_coords = np.array(list(centroids.values()))
        if len(centroid_coords) < 2:
            return {}  

        # Compute initial distance matrix and stats just once
        dist_matrix = cdist(centroid_coords, centroid_coords)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=1)
        std_dev = np.std(nearest_distances)
        
        iteration = 0
        
        # Start second loop to merge pairs that are statistically close. 
        while True:
            centroids = {}
            label_list = np.unique(labels)
            label_list = label_list[label_list != 0]

            for label_val in label_list:
                mask = labels == label_val
                centroids[label_val] = (np.mean(X[mask]), np.mean(Y[mask]))

            centroid_coords = np.array(list(centroids.values()))
            if len(centroid_coords) < 2:
                break

            dist_matrix = cdist(centroid_coords, centroid_coords)
            np.fill_diagonal(dist_matrix, np.inf)
            nearest_distances = np.min(dist_matrix, axis=1)
            mean_dist = np.mean(nearest_distances)

            close_pairs = []
            label_keys = list(centroids.keys())

            for i, d in enumerate(nearest_distances):
                # First condition is detecting close pairs, and the second makes sure that they are not coming from two different grid squares.
                if (d < (mean_dist - std_merge_threshold * std_dev)):
                    j = np.argmin(dist_matrix[i])
                    label_i = label_keys[i]
                    label_j = label_keys[j]

                    # Do not merge edge touching regions
                    if (
                        label_i < distinct_label_offset and
                        label_j < distinct_label_offset and
                        label_i != label_j
                    ):
                        close_pairs.append((i, j))

            if not close_pairs:
                if debug: 
                    print("No more close pairs to merge.")
                    print("Mean distance and std deviation:")
                    print(mean_dist, std_dev)
                    print("Nearest distances:")
                    print(nearest_distances)
                    print("Standard deviation threshold:")
                    print(std_merge_threshold)
                break

            label_keys = list(centroids.keys())

            # Merge close pairs
            for i, j in close_pairs:
                label_i = label_keys[i]
                label_j = label_keys[j]
                if label_i == label_j:
                    continue
                mask_i = labels == label_i
                mask_j = labels == label_j
                labels[mask_i | mask_j] = label_j

            # Final centroid calculation
            centroids = {}
            for label_val in np.unique(labels):
                if label_val == 0:
                    continue
                mask = labels == label_val
                centroids[label_val] = (np.mean(X[mask]), np.mean(Y[mask]))
            
            if debug: 
                centroid_list = list(centroids.values())
                plotting_manager.plot_montage_from_image(binary_image, extent, centroid_list, save_path=f"{area_threshold_ratio}_{std_merge_threshold}_{iteration}.png")
                iteration += 1



        # Final centroid calculation
        centroids = {}
        for label_val in np.unique(labels):
            if label_val == 0:
                continue
            mask = labels == label_val
            centroids[label_val] = (np.mean(X[mask]), np.mean(Y[mask]))
        

        return centroids
