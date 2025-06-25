import numpy as np
import time
from shapely.geometry import Polygon, Point
from shapely import vectorized
from scipy.spatial.distance import cdist

# ---------------------------
# Define the CartImage class
class CartImage:
    def __init__(self, img, center, physical_width, physical_height, rotation, scale, is_fixed_grid=False):
        """
        Represents an image with properties in physical (Cartesian) space.
        Instead of using the pixel dimensions of the image to define its physical size,
        the physical width and height are provided by the user.
        
        Args:
            img (np.ndarray): 2D image (grayscale).
            center (tuple): (X, Y) physical coordinates of the image center.
            physical_width (float): Physical width of the image.
            physical_height (float): Physical height of the image.
            rotation (float): Rotation angle in degrees (counterclockwise).
                              (This rotation is applied relative to the canonical orientation.)
        """
        self.img = img
        self.center = np.array(center, dtype=float)

        # Store initial width, height, scale, and rotation
        self.init_width = physical_width
        self.init_height = physical_height
        self.init_rotation = rotation
        self.init_scale = scale

        # Current width, height, scale, and rotation (modified by transformations)
        self.physical_width = physical_width
        self.physical_height = physical_height
        self.rotation = rotation
        self.scale = scale

        self.pixel_height, self.pixel_width = img.shape[:2]

        # Cache for transformed image matrix
        self.transformed_image_matrix = None

        # Flag to determine if this is a fixed reference grid
        self.is_fixed_grid = is_fixed_grid


    def get_shapely_polygon(self):
        """pi
        Returns a Shapely Polygon representing the transformed image.
        """
        return Polygon(self.get_transformed_corners())  # Uses your existing transformation 
    
    # ---------------------------
    # Corners and Extent

    def get_init_corners_origin_centered(self):
        """
        Returns the physical coordinates of the four corners of the image before transformation.
        
        Returns:
            np.ndarray: 4x2 array of physical coordinates for the corners.
        """
        half_width = self.physical_width / 2
        half_height = self.physical_height / 2
        
        # Define the initial corners without rotation/scaling
        corners = np.array([
            [-half_width, -half_height],  # Bottom-left
            [ half_width, -half_height],  # Bottom-right
            [ half_width,  half_height],  # Top-right
            [-half_width,  half_height]   # Top-left
        ])
        
        return corners
    
    def get_init_corners_absolute(self):
        """
        Returns the absolute physical coordinates of the initial corners before transformation.
        This method takes into account the center position.
        
        Returns:
            np.ndarray: 4x2 array of absolute physical coordinates for the corners.
        """
        return self.get_init_corners_absolute() + self.center

    def get_init_top_left_origin_centered(self):
        """
        Returns the physical coordinates of the top-left corner before transformation.
        The image is centeed at (0,0)
        Returns:
            tuple: (x, y) coordinates of the top-left corner.
        """
        return self.get_init_corners_origin_centered()[3]

    def get_transformed_corners(self):
        """
        Returns the physical coordinates of the four corners of the image after transformation.
        
        Returns:
            np.ndarray: 4x2 array of physical coordinates for the transformed corners.
        """
        init_corners = self.get_init_corners_origin_centered()
        transform_matrix = self.get_transform_matrix()
        
        # Apply the transformation (scale and rotate)
        transformed_corners = init_corners.dot(transform_matrix.T)
        
        # Translate to the image center
        return transformed_corners + self.center

    def get_transformed_top_left(self):
        """
        Returns the physical coordinates of the top-left corner after transformation.
        
        Returns:
            tuple: (x, y) coordinates of the top-left corner.
        """
        return self.get_transformed_corners()[3]  # Top-left corner after transformation

    def get_extent(self):
        """
        Returns the axis-aligned physical extent [minX, maxX, minY, maxY] of the image.
        
        Returns:
            list: [minX, maxX, minY, maxY]
        """
        corners = self.get_transformed_corners()
        min_x = corners[:, 0].min()
        max_x = corners[:, 0].max()
        min_y = corners[:, 1].min()
        max_y = corners[:, 1].max()
        return [min_x, max_x, min_y, max_y]


    # ---------------------------
    # Pixel Coordinates
    def get_init_pixel_physical_coord_origin_centered(self):
        """
        Computes the initial physical coordinates (before transformation) of each pixel.
        
        Returns:
            X, Y (each a 2D numpy array): The initial physical coordinates.
        """
        dx = self.init_width / self.pixel_width
        dy = self.init_height / self.pixel_height
        tlx, tly = self.get_init_top_left_origin_centered()
        
        xs = tlx + (np.arange(self.pixel_width) + 0.5) * dx
        ys = tly + (np.arange(self.pixel_height) + 0.5) * (-dy)  # Assuming Y increases downward
        
        return np.meshgrid(xs, ys)


    def get_pixel_physical_coord(self):
        """
        Computes the transformed physical coordinates of each pixel after scale and rotation.
        
        Returns:
            X_trans, Y_trans: 2D numpy arrays representing transformed physical coordinates.
        """
        X, Y = self.get_init_pixel_physical_coord_origin_centered()
        pts = np.vstack([X.ravel(), Y.ravel()])  # Flatten grids
        
        # Apply transformation
        transform_matrix = self.get_transform_matrix()
        pts_trans = transform_matrix.dot(pts) + self.center[:, np.newaxis]
        
        # Reshape to original grid shape
        return pts_trans[0].reshape(X.shape), pts_trans[1].reshape(Y.shape)

    
    # ---------------------------
    # Grid Coordinates
    def get_init_grid_square_center_origin_centered(self, ratio, pitch):
        """
        Computes the initial physical coordinates (before transformation) of each pixel.
        Ratio is the ratio between the bar width and the grid space
        
        Returns:
            X, Y (each a 2D numpy array): The initial physical coordinates.
        """
        grid_space_x = self.init_width / (ratio * pitch + pitch - 1)
        grid_space_y = self.init_height / (ratio * pitch + pitch - 1)
        grid_bar_x = grid_space_x * ratio
        grid_bar_y = grid_space_y * ratio
        dx = grid_space_x + grid_bar_x
        dy = grid_space_y + grid_bar_y
        tlx, tly = self.get_init_top_left_origin_centered()
        
        # Adjust starting position: first point at (grid_bar + 1/2 * grid_space)
        xs = tlx + grid_bar_x + 0.5 * grid_space_x + np.arange(pitch - 1) * dx
        ys = tly - grid_bar_y - 0.5 * grid_space_y + np.arange(pitch - 1) * (-dy)  # Assuming Y increases downward

        return np.meshgrid(xs, ys)

    def get_gird_center_physical_coord(self, ratio, pitch):
        """
        Computes the transformed physical coordinates of each pixel after scale and rotation.
        
        Returns:
            X_trans, Y_trans: 2D numpy arrays representing transformed physical coordinates.
        """
        X, Y = self.get_init_grid_square_center_origin_centered(ratio, pitch)
        pts = np.vstack([X.ravel(), Y.ravel()])  # Flatten grids
        
        # Apply transformation
        transform_matrix = self.get_transform_matrix()
        pts_trans = transform_matrix.dot(pts) + self.center[:, np.newaxis]  # Apply rotation & shift
        
        # Convert transformed points into a list of tuples
        return pts_trans.T 

    def filter_grid_points(self, points):
        """
        Filters a list of coordinates, keeping only those inside the polygon formed by self's transformed corners.

        Args:
            points (np.ndarray): (N, 2) array of (x, y) coordinates.

        Returns:
            np.ndarray: (M, 2) array of points that are inside the polygon.
        """
        # Get transformed corners and form the polygon
        transformed_corners = self.get_transformed_corners()
        polygon = Polygon(transformed_corners)

        # Convert points to Shapely Point objects and check if they are inside the polygon
        filtered_points = np.array([point for point in points if polygon.contains(Point(point))])

        return filtered_points


    # ---------------------------
    # Transformations
    def get_transform_matrix(self):
        """
        Computes the 2x2 transformation matrix based on the rotation angle and
        the transform scaling factor.
        
        Returns:
            numpy.ndarray: A 2x2 matrix defined as:
                scale * [[cos(theta), -sin(theta)],
                                   [sin(theta),  cos(theta)]]
            where theta is the rotation angle in radians.
        """
        theta = np.deg2rad(self.rotation)
        return self.scale * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

    def apply_transform(self, new_scale, new_rotation):
        """
        Updates the current width, height, scale, and rotation based on the initial values.

        Args:
            new_scale (float): The new scale factor to be applied.
            new_rotation (float): The new rotation angle in degrees.
        """
        # Update scale and rotation
        self.scale = self.init_scale * new_scale
        self.rotation = (self.init_rotation + new_rotation) % 360  # Normalize rotation

        # Update the width and height according to the new scale
        self.physical_width = self.init_width * self.scale
        self.physical_height = self.init_height * self.scale


    def get_transformed_image_matrix(self):
        """
        Returns a 3D matrix where:
        - The first two dimensions give the transformed physical coordinates of the pixels.
        - The third dimension stores the image intensities.

        Returns:
            np.ndarray: A (H, W, 3) matrix where:
                - [:, :, 0] contains X physical coordinates.
                - [:, :, 1] contains Y physical coordinates.
                - [:, :, 2] contains pixel intensity values.
        """
        if self.is_fixed_grid and self.transformed_image_matrix is not None:
            return self.transformed_image_matrix
        else:
            # Get transformed coordinates of each pixel
            X_trans, Y_trans = self.get_pixel_physical_coord()
            
            # Reshape the image to match the (H, W, 1) shape for stacking
            img_values = self.img.reshape(self.pixel_height, self.pixel_width, 1)

            # Stack into a 3D matrix: (H, W, 3)
            transformed_matrix = np.dstack((X_trans, Y_trans, img_values))

            return transformed_matrix

    def merge_small_regions_centroids(self, area_threshold_ratio=0.3, distinct_label_offset: int = 1000):
        # Works when the img is already labeled 
        transformed_matrix = self.get_transformed_image_matrix()

        X = transformed_matrix[:, :, 0].ravel()
        Y = transformed_matrix[:, :, 1].ravel()
        labels = transformed_matrix[:, :, 2].ravel().astype(int)

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

            large_labels = [lbl for lbl in unique_labels if lbl not in small_labels]
            if not large_labels:
                break

            small_centroids = np.array([centroids[lbl] for lbl in small_labels])
            large_centroids = np.array([centroids[lbl] for lbl in large_labels])
            distances = cdist(small_centroids, large_centroids)
            closest_indices = np.argmin(distances, axis=1)

            next_label = labels.max() + 1
            for i, small_label in enumerate(small_labels):
                large_label = large_labels[closest_indices[i]]
                mask_small = labels == small_label
                mask_large = labels == large_label
                labels[mask_small | mask_large] = next_label
                next_label += 1


        centroid_coords = np.array(list(centroids.values()))
        if len(centroid_coords) < 2:
            return {}  # or handle however you'd like

        # Compute initial distance matrix and stats just once
        dist_matrix = cdist(centroid_coords, centroid_coords)
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_distances = np.min(dist_matrix, axis=1)
        
        std_dev = np.std(nearest_distances)

        # Start merging loop
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
                if d < (mean_dist - 2 * std_dev):
                    j = np.argmin(dist_matrix[i])
                    label_i = label_keys[i]
                    label_j = label_keys[j]

                    # Only add valid, mergeable pairs
                    if (
                        label_i < distinct_label_offset and
                        label_j < distinct_label_offset and
                        label_i != label_j
                    ):
                        close_pairs.append((i, j))

            if not close_pairs:
                break

            label_keys = list(centroids.keys())
            next_label = labels.max() + 1

            for i, j in close_pairs:
                label_i = label_keys[i]
                label_j = label_keys[j]
                if label_i == label_j:
                    continue
                mask_i = labels == label_i
                mask_j = labels == label_j
                labels[mask_i | mask_j] = next_label
                next_label += 1


        # Final centroid calculation
        centroids = {}
        for label_val in np.unique(labels):
            if label_val == 0:
                continue
            mask = labels == label_val
            centroids[label_val] = (np.mean(X[mask]), np.mean(Y[mask]))

        return centroids

        
    def get_image_in_polygon(self, polygon):
        """
        Masks the transformed image such that pixels outside the given polygon have zero intensity.

        Args:
            polygon (Polygon): A Shapely polygon defining the region of interest.

        Returns:
            tuple: 
                - np.ndarray: (H, W, 3) masked image matrix.
                - int: Number of pixels inside the polygon.
        """
        transformed_matrix = self.get_transformed_image_matrix()

        X_coords = transformed_matrix[:, :, 0]
        Y_coords = transformed_matrix[:, :, 1]
        intensity_values = transformed_matrix[:, :, 2]

        # Use vectorized Shapely for efficiency
        mask = vectorized.contains(polygon, X_coords, Y_coords)

        num_pixels_inside = np.count_nonzero(mask)
        masked_intensity = np.where(mask, intensity_values, 0)

        return np.dstack((X_coords, Y_coords, masked_intensity)), num_pixels_inside

    # ---------------------------
    # Interpolation

    def interpolate_onto_grid_fast(self, polygon, grid_resolution=100, method='nearest'):
        """
        Interpolates the nearest neighbor pixel values from self onto a uniform grid inside the given polygon.

        Args:
            polygon (Polygon): The region where interpolation should occur.
            grid_resolution (int): Number of grid points along each axis.
            method (str): Interpolation method ('nearest', 'linear', 'cubic').

        Returns:
            tuple: (grid_x, grid_y, interpolated_values)
                - grid_x: 2D NumPy array of X coordinates for the grid.
                - grid_y: 2D NumPy array of Y coordinates for the grid.
                - interpolated_values: 2D NumPy array of interpolated intensity values.
        """
        # Start timing
        start_time = time.time()

        # Step 1: Get transformed image matrix
        src_matrix, _ = self.get_image_in_polygon(polygon)  # Masked pixels outside polygon = 0

        # Step 2: Extract pixel coordinates and intensities
        X_src, Y_src, Intensity_src = src_matrix[:, :, 0], src_matrix[:, :, 1], src_matrix[:, :, 2]

        # Flatten source image data
        src_points = np.column_stack((X_src.ravel(), Y_src.ravel()))
        src_values = Intensity_src.ravel()

        # Step 3: Define a uniform grid inside the polygon
        min_x, min_y, max_x, max_y = polygon.bounds  # Get bounding box of polygon
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, grid_resolution),
            np.linspace(min_y, max_y, grid_resolution)
        )

        # Step 4: Filter grid points that are inside the polygon
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        inside_mask = vectorized.contains(polygon, grid_points[:, 0], grid_points[:, 1])

        # Apply mask to remove points outside the polygon
        grid_points_filtered = grid_points[inside_mask]

        # If no valid grid points, return empty arrays
        if len(grid_points_filtered) == 0:
            return None, None, None

        # Step 5: Use `griddata` for fast interpolation instead of KDTree
        interpolated_values = griddata(
            src_points, src_values, grid_points_filtered, method=method, fill_value=np.nan
        )

        # Step 6: Reshape interpolated values into the grid shape
        interpolated_grid = np.full(grid_x.shape, np.nan)  # Initialize grid with NaNs
        interpolated_grid.ravel()[inside_mask] = interpolated_values  # Assign values

        # End timing
        end_time = time.time()
        print(f"Interpolation onto grid took {end_time - start_time:.6f} seconds")

        return grid_x, grid_y, interpolated_grid
    


    def interpolate_onto(self, image2, polygon):
        """
        Interpolates the nearest neighbor pixel values from self (source image) to image2 (target image) 
        inside a given polygon.

        Args:
            image2 (CartImage): The target image to interpolate onto.
            polygon (Polygon): The region where interpolation should occur.

        Returns:
            np.ndarray: The updated image2 pixel array with interpolated values.
        """
        # Get transformed image matrices for source and target
        src_matrix, _ = self.get_image_in_polygon(polygon)  # Masked pixels outside polygon = 0
        tgt_matrix = image2.get_transformed_image_matrix()  # Full transformed target image

        # Extract pixel coordinates and intensities
        X_src, Y_src, Intensity_src = src_matrix[:, :, 0], src_matrix[:, :, 1], src_matrix[:, :, 2]
        X_tgt, Y_tgt, _ = tgt_matrix[:, :, 0], tgt_matrix[:, :, 1], tgt_matrix[:, :, 2]

        # Flatten source image data
        src_points = np.column_stack((X_src.ravel(), Y_src.ravel()))
        src_values = Intensity_src.ravel()

        # Filter target points: Only keep those inside the polygon
        tgt_points = np.column_stack((X_tgt.ravel(), Y_tgt.ravel()))
        inside_mask = vectorized.contains(polygon, tgt_points[:, 0], tgt_points[:, 1])

        # Apply mask to get only valid target pixels
        tgt_points_filtered = tgt_points[inside_mask]

        # Ensure we don't proceed if there are no valid target pixels
        if len(tgt_points_filtered) == 0:
            return image2.img  # Return original image if nothing to interpolate

        # Build KDTree for fast nearest-neighbor lookup
        tree = cKDTree(src_points)

        # Find nearest neighbors for only the valid target pixels
        _, indices = tree.query(tgt_points_filtered)

        # Get interpolated intensity values
        interpolated_values = src_values[indices]

        # Create a copy of the target image to update
        updated_img = image2.img.copy()

        # Update only the pixels inside the polygon
        updated_img.ravel()[inside_mask] = interpolated_values

        return updated_img


    def get_cross_correlation(self, image2):
        """
        Computes the cross-correlation between the interpolated image from self onto image2.

        Steps:
        1. Compute the intersection polygon between self and image2.
        2. Interpolate self onto image2 within this polygon.
        3. Use the existing compute_cross_correlation function.

        Args:
            image2 (CartImage): The second image to compare.

        Returns:
            float: The cross-correlation coefficient, or None if no valid intersection.
        """
        # Step 1: Get the intersection polygon
        intersection_polygon, _ = self.get_intersection_polygon_and_area(image2)

        if intersection_polygon is None:
            return None  # No valid intersection

        # Step 2: Interpolate self onto image2 within the intersection polygon
        interpolated_img = image2.interpolate_onto(self, intersection_polygon)
        # _, _, _ = image2.interpolate_onto_grid_fast(intersection_polygon)

        
        # Step 3: Extract only the valid pixels inside the polygon from both images
        transformed_matrix1, _ = self.get_image_in_polygon(intersection_polygon)
        valid_mask = transformed_matrix1[:, :, 2] > 0  # Only pixels inside the polygon

        # Ensure we have enough valid pixels to compute correlation
        if np.sum(valid_mask) == 0:
            return None

        # Keep the 2D structure by masking instead of flattening
        img1_valid = np.where(valid_mask, self.img, 0)
        img2_valid = np.where(valid_mask, interpolated_img, 0)

        # Step  4: Compute NCC
        NCC = compute_similarity(img1_valid, img2_valid)
        print(NCC)
        return NCC


    # ---------------------------
    # Intersection with other Images

    def find_intersection_points(self, other_image):
        """
        Computes all intersection points between this image and another image.

        Args:
            other_image (CartImage): The other image.

        Returns:
            list: A list of (x, y) intersection points.
        """
        poly1 = self.get_shapely_polygon()
        poly2 = other_image.get_shapely_polygon()
        
        intersections = poly1.intersection(poly2)

        if intersections.is_empty:
            return []

        if isinstance(intersections, Point):
            return [(intersections.x, intersections.y)]

        if isinstance(intersections, Polygon):
            return list(intersections.exterior.coords)

        return [(pt.x, pt.y) for pt in intersections]
    
    def get_intersection_polygon_and_area(self, other_image):
        """
        Computes the intersection polygon and its area between this image and another image.

        Args:
            other_image (CartImage): The other image.

        Returns:
            tuple: (Polygon, float) representing the intersection polygon and its area.
                Returns (None, 0) if the intersection is empty, a point, or a line.
        """
        poly1 = self.get_shapely_polygon()
        poly2 = other_image.get_shapely_polygon()

        intersection = poly1.intersection(poly2)

        # Check if the intersection is valid (not empty, not a point, and not a line)
        if intersection.is_empty or isinstance(intersection, Point) or intersection.geom_type == "LineString":
            return None, 0  # Invalid intersection

        return intersection, intersection.area
