import numpy as np

class GridProcessor:
    """Handles grid pattern creation and analysis of bright/dark regions."""

    def __init__(self, pitch=12, grid_bar_width=5, grid_spacing=10):
        """
        Initializes the GridProcessor with grid parameters.

        Args:
            pitch (int): Number of times the grid bars repeat.
            grid_bar_width (int): Width of the black grid bars in pixels.
            grid_spacing (int): Distance between grid bars in pixels.
        """
        self.pitch = pitch
        self.grid_bar_width = grid_bar_width
        self.grid_spacing = grid_spacing
        self.image = self.create_grid_pattern()

    @staticmethod
    def compute_grid_bar_ratio(bright_dark_ratio):
        """
        Computes the ratio between the width of the grid bar and the grid spacing,
        given the ratio of bright to dark area in the image.

        Solves for `x` in:  x(2 + x) = R, where x is the ratio of the grid bar width to spacing.

        Args:
            bright_dark_ratio (float): The ratio of bright area to dark area.

        Returns:
            float: The computed ratio of grid bar width to spacing.
        """
        if bright_dark_ratio <= 0:
            raise ValueError("Ratio must be positive.")

        # Solve for x using the quadratic equation x(2 + x) = R
        a = 1
        b = 2
        c = -bright_dark_ratio * 0.7

        # Quadratic formula: x = (-b + sqrt(b^2 - 4ac)) / 2a
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError("No real solution for the given ratio.")

        x = (-b + np.sqrt(discriminant)) / (2 * a)  # Only take the positive solution
        return x


