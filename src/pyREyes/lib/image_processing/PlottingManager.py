from pathlib import Path
import hyperspy.api as hs
from typing import Any, Dict, List, Optional, Tuple
from pyREyes.lib.REyes_microscope_configurations import MicroscopeConfig, load_microscope_configs
import numpy as np
from pyREyes.lib.REyes_errors import GridSquareError, CentroidDetectionError, FileNotFoundError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.ndimage import rotate, label


class PlottingManager:
    """Handles plotting and visualization operations."""
    
    def __init__(self, output_dir: Path, config: MicroscopeConfig):
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(exist_ok=True)

    
    @staticmethod
    def clip_image(image: np.ndarray) -> np.ndarray:
        """Clip image values to 1st-99th percentile range."""
        try:
            lower_bound = np.percentile(image, 1)
            upper_bound = np.percentile(image, 99)
            return np.clip(image, lower_bound, upper_bound)
        except Exception as e:
            raise GridSquareError(f"Error clipping image: {str(e)}") from e
    
    @staticmethod
    def rotate_with_nan(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image with NaN handling for transparency.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees
            
        Returns:
            Rotated image with NaN values in undefined areas
        """
        try:
            # First rotate without NaN handling
            rotated_image = rotate(image, angle, reshape=True, order=1, mode='constant', cval=0)
            
            # Create and rotate a mask
            mask = np.ones_like(image)
            rotated_mask = rotate(mask, angle, reshape=True, order=0, mode='constant', cval=0)
            
            # Apply mask - set values where mask is 0 to NaN
            result = rotated_image.astype(np.float64)  # Convert to float64 to handle NaN
            result[rotated_mask == 0] = np.nan
            
            return result
            
        except Exception as e:
            raise GridSquareError(f"Error rotating image: {str(e)}") from e
       
    def plot_montage_from_image(self,
                                montage_image: np.ndarray,
                                extent,
                                centroids: List[Tuple[float, float]],
                                save_path = "combined_grid_montage",
                                title: str = "Combined Grid Montage with Found Grid Squares") -> None:
        """Plot a pre-rendered montage image with overlaid centroids."""
        try:
            fig, ax = plt.subplots(figsize=(10, 10))

            ax.imshow(montage_image, extent=extent, origin='upper', cmap='gray')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Plot centroids
            self._plot_centroids(centroids, ax)

            ax.set_title(title)
            ax.set_xlabel(r'X ($\mu$m)')
            ax.set_ylabel(r'Y ($\mu$m)')

            # Save and close
            plt.savefig(
                self.output_dir / save_path,
                transparent=True,
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

        except Exception as e:
            raise GridSquareError(f"Error plotting montage from image: {str(e)}") from e

    def plot_diagnostics_with_lines(self,
                                montage_image: np.ndarray,
                                extent,
                                centroids: List[Tuple[float, float]],
                                lines_to_plot: List[Tuple[Tuple[float, float], Tuple[float, float], float, bool]],
                                save_path="debug_diagnostic.png",
                                title: str = "Grid Merging Diagnostics") -> None:
        """Plot a montage image with overlaid centroids and lines indicating merging operations."""
        try:
            fig, ax = plt.subplots(figsize=(10, 10))

            ax.imshow(montage_image, extent=extent, origin='upper', cmap='gray')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

            # Plot centroids
            self._plot_centroids(centroids, ax)

            # Overlay lines
            for (p1, p2, dist, will_merge) in lines_to_plot:
                color = 'red' if will_merge else 'green'
                line_style = '-' if will_merge else '--'
                linewidth = 2.5 if will_merge else 1.5
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, linestyle=line_style)
                
                # Add distance label near the line middle
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                # Add slight offset to avoid drawing right over the line
                label_weight = 'normal'
                label_alpha = 0.3

                ax.text(mid_x, mid_y, f"{dist:.1f}", color=color, fontsize=10, fontweight=label_weight, 
                        bbox=dict(facecolor='white', alpha=label_alpha, edgecolor='none', pad=1))

            ax.set_title(title)
            ax.set_xlabel(r'X ($\mu$m)')
            ax.set_ylabel(r'Y ($\mu$m)')

            # Save and close
            plt.savefig(
                self.output_dir / save_path,
                dpi=300,
                bbox_inches='tight'
            )
            plt.close(fig)

        except Exception as e:
            raise GridSquareError(f"Error plotting diagnostics with lines: {str(e)}") from e

    
    def plot_return_image(
        self,
        mrc_data: hs.signals.Signal2D,
        rotation_angle: float,
        extent,
        coordinates: np.ndarray,
        dpi: int = 100
    ) -> np.ndarray:
        """Plot montage with overlaid centroids and return it as an image array (HxWx3 RGB)."""
        try:
            nearest_distances = [
                np.linalg.norm(coordinates[i] - coordinates[i-1])
                for i in range(1, len(coordinates))
            ]
            base_image_size = min(nearest_distances)
            image_size = base_image_size * self.config.scaling

            width = np.abs(extent[1] - extent[0])/100
            height = np.abs(extent[3] - extent[2])/100

            fig = plt.figure(figsize=(width, height), dpi=dpi)
            canvas = FigureCanvas(fig)  # Bind canvas for rendering
            fig.set_canvas(canvas)
            ax = fig.add_subplot(111)


            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            ax.set_aspect("equal")
            ax.axis("off")  # ← Hides axes, ticks, labels

            # Plot frames
            for i in range(len(mrc_data)):
                self._plot_frame(
                    mrc_data.inav[i].data,
                    rotation_angle,
                    coordinates[i],
                    ax,
                    image_size
                )

            # Draw the canvas and get the image as array
            fig.tight_layout(pad=0)
            canvas.draw()  # make sure it's rendered
            width, height = fig.canvas.get_width_height()

            rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            rgb = rgba[..., :3]  # Strip alpha
            image = np.dot(rgb.astype(np.float32), [0.2989, 0.5870, 0.1140])

            plt.close(fig)  # Clean up

            return image  # This is your RGB image

        except Exception as e:
            raise GridSquareError(f"Error plotting montage: {str(e)}") from e


    def _plot_frame(self,
                   frame: np.ndarray,
                   rotation_angle: float,
                   position: Tuple[float, float],
                   ax,
                   image_size: float) -> None:
        """Plot a single frame in the montage."""
        frame_clipped = PlottingManager.clip_image(frame)
        frame_rotated = PlottingManager.rotate_with_nan(frame_clipped, rotation_angle)
        
        x, y = position
        extent = [
            x - image_size / 2,
            x + image_size / 2,
            y - image_size / 2,
            y + image_size / 2
        ]
        ax.imshow(frame_rotated, extent=extent, origin='lower', cmap='gray', alpha=1)
    
    def _plot_centroids(self, centroids: List[Tuple[float, float]], ax) -> None:
        """Plot centroids with labels on the given axis."""
        if not centroids:
            return

        x_coords, y_coords = zip(*centroids)

        # Plot point markers
        ax.scatter(x_coords, y_coords, color='blue', marker='o', label='Centroids')

        # Annotate each centroid
        for i, (x, y) in enumerate(zip(x_coords, y_coords), start=2):
            ax.text(
                x - 5, y + 5,
                str(i),
                fontsize=9,
                ha='right',
                va='bottom',
                color='blue',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.25, edgecolor='none')
            )

    
    def plot_image(image: np.ndarray, save_path: str):
        """
        Save a 0-255 grayscale image without displaying it.

        Args:
            image (np.ndarray): Grayscale image (0–255).
            save_path (str): File path to save the image.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray', vmin=0, vmax=255, origin='upper')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_intensity_histogram(
        self,
        image: np.ndarray,
        save_path: str = "intensity_distribution.png",
        title: str = "Pixel Intensity Distribution",
        mu_dark: float = None,
        kde_dark=None,
        mu_bright: float = None,
        kde_bright=None,
        threshold: float = None,
    ):
        """
        Plot a histogram of pixel intensities (ignoring NaNs), with optional
        overlays of the KDE density curves and the binarization threshold.

        Args:
            image:      Input image (raw, un-normalized). NaNs are ignored.
            save_path:  Output filename inside self.output_dir.
            title:      Plot title.
            mu_dark:    Mode (peak) of the dark-half KDE.
            kde_dark:   scipy gaussian_kde object fitted to dark pixels.
            mu_bright:  Mode (peak) of the bright-half KDE.
            kde_bright: scipy gaussian_kde object fitted to bright pixels.
            threshold:  Midpoint between the two peaks used as the cutoff.
        """
        try:
            valid_pixels = image[~np.isnan(image)].ravel()

            # Normalize to [0, 1] so x-axis matches the KDE domain
            pmin, pmax = valid_pixels.min(), valid_pixels.max()
            norm_pixels = (valid_pixels - pmin) / (pmax - pmin) if pmax > pmin else valid_pixels

            fig, ax = plt.subplots(figsize=(9, 6))

            # Histogram (density=True so KDE curves are on the same scale)
            ax.hist(norm_pixels, bins=150, color="#4C9BE8", alpha=0.55,
                    density=True, label="Pixel intensity")

            x_full = np.linspace(0, 1, 1000)

            # Dark KDE curve
            if kde_dark is not None:
                x_dark = np.linspace(0.0, 0.5, 1000)
                y_dark = kde_dark(x_dark)
                ax.plot(x_dark, y_dark, color="#E05C5C", linewidth=2.0,
                        label=f"Dark KDE  peak={mu_dark:.3f}")
            if mu_dark is not None:
                ax.axvline(mu_dark, color="#E05C5C", linewidth=1.4, linestyle=":")

            # Bright KDE curve
            if kde_bright is not None:
                x_bright = np.linspace(0.5, 1.0, 1000)
                y_bright = kde_bright(x_bright)
                ax.plot(x_bright, y_bright, color="#5CB85C", linewidth=2.0,
                        label=f"Bright KDE  peak={mu_bright:.3f}")
            if mu_bright is not None:
                ax.axvline(mu_bright, color="#5CB85C", linewidth=1.4, linestyle=":")

            # Threshold midpoint line
            if threshold is not None:
                ax.axvline(threshold, color="gold", linewidth=2.2, linestyle="--",
                           label=f"Threshold (midpoint) = {threshold:.3f}")

            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Normalised Pixel Intensity [0 – 1]", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=9, framealpha=0.85)
            ax.grid(True, linestyle="--", alpha=0.45)

            fig.tight_layout()
            fig.savefig(self.output_dir / save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            raise GridSquareError(f"Error plotting intensity histogram: {str(e)}") from e
