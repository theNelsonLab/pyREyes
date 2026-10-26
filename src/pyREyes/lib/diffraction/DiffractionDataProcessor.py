
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # files only; never a GUI backend, even with X11 forwarding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyREyes.lib.diffraction import beam_center_search as bcs
from pyREyes.lib.REyes_logging import log_print
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs

MICROSCOPE_CONFIGS = load_microscope_configs()

class DiffractionProcessingError(Exception):
    """Custom exception for diffraction processing errors."""
    pass

class DiffractionDataProcessor:
    def __init__(self, microscope_type: str, camera_length: Optional[float] = None, pixel_size: Optional[float] = None,
                 beam_center: bcs.Mode = bcs.MODE_AUTO):
        """Initialize processor with microscope configuration and optional overrides.
        
        Args:
            microscope_type: Type of microscope from MICROSCOPE_CONFIGS
            camera_length: Optional override for camera length in mm
            pixel_size: Optional override for pixel size in mm/pixel
            beam_center: Where snapshot rings are centred: 'auto' to find the
                direct beam, 'frame' for the middle of the frame, or (x, y)
        """
        if microscope_type not in MICROSCOPE_CONFIGS:
            raise ValueError(f"Unsupported microscope type. Choose from: {list(MICROSCOPE_CONFIGS.keys())}")
        
        self.config = MICROSCOPE_CONFIGS[microscope_type]
        
        # Apply overrides if provided
        if camera_length is not None:
            log_print(f"Overriding camera length: {camera_length} mm")
            self.config.camera_length = camera_length
        
        if pixel_size is not None:
            log_print(f"Overriding pixel size: {pixel_size} mm/pixel")
            self.config.pixel_size = pixel_size

        self.beam_center_mode = beam_center
        self._beam_centres: Dict[Tuple[int, int], bcs.BeamCentre] = {}

    def find_beam_center(self, mrc_files: Iterable[str], output_dir: Optional[str] = None) -> None:
        """Find the direct beam from the frames that are about to be plotted.

        One centre serves every snapshot of a session, because a single frame's
        estimate is not to be relied on. Call this before the plotting loop; a
        snapshot plotted without it has only its own frame to go on.

        Args:
            mrc_files: The MRC files whose snapshots will be plotted
            output_dir: Where to record the centre, beside the snapshots
        """
        from pyREyes.dif_map import load_diffraction_frame

        if self.beam_center_mode == bcs.MODE_AUTO:
            frames = []
            for mrc_file in bcs.sample(list(dict.fromkeys(str(f) for f in mrc_files))):
                try:
                    frames.append(load_diffraction_frame(mrc_file)[0])
                except Exception as e:
                    log_print(f"Beam centre: could not read {mrc_file}: {str(e)}", logging.WARNING)
            self._beam_centres.update(bcs.find_beam_centres(frames))

        if output_dir is not None and self.beam_center_mode != bcs.MODE_FRAME:
            bcs.write_sidecar(output_dir, self.beam_center_mode, self._beam_centres.values())

    def _beam_centre_for(self, frame: np.ndarray) -> bcs.BeamCentre:
        """The centre to draw this frame's rings about."""
        shape = (frame.shape[0], frame.shape[1])
        if self.beam_center_mode == bcs.MODE_AUTO and shape not in self._beam_centres:
            self._beam_centres.update(bcs.find_beam_centres([frame]))
        return bcs.resolve(self.beam_center_mode, shape, self._beam_centres)

    def process_csv(self, file_path: str) -> pd.DataFrame:
        """Process CSV file with error handling and validation."""
        try:
            df = pd.read_csv(file_path)
            required_columns = {'DifQuality', 'FTPeaks', 'FilteredPeaks', 'Sum', 'Coordinates'}
            missing_columns = required_columns - set(df.columns)
            
            if missing_columns:
                raise DiffractionProcessingError(f"Missing required columns: {missing_columns}")

            if df.duplicated(subset=['FilteredPeaks', 'Sum']).any():
                log_print("Warning: Duplicate FilteredPeaks and Sum values found")

            quality_order = {
                'Good diffraction': 0,
                'Bad diffraction': 1,
                'Poor diffraction': 2,
                'No diffraction': 3,
                'Grid': 4
            }
            
            df['QualityRank'] = df['DifQuality'].map(quality_order)
            return df.sort_values(
                by=['QualityRank', 'FTPeaks', 'FilteredPeaks', 'Sum'],
                ascending=[True, False, False, False]
            )

        except Exception as e:
            raise DiffractionProcessingError(f"Error processing CSV file: {str(e)}")

    def generate_nav_entries(self, df: pd.DataFrame, start_item_number: Optional[int]) -> List[str]:
        """Generate .nav file entries from DataFrame."""
        items = []
        nav_template = """
    [Item = {item_number}]
    Color = 0
    StageXYZ = {stage_x} {stage_y} {stage_z}
    NumPts = 1
    Regis = 1
    Type = 0
    RawStageXY = {stage_x} {stage_y}
    MapID = {map_id}
    PtsX = {stage_x}
    PtsY = {stage_y}"""

        if start_item_number is None:
            for row in df.itertuples():
                try:
                    item_number = getattr(row, "CustomItemNumber", None)
                    if item_number is None:
                        raise ValueError("Missing CustomItemNumber for row")

                    stage_x, stage_y, stage_z = eval(row.Coordinates)
                    map_id = int(row.Sum)
                    nav_entry = nav_template.format(
                        item_number=item_number,
                        stage_x=stage_x,
                        stage_y=stage_y,
                        stage_z=stage_z,
                        map_id=map_id
                    )
                    items.append(nav_entry)
                except Exception as e:
                    logging.error(f"Error generating nav entry for row with CustomItemNumber={item_number}: {str(e)}")
                    continue
        else:
            for index, row in enumerate(df.itertuples(), start=start_item_number):
                try:
                    stage_x, stage_y, stage_z = eval(row.Coordinates)
                    map_id = int(row.Sum)
                    nav_entry = nav_template.format(
                        item_number=index,
                        stage_x=stage_x,
                        stage_y=stage_y,
                        stage_z=stage_z,
                        map_id=map_id
                    )
                    items.append(nav_entry)
                except Exception as e:
                    logging.error(f"Error generating nav entry for row {index}: {str(e)}")
                    continue

        return items

    def plot_diffraction_snapshot(
        self,
        mrc_file: str,
        output_path: str,
        index: int,
        spots_value: int,
        sum_value: int
    ) -> None:
        """Plot single diffraction snapshot with resolution rings."""
        try:
            # The loader the diffraction map scored the frame with, so a
            # multi-frame MRC is summed here exactly as it was there
            from pyREyes.dif_map import load_diffraction_frame

            frame, _ = load_diffraction_frame(mrc_file)

            # Calculate pixel size based on image dimensions
            pixel_size = self._calculate_pixel_size(frame.shape)
            
            centre = self._beam_centre_for(frame)
            center_x, center_y = centre.x, centre.y
            max_radius = centre.max_ring_radius()

            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 8))
            vmin, vmax = np.percentile(frame, (1, 99))
            ax.imshow(frame, cmap='Greys', vmin=vmin, vmax=vmax)

            # Add resolution rings
            self._add_resolution_rings(ax, center_x, center_y, max_radius, pixel_size)
            bcs.annotate(ax, centre)

            # Customize plot
            ax.set_axis_off()
            ax.set_title(f"{os.path.basename(mrc_file)} | Spots: {int(spots_value)} | Sum: {int(sum_value)}")
            
            # Save figure
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            log_print(f"Saved: {os.path.basename(output_path)}")

        except Exception as e:
            log_print(f"Error processing diffraction snapshot {mrc_file}: {str(e)}", logging.ERROR)

    def _calculate_pixel_size(self, shape: Tuple[int, int]) -> float:
        """Calculate pixel size based on image dimensions."""
        base_size = self.config.pixel_size
        if shape[0] == 4096:
            return base_size / 2
        elif shape[0] == 8192:
            return base_size / 4
        elif shape[0] == 1024:
            return base_size * 2
        return base_size

    def _add_resolution_rings(
        self,
        ax: plt.Axes,
        center_x: float,
        center_y: float,
        max_radius: float,
        pixel_size: float
    ) -> None:
        """Add resolution rings to the plot."""
        num_rings = 4
        ring_radii = np.linspace(max_radius / (num_rings + 1), max_radius, num_rings)

        for radius in ring_radii:
            if radius <= max_radius:
                circle = plt.Circle(
                    (center_x, center_y),
                    radius,
                    color='dimgrey',
                    fill=False,
                    linestyle='--'
                )
                ax.add_patch(circle)

                resolution = (self.config.wavelength * self.config.camera_length) / (radius * pixel_size)
                label_x = center_x + radius / np.sqrt(2)
                label_y = center_y - radius / np.sqrt(2)
                ax.text(
                    label_x,
                    label_y,
                    f"{resolution:.2f} Å",
                    color='dimgrey',
                    fontsize=10
                )
