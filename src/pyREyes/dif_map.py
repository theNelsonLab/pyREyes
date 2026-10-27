"""
REyes DMP (Diffraction Map Processor)
A tool to process blocks of diffraction images and classify their quality
based on diffraction peaks and LQP (Lattice Quality Peaks) using 
DQI (Diffraction Quality Index) analysis.
"""

import argparse
import logging
import os
import re
import sys
import uuid
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hyperspy.api as hs
import matplotlib
matplotlib.use('Agg')  # files only; never a GUI backend, even with X11 forwarding
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import label, regionprops_table

from pyREyes.lib.ui.REyes_ui import print_banner
from pyREyes.lib.REyes_logging import setup_logging, log_print
from pyREyes.lib.REyes_microscope_configurations import load_microscope_configs, MicroscopeConfig
# The MRC mode 0 signedness rule (see lib/mrc_io.py); IMOD_STAMP,
# IMOD_FLAG_SIGNED_BYTES and mrc_mode0_signed stay importable here.
from pyREyes.lib.mrc_io import (IMOD_FLAG_SIGNED_BYTES, IMOD_STAMP,  # noqa: F401
                                mode0_signed as mrc_mode0_signed, mode0_view)

__version__ = '3.6.1'
__min_required_version__ = '3.4.0'

# Configuration Constants
MICROSCOPE_CONFIGS = load_microscope_configs()

# Detection parameters in the microscope configs are tuned for 2048x2048 frames.
# Larger frames are binned down to this size before detection.
TARGET_FRAME_SIZE = 2048

# How far past a frame's file line its Stage line is looked for when it is not
# at the configured offset (serialEM_log). The search stops at the next frame.
STAGE_LINE_LOOKAHEAD = 12

# Detection parameters for frames smaller than TARGET_FRAME_SIZE, keyed by frame
# size. A frame uses the entry of the largest key not exceeding its size; frames
# smaller than every key use the smallest entry.
SMALL_FRAME_PARAMS = {
    1024: {'light_sigma_prim': 1.0, 'harsh_sigma_prim': 3,
           'threshold_std_prim': 6, 'min_pixels_prim': 3},
    512: {'light_sigma_prim': 1.0, 'harsh_sigma_prim': 3,
          'threshold_std_prim': 6, 'min_pixels_prim': 3},
}

# Bad line (hot/dead column or row) removal, applied only to frames larger than
# TARGET_FRAME_SIZE before binning. Each line is split into segments; a line is
# bad when enough segments deviate from the neighbouring lines in the same
# direction. Real diffraction spots only raise one or two segments of a line.
BAD_LINE_SEGMENTS = 16          # segments per line
BAD_LINE_WINDOW = 15            # neighbouring lines used as local reference
BAD_LINE_Z = 5.0                # per-segment deviation threshold (robust sigma)
BAD_LINE_MIN_SEGMENTS = 6       # deviating segments needed to flag a line
BAD_LINE_MAX_WIDTH = 3          # wider runs are image structure, not defects
BAD_LINE_MAX_FRACTION = 0.01    # more flagged lines than this -> do not clean
BORDER_FRACTION = 0.03          # border blanked during detection


def _border_width(length: int) -> int:
    """Width of the blanked border for an image dimension."""
    return max(1, int(length * BORDER_FRACTION))


def _read_mrc(mrc_file: str) -> np.ndarray:
    """
    Read MRC data with HyperSpy, falling back to mrcfile if HyperSpy fails.

    Mode 0 (8-bit) data are returned with the signedness the header gives
    (see mrc_mode0_signed), whatever the reader chose, so a frame reads the same
    on every machine.
    """
    try:
        data = np.asarray(hs.load(mrc_file).data)
    except Exception as hs_error:
        try:
            import mrcfile
        except ImportError:
            raise hs_error
        log_print(f"HyperSpy could not read {Path(mrc_file).name} ({hs_error}); "
                  f"retrying with mrcfile", logging.WARNING)
        with mrcfile.open(mrc_file, mode='r', permissive=True) as mrc:
            if mrc.data is None:
                raise ValueError(f"No image data in {mrc_file}")
            data = np.array(mrc.data)
    return mode0_view(data, mrc_file)


def _frame_axis(shape: Tuple[int, int, int]) -> int:
    """Axis holding the frames of a 3D stack: the one not matching the image axes."""
    if shape[1] == shape[2] or shape[0] not in (shape[1], shape[2]):
        return 0
    return 2 if shape[0] == shape[1] else 1


def _line_runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Group sorted indices into (start, end) runs of consecutive values."""
    runs = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], int(i))
        else:
            runs.append((int(i), int(i)))
    return runs


def find_bad_lines(data: np.ndarray, axis: int) -> List[Tuple[int, int]]:
    """
    Find narrow bad lines in a 2D frame.

    Args:
        data: 2D frame
        axis: 0 for rows, 1 for columns

    Returns:
        List of (start, end) runs of bad lines, excluding the blanked border
    """
    lines = data if axis == 0 else data.T
    n_lines, length = lines.shape
    usable = (length // BAD_LINE_SEGMENTS) * BAD_LINE_SEGMENTS
    if n_lines < BAD_LINE_WINDOW or usable == 0:
        return []

    segments = lines[:, :usable].reshape(n_lines, BAD_LINE_SEGMENTS, -1).mean(axis=2, dtype=np.float64)
    residual = segments - median_filter(segments, size=(BAD_LINE_WINDOW, 1), mode='nearest')
    sigma = np.median(np.abs(residual - np.median(residual, axis=0)), axis=0) * 1.4826
    sigma = np.where(sigma > 0, sigma, np.inf)
    z = residual / sigma

    bad = ((z > BAD_LINE_Z).sum(axis=1) >= BAD_LINE_MIN_SEGMENTS) | \
          ((z < -BAD_LINE_Z).sum(axis=1) >= BAD_LINE_MIN_SEGMENTS)
    border = _border_width(n_lines)
    bad[:border] = False
    bad[-border:] = False

    return [run for run in _line_runs(np.flatnonzero(bad))
            if run[1] - run[0] < BAD_LINE_MAX_WIDTH]


def remove_bad_lines(data: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
    """
    Replace narrow bad columns and rows with the mean of their neighbours.

    Args:
        data: 2D float frame (modified in place)

    Returns:
        The cleaned frame and the runs replaced per axis ('rows', 'columns')
    """
    replaced = {}
    for axis, name in ((1, 'columns'), (0, 'rows')):
        runs = find_bad_lines(data, axis)
        n_bad = sum(end - start + 1 for start, end in runs)
        if n_bad > BAD_LINE_MAX_FRACTION * data.shape[1 - axis]:
            log_print(f"Warning: {n_bad} bad {name} detected, more than "
                      f"{BAD_LINE_MAX_FRACTION:.0%} of the frame; not cleaning {name}",
                      logging.WARNING)
            continue
        for start, end in runs:
            before = max(start - 1, 0)
            after = min(end + 1, data.shape[1 - axis] - 1)
            if axis == 1:
                data[:, start:end + 1] = ((data[:, before] + data[:, after]) / 2)[:, None]
            else:
                data[start:end + 1, :] = ((data[before, :] + data[after, :]) / 2)[None, :]
        if runs:
            replaced[name] = runs
    return data, replaced


def bin_to_target(data: np.ndarray, target: int = TARGET_FRAME_SIZE) -> Tuple[np.ndarray, int]:
    """
    Mean-bin a 2D frame by one integer factor so that it fits within target.

    The frame is cropped symmetrically to a multiple of the factor, keeping the
    beam centre in place.

    Returns:
        The binned frame and the binning factor
    """
    factor = int(np.ceil(max(data.shape) / target))
    if factor <= 1:
        return data, 1
    height = (data.shape[0] // factor) * factor
    width = (data.shape[1] // factor) * factor
    top = (data.shape[0] - height) // 2
    left = (data.shape[1] - width) // 2
    data = data[top:top + height, left:left + width]
    binned = data.reshape(height // factor, factor, width // factor, factor).mean(
        axis=(1, 3), dtype=np.float64)
    return binned.astype(data.dtype, copy=False), factor


def load_diffraction_frame(mrc_file: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an MRC file as one 2D frame ready for diffraction analysis.

    - Stacks of frames are summed
    - MRC mode 0 bytes are read as the header says (unsigned counts unless the
      IMOD signed-bytes flag is set), whatever the installed reader decides
    - int16 frames up to TARGET_FRAME_SIZE are returned unchanged; everything
      else is converted to float32 (exact for 8- and 16-bit counts)
    - NaN and inf values are replaced with 0
    - Frames larger than TARGET_FRAME_SIZE have bad lines removed and are
      mean-binned by one integer factor

    Args:
        mrc_file: Path to the MRC file

    Returns:
        The frame and a dict describing what was done (shape, dtype, frames,
        binning, non_finite, bad_lines)
    """
    raw = np.squeeze(_read_mrc(mrc_file))
    info = {'shape': raw.shape, 'dtype': str(raw.dtype), 'frames': 1,
            'binning': 1, 'non_finite': 0, 'bad_lines': {}}

    if raw.dtype.kind not in 'uif':
        raise ValueError(f"Unsupported MRC data type {raw.dtype}")

    if raw.ndim == 3:
        axis = _frame_axis(raw.shape)
        info['frames'] = raw.shape[axis]
        # Accumulate in float64 so long stacks of bright frames stay exact
        data = np.moveaxis(raw, axis, 0).sum(axis=0, dtype=np.float64)
    elif raw.ndim == 2:
        data = raw
    else:
        raise ValueError(f"Unsupported MRC shape {raw.shape}")

    oversized = max(data.shape) > TARGET_FRAME_SIZE
    if not (data.dtype == np.int16 and not oversized):
        data = data.astype(np.float32)

    if data.dtype.kind == 'f':
        non_finite = ~np.isfinite(data)
        info['non_finite'] = int(non_finite.sum())
        if info['non_finite']:
            data[non_finite] = 0

    if oversized:
        data, info['bad_lines'] = remove_bad_lines(data)
        data, info['binning'] = bin_to_target(data)

    return data, info


def block_logs(folder_name: str, directory: Optional[Path] = None) -> List[Path]:
    """
    The SerialEM logs of a block folder, newest first.

    `<...><folder>.log`, and the logs SerialEM opens when the square is mapped
    again after a resume (`<...><folder>-1.log`, `-2`, ...). The newest log
    holds the positions of the frames written last.
    """
    directory = Path.cwd() if directory is None else Path(directory)
    pattern = re.compile(rf"{re.escape(folder_name)}(?:-(\d+))?\.log$")
    found = []
    for path in directory.glob(f"*{folder_name}*.log"):
        match = pattern.search(path.name)
        if match and path.is_file():
            found.append((int(match.group(1) or 0), path.name, path))
    return [path for _, _, path in sorted(found, reverse=True)]


def frame_sum(data: np.ndarray):
    """
    Total counts of a frame, the same on every machine.

    Integer frames are summed exactly in int64 (an int, as before); float frames
    are accumulated in float64 (a float). A float32 accumulator rounds with the
    CPU's SIMD summation order.
    """
    if data.dtype.kind in 'iu':
        return int(np.sum(data, dtype=np.int64))
    return float(np.sum(data, dtype=np.float64))


class ImageProcessor:
    """Handles core image processing operations for diffraction pattern analysis."""
    
    def __init__(self, config: MicroscopeConfig, legacy_parse: bool = False):
        """
        Initialize ImageProcessor with microscope configuration.
        
        Args:
            config: MicroscopeConfig object containing processing parameters
            legacy_parse: Load frames exactly as REyes 3.4.0 and earlier did,
                with configuration detection parameters for every frame size
        """
        self.config = config
        self.legacy_parse = legacy_parse
        self._conversion_logged = False  # Track if uint16 conversion message was already logged
        self._logged_messages = set()

    def reset_block_log(self) -> None:
        """Allow once-per-block messages to be logged again for a new block."""
        self._logged_messages.clear()

    def _log_once(self, message: str, level: int = logging.INFO) -> None:
        """Log a message once per block."""
        if message not in self._logged_messages:
            self._logged_messages.add(message)
            log_print(message, level)

    def parse_mrc(self, mrc_file: str) -> np.ndarray:
        """
        Loads an MRC file as a single 2D frame for diffraction analysis.
        
        Args:
            mrc_file: Path to the MRC file
            
        Returns:
            Processed image data as numpy array

        Notes:
            See load_diffraction_frame. With legacy_parse the loading of
            REyes 3.4.0 and earlier is used instead.
        """
        if self.legacy_parse:
            return self._parse_mrc_legacy(mrc_file)

        data, info = load_diffraction_frame(mrc_file)
        name = Path(mrc_file).name

        description = f"{info['shape']} {info['dtype']}"
        if info['frames'] > 1:
            description += f", {info['frames']} frames summed"
        if info['binning'] > 1:
            description += f", binned x{info['binning']}"
        self._log_once(f"Frames loaded as {description} -> {data.shape}")

        if info['non_finite']:
            log_print(f"Warning: {name} contains {info['non_finite']} NaN/inf pixels, set to 0",
                      logging.WARNING)
        for axis_name, runs in info['bad_lines'].items():
            log_print(f"Removed bad {axis_name} {runs} (unbinned coordinates) in {name}")
        if max(data.shape) < TARGET_FRAME_SIZE:
            params = self.detection_params(data.shape)
            self._log_once(f"Frame smaller than {TARGET_FRAME_SIZE}x{TARGET_FRAME_SIZE}; "
                           f"using small-frame detection parameters {params}")
        return data

    def _parse_mrc_legacy(self, mrc_file: str) -> np.ndarray:
        """Loads an MRC file exactly as REyes 3.4.0 and earlier did."""
        # Load the MRC file
        signal = hs.load(mrc_file)
        data = signal.data
        
        # Convert unsigned integer data to float64 early: uint16 MRC data would
        # otherwise overflow in later arithmetic
        if data.dtype in [np.uint8, np.uint16, np.uint32]:
            if not self._conversion_logged:
                log_print(f"Converting {data.dtype} data to float64 to prevent overflow (this message will only appear once)")
                self._conversion_logged = True
            data = data.astype(np.float64)
        
        # Check the shape of the data
        if data.shape == (2048, 2048):
            return data
        elif data.shape[0] < 2048 and data.shape[1] < 2048:
            log_print(f"Warning: Image size is smaller than 2048x2048 ({data.shape}). Proceeding with processing.")
            return data
        else:
            # Image is larger than 2048x2048, downsample stepwise by 2 using mean binning
            while data.shape[0] > 2048 or data.shape[1] > 2048:
                # Ensure dimensions are even for proper binning
                if data.shape[0] % 2 != 0 or data.shape[1] % 2 != 0:
                    data = data[:(data.shape[0]//2)*2, :(data.shape[1]//2)*2]
                
                # Apply mean binning
                data = (data[0::2, 0::2] + data[1::2, 0::2] + 
                       data[0::2, 1::2] + data[1::2, 1::2]) / 4
                log_print(f"Downsampling image using mean binning. New size: {data.shape}")

            if data.shape != (2048, 2048):
                log_print(f"Warning: Image size is not exactly 2048x2048 after downsampling ({data.shape})")
            
            return data

    def detection_params(self, shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Primary detection parameters for a frame of the given shape.

        Frames of TARGET_FRAME_SIZE use the microscope configuration; smaller
        frames use SMALL_FRAME_PARAMS (unless legacy_parse is set).
        """
        size = max(shape)
        if self.legacy_parse or size >= TARGET_FRAME_SIZE:
            return {
                'light_sigma_prim': self.config.light_sigma_prim,
                'harsh_sigma_prim': self.config.harsh_sigma_prim,
                'threshold_std_prim': self.config.threshold_std_prim,
                'min_pixels_prim': self.config.min_pixels_prim,
            }
        fitting = [key for key in SMALL_FRAME_PARAMS if key <= size]
        key = max(fitting) if fitting else min(SMALL_FRAME_PARAMS)
        return SMALL_FRAME_PARAMS[key]

    def create_binary_image(self, 
                        data: np.ndarray, 
                        ft_done: bool = False) -> np.ndarray:
        """Generates a binary image using Gaussian blurs and thresholding."""
        # Everything in float64: integer frames would overflow when the blurred
        # images are subtracted, and float32 filtering, mean and std round
        # differently with the CPU's SIMD width, which moved the threshold and
        # changed spot counts between machines on identical frames.
        # legacy_parse keeps the REyes 3.4.10 and earlier arithmetic (only
        # unsigned integers were converted).
        if not self.legacy_parse or data.dtype in [np.uint8, np.uint16, np.uint32]:
            data = np.asarray(data, dtype=np.float64)
        
        if ft_done:
            # Use original hardcoded values for FT processing
            harsh_blur_data = gaussian_filter(data, sigma=20)
            difference_data = data - harsh_blur_data
            threshold_value = np.mean(difference_data) + (3 * np.std(difference_data))
        else:
            params = self.detection_params(data.shape)
            light_blur_data = gaussian_filter(data, sigma=params['light_sigma_prim'])
            harsh_blur_data = gaussian_filter(data, sigma=params['harsh_sigma_prim'])
            difference_data = light_blur_data - harsh_blur_data
            threshold_value = np.mean(difference_data) + (params['threshold_std_prim'] * np.std(difference_data))

        binary_image = difference_data > threshold_value

        # Mask 3% on all four sides
        mask_size_x = _border_width(binary_image.shape[1])
        mask_size_y = _border_width(binary_image.shape[0])
        
        binary_image[:, :mask_size_x] = False
        binary_image[:, -mask_size_x:] = False
        binary_image[:mask_size_y, :] = False
        binary_image[-mask_size_y:, :] = False
        
        return binary_image

    def get_binary_ft(self, data: np.ndarray) -> np.ndarray:
        """
        Computes and processes the 2D Fourier Transform of input data.
        
        Args:
            data: Input image data
            
        Returns:
            Logarithmically scaled Fourier Transform magnitude
        """
        ft_data = np.fft.fftshift(np.fft.fft2(data))
        ft_magnitude = np.log(np.abs(ft_data) + 1)
        return ft_magnitude

    def get_centroids(self, 
                    binary_image: np.ndarray, 
                    exclude_center: Optional[float] = None, 
                    ft_done: bool = False) -> Tuple[List[Tuple[float, float]], int]:
        """Identifies and filters centroids in a binary image."""
        labeled_array = label(binary_image)
        
        # Use min_pixels=1 for FT patterns, detection parameters otherwise
        min_pixels = 1 if ft_done else self.detection_params(binary_image.shape)['min_pixels_prim']
        
        properties = regionprops_table(labeled_array, properties=('centroid', 'area'))
        centroids = [(x, y) for x, y, area in 
                    zip(properties['centroid-0'], 
                        properties['centroid-1'], 
                        properties['area']) 
                    if area >= min_pixels]
        
        if not ft_done and exclude_center is not None:
            image_shape = binary_image.shape
            center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
            min_distance = min(center_x, center_y)
            exclude_radius = (exclude_center / 100) * min_distance
            
            filtered_centroids = [
                centroid for centroid in centroids
                if not (center_x - exclude_radius < centroid[1] < center_x + exclude_radius and
                    center_y - exclude_radius < centroid[0] < center_y + exclude_radius)
            ]
        else:
            filtered_centroids = centroids
        
        return filtered_centroids, len(filtered_centroids)

@dataclass
class DiffractionResult:
    """Container for diffraction analysis results."""
    n_diffraction_spots: int
    n_pattern_spots: int
    total_sum: float
    quality: str
    
class DiffractionQuality(Enum):
    """Enumeration of possible diffraction quality classifications."""
    NO_DIFFRACTION = "No diffraction"
    POOR_DIFFRACTION = "Poor diffraction"
    BAD_DIFFRACTION = "Bad diffraction"
    GOOD_DIFFRACTION = "Good diffraction"
    GRID = "Grid"

class DiffractionAnalyzer:
    """Analyzes diffraction patterns and classifies their quality."""
    
    def __init__(self, config: MicroscopeConfig, image_processor: ImageProcessor):
        """
        Initialize DiffractionAnalyzer with configuration and image processor.
        
        Args:
            config: MicroscopeConfig object containing analysis parameters
            image_processor: ImageProcessor instance for image processing operations
        """
        self.config = config
        self.image_processor = image_processor

    def process_mrc_file(self, file_path: str) -> Optional[DiffractionResult]:
        """
        Process an MRC file to analyze diffraction pattern quality.
        
        Args:
            file_path: Path to the MRC file
            
        Returns:
            DiffractionResult object containing analysis results or None if processing fails
            
        Notes:
            Processes the file through multiple stages:
            1. Primary binary image creation for diffraction spot detection
            2. Secondary FT analysis for LQP (Lattice Quality Peaks) detection
            3. Quality classification based on DQI (Diffraction Quality Index) calculation
        """
        try:
            # Load and process the MRC file
            data = self.image_processor.parse_mrc(file_path)
            total_sum = frame_sum(data)
            
            # Create primary binary image and detect diffraction spots
            primary_binary_image = self.image_processor.create_binary_image(
                data, 
                ft_done=False
            )
            _, n_dif_spots = self.image_processor.get_centroids(
                primary_binary_image,
                exclude_center=25,
                ft_done=False
            )
            
            # Process FT pattern
            ft_binary_image = self.image_processor.get_binary_ft(primary_binary_image)
            secondary_binary_image = self.image_processor.create_binary_image(
                ft_binary_image,
                ft_done=True
            )
            _, n_pat_spots = self.image_processor.get_centroids(
                secondary_binary_image,
                exclude_center=None,
                ft_done=True
            )
            
            # Classify diffraction quality using DQI (Diffraction Quality Index)
            quality = self._classify_diffraction(n_dif_spots, n_pat_spots)
            
            return DiffractionResult(
                n_diffraction_spots=n_dif_spots,
                n_pattern_spots=n_pat_spots,
                total_sum=total_sum,
                quality=quality.value  # Convert enum to string value
            )
        except Exception as e:
            log_print(f"Error processing MRC file {file_path}: {str(e)}", logging.ERROR)
            return None

    def _classify_diffraction(
        self,
        n_dif_spots: int,
        n_pat_spots: int
    ) -> DiffractionQuality:
        """
        Classify diffraction quality based on spot counts and DQI (Diffraction Quality Index).
        
        Args:
            n_dif_spots: Number of detected diffraction spots
            n_pat_spots: Number of detected LQP (Lattice Quality Peaks)
            
        Returns:
            DiffractionQuality enum indicating the classification
            
        Notes:
            DQI (Diffraction Quality Index) = n_pat_spots / n_dif_spots
            Classification uses DQI threshold comparison via good_rule parameter
        """
        if n_dif_spots < 3:
            return DiffractionQuality.NO_DIFFRACTION
        elif n_dif_spots < 10:
            return DiffractionQuality.POOR_DIFFRACTION
        elif self.config.good_rule * n_dif_spots > n_pat_spots:
            # DQI (n_pat_spots/n_dif_spots) is below good_rule threshold
            return DiffractionQuality.BAD_DIFFRACTION
        else:
            # DQI (n_pat_spots/n_dif_spots) meets good_rule threshold
            return DiffractionQuality.GOOD_DIFFRACTION

    def update_diffraction_quality(self, df: pd.DataFrame) -> None:
        """
        Update diffraction quality classifications in a DataFrame using DQI (Diffraction Quality Index).
        
        Args:
            df: DataFrame containing diffraction analysis data
            
        Notes:
            - Updates in place, modifying the 'DifQuality' column
            - Also updates 'FilteredPeaks' and 'FTPeaks' values for consistency
            - Classifications are based on sum threshold and DQI calculations
            - DQI (Diffraction Quality Index) = LQP / Diffraction Peaks
        """
        mean_sum = df['Sum'].mean()
        threshold = mean_sum / self.config.grid_rule

        def classify_row(row: pd.Series) -> Tuple[str, int, int]:
            """Classify a single row of data using DQI (Diffraction Quality Index)."""
            if row['Sum'] < threshold:
                return DiffractionQuality.GRID.value, 0, 0
            
            n_dif_spots = row['FilteredPeaks']
            n_pat_spots = row['FTPeaks']
            
            # Apply DQI (Diffraction Quality Index) classification
            quality = self._classify_diffraction(n_dif_spots, n_pat_spots)
            
            if quality == DiffractionQuality.GRID:
                return quality.value, 0, 0
            else:
                return quality.value, n_dif_spots, n_pat_spots

        # Apply classification to each row
        df[['DifQuality', 'FilteredPeaks', 'FTPeaks']] = df.apply(
            lambda row: classify_row(row), 
            axis=1, 
            result_type='expand'
        )

    def get_quality_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics about diffraction quality classifications.
        
        Args:
            df: DataFrame containing diffraction analysis results
            
        Returns:
            Dictionary containing quality statistics:
            - Counts and percentages for each quality classification
            - Average spots and patterns for each classification
            - Overall quality metrics
        """
        stats = {
            'total_images': len(df),
            'quality_counts': df['DifQuality'].value_counts().to_dict(),
            'quality_percentages': (df['DifQuality'].value_counts(normalize=True) * 100).to_dict(),
            'average_spots': df.groupby('DifQuality')['FilteredPeaks'].mean().to_dict(),
            'average_patterns': df.groupby('DifQuality')['FTPeaks'].mean().to_dict(),
            'overall_good_percentage': (
                len(df[df['DifQuality'] == DiffractionQuality.GOOD_DIFFRACTION.value]) / 
                len(df) * 100
            )
        }
        return stats

@dataclass
class PlotConfig:
    """Configuration for plot styling and saving."""
    figsize: Tuple[int, int] = (8, 8)
    dpi: int = 300
    cmap_heatmap: str = 'gist_heat'
    quality_colors: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default quality colors if not provided."""
        if self.quality_colors is None:
            self.quality_colors = {
                'Good diffraction': '#66c2a5',
                'Bad diffraction': '#fc8d62',
                'Poor diffraction': '#ffd92f',
                'No diffraction': '#8da0cb',
                'Grid': '#4d4d4d'
            }

class DataVisualizer:
    """Handles all plotting and visualization tasks for diffraction analysis."""
    
    def __init__(self, save_directory: str, plot_config: Optional[PlotConfig] = None):
        """
        Initialize DataVisualizer with configuration.
        
        Args:
            save_directory: Directory path for saving plots
            plot_config: Optional PlotConfig object for custom plot styling
        """
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.plot_config = plot_config or PlotConfig()
        
    def plot_diffraction_map(self,
                           data_array: np.ndarray,
                           folder_name: str,
                           title_suffix: str = '',
                           colorbar_label: Optional[str] = None) -> None:
        """
        Plot and save a diffraction map.
        
        Args:
            data_array: 2D array of diffraction data
            folder_name: Name of the folder being processed
            title_suffix: Optional suffix for the plot title
            colorbar_label: Optional custom label for the colorbar
        """
        plt.figure(figsize=self.plot_config.figsize, dpi=self.plot_config.dpi)
        
        # Create main plot
        im = plt.imshow(data_array, 
                       cmap=self.plot_config.cmap_heatmap,
                       interpolation='nearest')
        
        # Configure colorbar
        cbar = self._configure_colorbar(im)
        
        # Set labels and title
        self._set_plot_labels(folder_name, title_suffix, colorbar_label, cbar)
        
        # Save plot
        save_path = self._get_save_path(folder_name, 'diffraction_map', title_suffix)
        self._save_plot(save_path)

    def plot_diffraction_quality_map(self,
                                   quality_array: np.ndarray,
                                   folder_name: str) -> None:
        """
        Plot and save a diffraction quality heatmap.
        
        Args:
            quality_array: 2D array of quality classifications
            folder_name: Name of the folder being processed
        """
        plt.figure(figsize=self.plot_config.figsize, dpi=self.plot_config.dpi)
        
        # Create custom colormap for quality levels
        quality_levels = list(self.plot_config.quality_colors.keys())
        colors = [self.plot_config.quality_colors[level] for level in quality_levels]
        cmap = mcolors.ListedColormap(colors)
        
        # Create quality mapping
        quality_to_int = {quality: i for i, quality in enumerate(quality_levels)}
        int_array = np.vectorize(quality_to_int.get)(quality_array)
        
        # Set up normalization
        bounds = np.arange(len(quality_levels) + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # Create main plot
        plt.imshow(int_array, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Configure colorbar
        cbar = plt.colorbar(ticks=np.arange(len(quality_levels)) + 0.5)
        cbar.ax.set_yticklabels(quality_levels)
        
        # Set title
        plt.title(f'Diffraction Quality: {folder_name}', fontsize=12)
        
        # Save plot
        save_path = self._get_save_path(folder_name, 'diffraction_quality_map')
        self._save_plot(save_path)

    def _configure_colorbar(self, im: plt.cm.ScalarMappable) -> plt.colorbar:
        """
        Configure colorbar properties.
        
        Args:
            im: The image mappable object
            
        Returns:
            Configured colorbar object
        """
        cbar = plt.colorbar(im, aspect=10, shrink=0.8, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        # Ensure integer ticks for count data
        tick_locator = ticker.MaxNLocator(integer=True)
        cbar.locator = tick_locator
        cbar.update_ticks()
        
        return cbar

    def _set_plot_labels(self,
                        folder_name: str,
                        title_suffix: str,
                        colorbar_label: Optional[str],
                        cbar: plt.colorbar) -> None:
        """
        Set appropriate labels for plot elements.
        
        Args:
            folder_name: Name of the folder being processed
            title_suffix: Suffix for the plot title
            colorbar_label: Custom label for the colorbar
            cbar: Colorbar object to be labeled
        """
        # Set title based on suffix
        if title_suffix == '_diffraction_peaks':
            plt.title(f'Diffraction Peaks: {folder_name}', fontsize=12)
            cbar.set_label('Number of Diffraction Peaks', fontsize=12)
        elif title_suffix == '_lqp':
            plt.title(f'LQP: {folder_name}', fontsize=12)
            cbar.set_label('LQP Number', fontsize=12)
        else:
            plt.title(f'Diffraction Map: {folder_name} {title_suffix}', fontsize=12)
            cbar.set_label(colorbar_label or 'Diff. Int.', fontsize=12)

    def _get_save_path(self,
                      folder_name: str,
                      plot_type: str,
                      suffix: str = '') -> Path:
        """
        Generate save path for plot file.
        
        Args:
            folder_name: Name of the folder being processed
            plot_type: Type of plot being saved
            suffix: Optional suffix for filename
            
        Returns:
            Path object for save location
        """
        filename = f'{folder_name}_{plot_type}{suffix}.png'
        return self.save_directory / filename

    def _save_plot(self, save_path: Path) -> None:
        """
        Save plot to file and clean up.
        
        Args:
            save_path: Path where plot should be saved
        """
        plt.savefig(save_path, bbox_inches='tight', dpi=self.plot_config.dpi)
        plt.close()  # Close the plot to free memory
        log_print(f"Saved plot to {save_path}")

    def create_summary_plots(self,
                           df: pd.DataFrame,
                           folder_name: str) -> None:
        """
        Create summary plots for a complete analysis.
        
        Args:
            df: DataFrame containing analysis results
            folder_name: Name of the folder being processed
        """
        # Distribution of quality classifications
        plt.figure(figsize=(10, 6))
        quality_counts = df['DifQuality'].value_counts()
        quality_counts.plot(kind='bar')
        plt.title(f'Distribution of Diffraction Quality: {folder_name}')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        save_path = self._get_save_path(folder_name, 'quality_distribution')
        self._save_plot(save_path)
        
        # Correlation between spots and patterns
        plt.figure(figsize=(8, 8))
        plt.scatter(df['FilteredPeaks'], df['FTPeaks'], alpha=0.5)
        plt.xlabel('Number of Diffraction Peaks')
        plt.ylabel('LQP Number')
        plt.title(f'Diffraction Peaks vs LQP Correlation: {folder_name}')
        save_path = self._get_save_path(folder_name, 'spots_correlation')
        self._save_plot(save_path)

@dataclass
class FileIndices:
    """Container for file indexing information."""
    date_code: str  # Date code of the first file; a block may span several dates
    min_index: int
    max_index: int
    total_files: int
    file_suffix: str = '_integrated_movie.mrc'  # Default suffix, can be '_0_movie.mrc'
    index_width: int = 5  # Zero-padding width of the index, detected from filenames
    files: List[str] = field(default_factory=list)  # File names in acquisition order

# Diffraction image names: YYYYMMDD_<index>_integrated_movie.mrc or YYYYMMDD_<index>_0_movie.mrc.
# The date comes from the acquisition computer and changes at midnight; the index
# keeps counting across the date change.
MOVIE_FILE_PATTERN = re.compile(r"^(\d{8})_(\d+)(_integrated_movie\.mrc|_0_movie\.mrc)$")

@dataclass
class ProcessingStatus:
    """Container for processing status information."""
    is_processed: bool
    last_modified: Optional[datetime] = None
    error_message: Optional[str] = None

class FileHandler:
    def __init__(self, 
                 base_directory: str,
                 config: MicroscopeConfig,
                 log_filename: str = "processing_log.txt"):
        if not os.path.isdir(base_directory):
            raise ValueError(f"Base directory does not exist: {base_directory}")
        if not isinstance(config, MicroscopeConfig):
            raise TypeError("config must be an instance of MicroscopeConfig")
            
        self.base_dir = Path(base_directory)
        self.config = config
        self.log_path = self.base_dir / "REyes_logs" / log_filename
        self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """Initialize log file if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.touch()
            log_print(f"Created new processing log file: {self.log_path}")
    
    def _generate_block_uid(self) -> str:
        """Generate a unique identifier for a block.
        
        Returns:
            Unique identifier string in UUID format
        """
        return str(uuid.uuid4())

    def create_reyes_json(self, folder_path: str, indices: FileIndices) -> None:
        """Create a Reyes.json file in the specified folder.

        Args:
            folder_path: Path to the folder where the JSON file should be created
            indices: FileIndices object containing date and index information

        Notes:
            'format' expands min_id..max_id into file names for blocks recorded on a
            single date. 'files' lists every file of the block in acquisition order
            and is also correct for blocks recorded across midnight.
        """
        try:
            json_data = {
                'version': 1,
                'min_id': indices.min_index,
                'max_id': indices.max_index,
                'format': f'{indices.date_code}_%0{indices.index_width}d{indices.file_suffix}',
                'uid': self._generate_block_uid(),
                'files': indices.files
            }

            json_path = Path(folder_path) / 'reyes.json'
            with json_path.open('w') as f:
                json.dump(json_data, f, indent=4)

            log_print(f"Created Reyes.json file in {folder_path}")

        except Exception as e:
            log_print(f"Error creating Reyes.json file: {str(e)}", logging.ERROR)
    
    def determine_indices(self, folder_path: str) -> Optional[FileIndices]:
        """
        Determine the files of a block and their index range from MRC filenames.

        Args:
            folder_path: Path to folder containing MRC files

        Returns:
            FileIndices object for the block, or None if the folder holds no valid,
            unambiguous block

        Note:
            Expects filenames in format "YYYYMMDD_XXXXX_integrated_movie.mrc" or "YYYYMMDD_XXXXX_0_movie.mrc".
            The index may be zero-padded or not and may outgrow its padding
            (e.g. "20250101_107458_0_movie.mrc"). Files are ordered by index, so a
            block recorded across midnight keeps all its files. If the folder holds
            a repeated acquisition (a complete block followed, after a large index
            gap, by another complete block), the latest acquisition is used.
        """
        try:
            folder = Path(folder_path)
            if not folder.exists():
                return None

            files_by_suffix: Dict[str, List[Tuple[int, str, str, int]]] = {}
            for file in folder.glob("*.mrc"):
                match = MOVIE_FILE_PATTERN.match(file.name)
                if match:
                    date_code, index, suffix = match.groups()
                    files_by_suffix.setdefault(suffix, []).append(
                        (int(index), date_code, file.name, len(index)))

            if not files_by_suffix:
                return None

            suffix = max(files_by_suffix, key=lambda s: len(files_by_suffix[s]))
            for other, other_files in files_by_suffix.items():
                if other != suffix:
                    log_print(f"Warning: Ignoring {len(other_files)} '*{other}' files in {folder.name}; "
                              f"the block uses '*{suffix}' files", logging.WARNING)

            files = sorted(files_by_suffix[suffix])
            runs = self._consecutive_runs(files)
            selected = runs[-1]

            if len(runs) > 1:
                previous = runs[-2]
                gap = selected[0][0] - previous[-1][0] - 1
                side = int(np.sqrt(len(selected)))
                is_repeat = (side * side == len(selected)
                             and gap >= len(selected)
                             and all(len(run) <= len(selected) for run in runs[:-1]))
                if not is_repeat:
                    present = {entry[0] for entry in files}
                    missing = sorted(set(range(files[0][0], files[-1][0] + 1)) - present)
                    index_counts = Counter(entry[0] for entry in files)
                    duplicates = sorted(index for index, count in index_counts.items() if count > 1)
                    details = []
                    if missing:
                        shown = ', '.join(str(i) for i in missing[:10])
                        details.append(f"{len(missing)} missing indices ({shown}{', ...' if len(missing) > 10 else ''})")
                    if duplicates:
                        details.append(f"duplicate indices {duplicates[:10]}")
                    log_print(f"Warning: Skipping block {folder.name}: file indices are not one continuous "
                              f"acquisition: {'; '.join(details)}", logging.WARNING)
                    return None
                ignored = sum(len(run) for run in runs[:-1])
                log_print(f"Warning: {folder.name} holds a repeated acquisition; using the latest "
                          f"{len(selected)} files ({selected[0][2]} .. {selected[-1][2]}) and ignoring "
                          f"{ignored} earlier files", logging.WARNING)

            dates = sorted({entry[1] for entry in selected})
            if len(dates) > 1:
                log_print(f"Block {folder.name} spans dates {', '.join(dates)}")

            # The padding width is the shortest index seen: longer indices are
            # numbers that simply outgrew the padding, not a wider padding.
            return FileIndices(
                date_code=selected[0][1],
                min_index=selected[0][0],
                max_index=selected[-1][0],
                total_files=len(selected),
                file_suffix=suffix,
                index_width=min(entry[3] for entry in selected),
                files=[entry[2] for entry in selected]
            )

        except Exception as e:
            log_print(f"Error determining indices in {folder_path}: {str(e)}",
                     logging.ERROR)
            return None

    @staticmethod
    def _consecutive_runs(files: List[Tuple[int, str, str, int]]) -> List[List[Tuple[int, str, str, int]]]:
        """Split index-sorted files into runs of consecutive indices."""
        runs: List[List[Tuple[int, str, str, int]]] = []
        for entry in files:
            if runs and entry[0] == runs[-1][-1][0] + 1:
                runs[-1].append(entry)
            else:
                runs.append([entry])
        return runs

    def get_coordinates_for_file(self, 
                               log_file_path: str, 
                               filename: str) -> Optional[List[float]]:
        """
        Retrieve stage coordinates from SerialEM log file.
        
        Args:
            log_file_path: Path to the log file
            filename: Name of the file to find coordinates for
            
        Returns:
            List of [x, y, z] coordinates or None if not found
        """
        try:
            log_path = Path(log_file_path)
            if not log_path.exists():
                log_print(f"Log file not found: {log_file_path}", logging.WARNING)
                log_print("Proceeding without coordinates", logging.INFO)
                return None

            pattern_file = re.compile(re.escape(filename))
            pattern_stage = re.compile(r"Stage\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
            
            lines = log_path.read_text().splitlines()
            for i, line in enumerate(lines):
                if pattern_file.search(line):
                    skip_lines = self.config.serialEM_log
                    if i + skip_lines < len(lines):
                        stage_match = pattern_stage.search(lines[i + skip_lines])
                        if stage_match:
                            return [float(stage_match.group(j)) for j in range(1, 4)]

            # The Stage line was not at its fixed offset. A macro that logs
            # anything between Record and ReportStageXYZ moves it, and the frame
            # then lost its position without a word, so it could never become a
            # target. Look on for it, but never past the next frame's file line:
            # a position taken from there would be another frame's.
            pattern_other_file = re.compile(r"\.mrc\b")
            for i, line in enumerate(lines):
                if pattern_file.search(line):
                    for k in range(i + 1, min(i + 1 + STAGE_LINE_LOOKAHEAD, len(lines))):
                        if pattern_other_file.search(lines[k]) and not pattern_file.search(lines[k]):
                            break
                        stage_match = pattern_stage.search(lines[k])
                        if stage_match:
                            log_print(f"Stage position for {filename} found {k - i} line(s) after its "
                                      f"file line instead of {self.config.serialEM_log}: the log has "
                                      f"extra lines between Record and ReportStageXYZ", logging.WARNING)
                            return [float(stage_match.group(j)) for j in range(1, 4)]

            log_print(f"No coordinates found for {filename}", logging.WARNING)
            return None

        except Exception as e:
            log_print(f"Error getting coordinates for {filename}: {str(e)}", 
                     logging.ERROR)
            return None

    def is_processed(self, block_name: str) -> ProcessingStatus:
        """
        Check if a block has been processed.
        
        Args:
            block_name: Name of the block to check
            
        Returns:
            ProcessingStatus object containing processing information
        """
        try:
            processed_blocks = self._read_processed_blocks()
            if block_name in processed_blocks:
                timestamp = processed_blocks[block_name].get('timestamp')
                if timestamp:
                    last_modified = datetime.fromisoformat(timestamp)
                else:
                    last_modified = None
                return ProcessingStatus(
                    is_processed=True,
                    last_modified=last_modified
                )
            return ProcessingStatus(is_processed=False)

        except Exception as e:
            return ProcessingStatus(
                is_processed=False,
                error_message=str(e)
            )

    def log_processed_block(self, 
                          block_name: str, 
                          metadata: Optional[Dict] = None) -> None:
        """
        Log a block as processed with optional metadata.
        
        Args:
            block_name: Name of the block to log
            metadata: Optional dictionary of metadata to store
        """
        try:
            timestamp = datetime.now().isoformat()
            processed_blocks = self._read_processed_blocks()
            
            block_info = {
                'timestamp': timestamp
            }
            if metadata:
                # Validate metadata to prevent corruption
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            block_info[key] = value
                        else:
                            log_print(f"Skipping invalid metadata type for {key}: {type(value)}", logging.WARNING)
                else:
                    log_print(f"Skipping non-dict metadata: {type(metadata)}", logging.WARNING)
                
            processed_blocks[block_name] = block_info
            self._write_processed_blocks(processed_blocks)
            
            log_print(f"Logged processing of block: {block_name}")

        except Exception as e:
            log_print(f"Error logging processed block {block_name}: {str(e)}", 
                     logging.ERROR)

    def _read_processed_blocks(self) -> Dict:
        """
        Read and parse the processing log file.
        
        Returns:
            Dictionary of processed blocks and their metadata
        """
        try:
            content = self.log_path.read_text()
            processed_blocks = {}
            
            for line in content.splitlines():
                if line.strip():
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        block_name = parts[0].strip()
                        try:
                            metadata = json.loads(parts[1].strip())
                            if isinstance(metadata, dict):
                                processed_blocks[block_name] = metadata
                        except (json.JSONDecodeError, ValueError):
                            # Skip corrupted entries instead of storing them as timestamp
                            log_print(f"Skipping corrupted log entry for {block_name}: {parts[1][:100]}...", logging.WARNING)
                            continue
                    else:
                        processed_blocks[line.strip()] = {}
                        
            return processed_blocks

        except Exception as e:
            log_print(f"Error reading processing log: {str(e)}", logging.ERROR)
            return {}

    def _write_processed_blocks(self, processed_blocks: Dict) -> None:
        """
        Write processed blocks information to log file.
        
        Args:
            processed_blocks: Dictionary of blocks and their metadata
        """
        try:
            with self.log_path.open('w') as f:
                for block_name, metadata in processed_blocks.items():
                    if metadata:
                        # Use json.dumps instead of str() to ensure proper JSON format
                        f.write(f"{block_name}: {json.dumps(metadata)}\n")
                    else:
                        f.write(f"{block_name}\n")

        except Exception as e:
            log_print(f"Error writing to processing log: {str(e)}", logging.ERROR)

    def get_unprocessed_folders(self) -> List[Path]:
        """
        Get list of unprocessed folders in base directory.
        
        Returns:
            List of Path objects for unprocessed folders
        """
        processed_blocks = self._read_processed_blocks()
        
        unprocessed = []
        for folder in self.base_dir.iterdir():
            if folder.is_dir():
                # First check if the directory contains any MRC files
                has_mrc = any(folder.glob("*.mrc"))
                if has_mrc and folder.name not in processed_blocks:
                    unprocessed.append(folder)
                elif not has_mrc:
                    log_print(f"Skipping directory without MRC files: {folder.name}")
        
        return unprocessed

    def cleanup_old_logs(self, days_threshold: int = 30) -> None:
        """
        Clean up old processing records.
        
        Args:
            days_threshold: Number of days after which to remove records
        """
        try:
            current_time = datetime.now()
            processed_blocks = self._read_processed_blocks()
            
            updated_blocks = {}
            for block_name, metadata in processed_blocks.items():
                if 'timestamp' in metadata:
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    days_old = (current_time - timestamp).days
                    if days_old <= days_threshold:
                        updated_blocks[block_name] = metadata
                else:
                    updated_blocks[block_name] = metadata
            
            self._write_processed_blocks(updated_blocks)
            
        except Exception as e:
            log_print(f"Error cleaning up old logs: {str(e)}", logging.ERROR)

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured argument parser with necessary options
    """
    parser = argparse.ArgumentParser(
        description='Process diffraction patterns and create quality maps.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Microscope configuration
    parser.add_argument(
        '--microscope',
        type=str,
        choices=list(MICROSCOPE_CONFIGS.keys()),
        default="Arctica-CETA",
        help='Microscope configuration to use (default: Arctica-CETA)'
    )

    # Folder to process
    parser.add_argument(
        '--folder',
        type=str,
        default=None,
        help='Optional: Name of the folder to process. If not provided, all folders will be scanned.'
    )

    # Processing options
    parser.add_argument(
        '--targets-per-block',
        type=int,
        default=4,
        help='Number of top targets to select per block',
        choices=range(1, 11)
    )

    parser.add_argument(
        '--skip-processed',
        action='store_true',
        help='Skip previously processed blocks'
    )

    parser.add_argument(
        '--legacy-parse',
        action='store_true',
        help='Load frames exactly as REyes 3.4.0 and earlier did (no stack summing, bad line '
             'removal or small-frame detection parameters), to reproduce old results'
    )

    parser.add_argument(
        '--proc-blocks',
        type=int,
        help='Limit the number of blocks to process',
        metavar='N'
    )

    # Output file names
    parser.add_argument(
        '--output-csv',
        type=str,
        default='dif_map_sums.csv',
        help='Name of the main output CSV file'
    )

    parser.add_argument(
        '--targets-sum-csv',
        type=str,
        default='targets_sum.csv',
        help='Name of the CSV file for sum-based targets'
    )

    parser.add_argument(
        '--targets-spots-csv',
        type=str,
        default='targets_spots.csv',
        help='Name of the CSV file for spot-based targets'
    )

    parser.add_argument(
        '--targets-quality-csv',
        type=str,
        default='targets_quality.csv',
        help='Name of the CSV file for quality-based targets'
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

        # Initialize logging
        setup_logging('dmp_processing.log', 'REyes_logs')
        print_banner()

        log_print(f"\nREyes DMP v{__version__} will process diffraction map blocks\n")
        log_print(f"Using {args.microscope} microscope configuration")
        if args.legacy_parse:
            log_print("Using legacy frame loading (REyes 3.4.0 and earlier)")

        # Initialize configuration and processors
        config = MICROSCOPE_CONFIGS[args.microscope]
        image_processor = ImageProcessor(config, legacy_parse=args.legacy_parse)
        diffraction_analyzer = DiffractionAnalyzer(config, image_processor)
        data_visualizer = DataVisualizer('dif_maps/diff_blocks_maps')
        file_handler = FileHandler(os.getcwd(), config)

        # Create output directory
        save_directory = Path('dif_maps/diff_blocks_maps')
        save_directory.mkdir(exist_ok=True)

        if args.folder:
            folder = Path(args.folder)
            if not folder.exists() or not folder.is_dir():
                log_print(f"Specified folder does not exist or is not a directory: {folder}", logging.ERROR)
                return 1
            folders_to_process = [folder]
        else:
            folders_to_process = [f for f in Path.cwd().iterdir() if f.is_dir()]
        
        # Counter for actually processed blocks
        processed_block_count = 0
        
        for folder in folders_to_process:
            folder_name = folder.name
            
            # Skip if already processed and flag is set
            if args.skip_processed:
                if file_handler.is_processed(folder_name).is_processed:
                    log_print(f"Skipping previously processed folder: {folder_name}")
                    continue

                # Check if directory has MRC files
                has_mrc = any(folder.glob("*.mrc"))
                if not has_mrc:
                    log_print(f"Skipping directory without MRC files: {folder_name}")
                    continue
                log_print(f"Processing unprocessed folder: {folder_name}")

            log_print(f"Processing folder: {folder_name}")
            image_processor.reset_block_log()
            
            # Determine the files of the block
            indices = file_handler.determine_indices(folder)
            if indices is None:
                log_print(f"No valid block of .mrc files found in {folder_name}. Skipping.")
                continue

            # Initialize data collection
            csv_data = []
            sums = []
            filtered_peaks = []
            ft_peaks = []
            quality_labels = []

            # Find SerialEM log file(s): the block's log, newest resume log first
            log_files = block_logs(folder_name)
            if not log_files:
                log_print(f"No SerialEM log file found for {folder_name}. Proceeding without coordinates.", logging.WARNING)

            # Process each file of the block in acquisition order
            failed_files = []
            for idx, filename in enumerate(indices.files):
                file_path = folder / filename

                # Process diffraction pattern
                result = diffraction_analyzer.process_mrc_file(str(file_path))
                if result is None:
                    failed_files.append(filename)
                    continue

                # Collect results
                sums.append(result.total_sum)
                filtered_peaks.append(result.n_diffraction_spots)
                ft_peaks.append(result.n_pattern_spots)
                quality_labels.append(result.quality)

                # Get coordinates if log file exists: from the newest log
                # that names the frame
                coords = None
                for block_log in log_files:
                    coords = file_handler.get_coordinates_for_file(str(block_log), filename)
                    if coords is not None:
                        break
                if not log_files:
                    coords = [None, None, None]

                # Add to CSV data
                csv_data.append([
                    str(file_path.absolute()),
                    result.total_sum,
                    result.n_diffraction_spots,
                    result.n_pattern_spots,
                    idx,
                    coords,
                    result.quality
                ])

            # Verify data consistency
            if failed_files:
                shown = ', '.join(failed_files[:5]) + (', ...' if len(failed_files) > 5 else '')
                log_print(f"Warning: Skipping block {folder_name}: {len(failed_files)} of "
                          f"{indices.total_files} files could not be processed ({shown})",
                          logging.WARNING)
                continue

            # Create DataFrame and process data
            try:
                array_size = int(np.sqrt(len(sums)))
                if array_size * array_size != len(sums):
                    raise ValueError(f"The number of sums ({len(sums)}) is not a perfect square.")

                # Create Reyes.json file only for a complete, valid block
                file_handler.create_reyes_json(folder, indices)

                # Update array indices
                for idx, row in enumerate(csv_data):
                    csv_data[idx][4] = np.unravel_index(idx, (array_size, array_size))

                # Create and process DataFrame
                df = pd.DataFrame(
                    csv_data,
                    columns=['Path', 'Sum', 'FilteredPeaks', 'FTPeaks', 'ItemNumber', 'Coordinates', 'DifQuality']
                )
                
                match = re.search(r'_(\d+)', folder.name)
                if match:
                    df['Block'] = int(match.group(1))
                else:
                    df['Block'] = processed_block_count  # fallback


                # Update diffraction quality
                diffraction_analyzer.update_diffraction_quality(df)

                # Define dif_map folder and file
                dif_map_folder = Path("dif_maps")  
                dif_map_folder.mkdir(parents=True, exist_ok=True) 
                dif_map_csv = dif_map_folder / args.output_csv

                # Save main results
                write_header = not dif_map_csv.exists()
                df.to_csv(dif_map_csv, mode='a', header=write_header, index=False)
                log_print(f"Data appended to {dif_map_csv}")

                # Process top targets
                # IMPORTANT: Only 'Good diffraction' and 'Bad diffraction' patterns can be targets
                # Exclude 'Poor diffraction', 'No diffraction', and 'Grid' from all target lists
                eligible_df = df[df['DifQuality'].isin(['Good diffraction', 'Bad diffraction'])].copy()
                
                if len(eligible_df) == 0:
                    log_print(f"Warning: No eligible targets found in {folder_name} (no Good/Bad diffraction patterns)")
                    top_items_sum = pd.DataFrame()
                    top_items_spots = pd.DataFrame() 
                    top_items_quality = pd.DataFrame()
                else:
                    top_items_sum = eligible_df.nlargest(args.targets_per_block, 'Sum').copy()
                    top_items_spots = eligible_df.nlargest(args.targets_per_block, 'FilteredPeaks').copy()
                    top_items_quality = eligible_df[eligible_df['DifQuality'] == 'Good diffraction'].copy()

                # Update filenames (only for non-empty DataFrames)
                for items in [top_items_sum, top_items_spots, top_items_quality]:
                    if len(items) > 0:
                        items.loc[:, 'Filename'] = items['Path'].apply(lambda x: Path(x).name)

                # Ensure correct number of quality items
                if len(top_items_quality) > args.targets_per_block:
                    top_items_quality = top_items_quality.head(args.targets_per_block)

                targets_folder = Path("targets")
                targets_folder.mkdir(parents=True, exist_ok=True)


                # Save target results (only for non-empty DataFrames)
                for items, filename in [
                    (top_items_sum, targets_folder/args.targets_sum_csv),
                    (top_items_spots, targets_folder/args.targets_spots_csv),
                    (top_items_quality, targets_folder/args.targets_quality_csv)
                ]:
                    if len(items) > 0:
                        items.to_csv(filename, mode='a', header=not Path(filename).exists(), index=False)
                    else:
                        log_print(f"No targets to save for {Path(filename).name} in {folder_name}")

                # Create visualization arrays
                arrays = {
                    '': np.array(sums).reshape((array_size, array_size)),
                    '_diffraction_peaks': np.array(filtered_peaks).reshape((array_size, array_size)),
                    '_lqp': np.array(ft_peaks).reshape((array_size, array_size))
                }

                # Generate plots
                for suffix, array in arrays.items():
                    data_visualizer.plot_diffraction_map(array, folder_name, title_suffix=suffix)
                
                quality_array = np.array(quality_labels).reshape((array_size, array_size))
                data_visualizer.plot_diffraction_quality_map(quality_array, folder_name)

                # Mark as processed
                file_handler.log_processed_block(folder_name)
                
                # Increment the counter for successfully processed blocks
                processed_block_count += 1
                
                # Check if we've reached the processing limit
                if args.proc_blocks is not None and processed_block_count >= args.proc_blocks:
                    log_print(f'Reached processing limit of {args.proc_blocks} blocks. Stopping.')
                    return 0

            except ValueError as e:
                log_print(f"Warning: Skipping block {folder_name}: {str(e)}", logging.WARNING)
                continue

        log_print('All diffraction map blocks have been processed!')
        return 0

    except Exception as e:
        log_print(f"Processing failed: {str(e)}", logging.ERROR)
        logging.exception("Detailed error traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())