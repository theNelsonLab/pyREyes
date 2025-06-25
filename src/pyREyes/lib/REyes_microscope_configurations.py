import json
from pathlib import Path
from dataclasses import dataclass


CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "microscope_configs.json"

@dataclass
class MicroscopeConfig:
    """Configuration settings for different microscope types."""
    scaling: float
    dark_threshold: float
    camera_length: float  # in mm
    pixel_size: float    # in mm/pixel for 2k images
    wavelength: float    # in Ångströms
    merge_distance: float  # in microns for merging nearby centroids

    # For dif_map
    light_sigma_prim: float
    harsh_sigma_prim: float
    threshold_std_prim: float
    min_pixels_prim: int
    good_rule: float
    serialEM_log: int
    grid_rule: int


def load_microscope_configs(config_file=CONFIG_FILE):
    """Loads microscope configurations from a JSON file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as file:
        config_data = json.load(file)

    return {name: MicroscopeConfig(**params) for name, params in config_data.items()}
