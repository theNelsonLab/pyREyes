import json
import logging
from pathlib import Path
from dataclasses import dataclass, fields


CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "microscope_configs.json"

@dataclass
class MicroscopeConfig:
    """Configuration settings for different microscope types."""
    scaling: float
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
    """Loads microscope configurations from a JSON file.

    Keys that are not MicroscopeConfig fields are ignored with one warning per
    key name. A missing field raises ValueError.
    """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, "r") as file:
        config_data = json.load(file)

    known = {f.name for f in fields(MicroscopeConfig)}
    unknown = set()
    configs = {}
    for name, params in config_data.items():
        unknown.update(key for key in params if key not in known)
        missing = sorted(known - params.keys())
        if missing:
            raise ValueError(
                f"Microscope configuration '{name}' in '{config_file}' is missing "
                f"required field(s): {', '.join(missing)}")
        configs[name] = MicroscopeConfig(**{k: v for k, v in params.items() if k in known})

    for key in sorted(unknown):
        logging.getLogger(__name__).warning(
            f"Ignoring unknown microscope configuration key '{key}' in '{config_file}'")

    return configs
