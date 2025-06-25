import json
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Dict

FILTER_FILE = Path(__file__).resolve().parent.parent.parent / "config" / "filtering_options.json"

@dataclass
class FilteringBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

def load_filtering_options(config_file=FILTER_FILE) -> Dict[str, Union[int, FilteringBounds]]:
    """Loads mixed-type filtering options from a JSON file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Filtering config file '{config_file}' not found.")

    with open(config_file, "r") as file:
        raw = json.load(file)

    filtering_options = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            filtering_options[key] = FilteringBounds(**val)
        else:
            filtering_options[key] = val  # It's just an int

    return filtering_options
