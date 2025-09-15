from typing import Optional, Tuple
from pathlib import Path
import os
import re
import logging
from pyREyes.lib.REyes_logging import log_print


def find_mrc_and_mdoc_files() -> Tuple[Optional[Path], Optional[Path]]:
    """Search for .mrc and corresponding .mdoc files in current directory."""
    current_dir = Path.cwd()
    
    for mrc_file in current_dir.glob("*.mrc"):
        mdoc_file = Path(f"{mrc_file}.mdoc")
        if mdoc_file.exists():
            return mrc_file, mdoc_file
            
    return None, None

def find_nav_file() -> Optional[str]:
    """Find the first .nav file that matches the pattern in current directory."""
    nav_pattern = re.compile(r'(.+)_(\d+)LM_(\d+)x(\d+)_nav\.nav$')
    
    try:
        for file in os.listdir():
            if nav_pattern.match(file):
                log_print(f"Found matching .nav file: {file}")
                return file
        
        log_print("No matching .nav file found", logging.ERROR)
        return None
        
    except Exception as e:
        log_print(f"Error searching for .nav file: {str(e)}", logging.ERROR)
        return None
