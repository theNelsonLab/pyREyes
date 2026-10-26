"""
The stages must plot with Agg, whatever display the shell has.

A monitor started over ssh with X11 forwarding has DISPLAY set, so matplotlib
may choose TkAgg; a stage can then write every output and still be aborted by
Tk on exit ("Tcl_AsyncDelete: async handler deleted by the wrong thread"), and
the monitor treats it as failed.

Each module is imported in a fresh interpreter with MPLBACKEND unset, so the
backend is what the module itself selects: on macOS the default would be
'macosx', under ssh -X it would be 'TkAgg'.
"""

import os
import subprocess
import sys

import pytest

from pyREyes.REyes import run_with_timeout


PLOTTING_MODULES = [
    'pyREyes.grid_squares',
    'pyREyes.eucentricity',
    'pyREyes.dif_map',
    'pyREyes.write_sorted_targets_nav',
    'pyREyes.append_targets_nav',
    'pyREyes.mnt_maps_targets',
    'pyREyes.lib.diffraction.DiffractionDataProcessor',
    'pyREyes.lib.grid_processing.CentroidProcessor',
    'pyREyes.lib.image_processing.CartImage_utils',
    'pyREyes.lib.image_processing.image_processor',
    'pyREyes.lib.image_processing.PlottingManager',
]


@pytest.mark.parametrize('module', PLOTTING_MODULES)
def test_module_selects_agg(module):
    env = {k: v for k, v in os.environ.items() if k != 'MPLBACKEND'}
    env['DISPLAY'] = env.get('DISPLAY', 'localhost:10.0')
    # The child imports pyREyes from wherever this test run does.
    env['PYTHONPATH'] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run(
        [sys.executable, '-c',
         f'import {module}, matplotlib; print(matplotlib.get_backend())'],
        capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip().splitlines()[-1].lower() == 'agg'


def test_monitor_stages_get_agg(monkeypatch):
    monkeypatch.delenv('MPLBACKEND', raising=False)
    result = run_with_timeout(
        [sys.executable, '-c', 'import os; print(os.environ.get("MPLBACKEND"))'],
        timeout=60)
    assert result.stdout.strip() == 'Agg'
