"""
The Ceta package writes the same set-aside and resume records as the DE Apollo
package, and the monitor reads them the same way with a Ceta configuration.

Two things differ on the Ceta and are checked here:

- its closed-valve line ends `action=not_marked_done` (the Ceta writes its
  frames in place, so nothing is copied); the monitor does not read `action`;
- its step-scan log puts the `Stage x y z` line two lines after the line naming
  the frame (one on the Apollo), and set-aside records may sit between frames.

The logs are synthetic, built from the exact strings the Ceta macros write.
"""

import os
import time

from conftest import TracingProcessor
from pyREyes.lib import REyes_serialem as serialem
from test_monitor_keep_going import ALL, DONE, drive_until_stop, log_session, status

import pyREyes.dif_map as dm

CETA = 'Arctica-CETA'


def ceta_session(tmp_path, monkeypatch, logs):
    root = log_session(tmp_path, 'g-reyes', {})
    for i, (block, text) in enumerate(sorted(logs.items())):
        path = root / f'g_{block}.log'
        path.write_text(text)
        os.utime(path, (time.time() + 100 + i, time.time() + 100 + i))
    monkeypatch.chdir(root)
    return root, TracingProcessor(working_directory=str(root), microscope=CETA,
                                  stepscan_only=True)


def dif_map_folders(trace):
    return [t.split()[2] for t in trace if t.startswith('MODULE dif_map')]


def test_a_ceta_valve_closed_square_is_set_aside(tmp_path, monkeypatch):
    root, p = ceta_session(tmp_path, monkeypatch, {
        1: DONE,
        2: 'x\nVALVE_CLOSED | stage=STPSCN | square=2 | action=not_marked_done\n'
           'BLOCK_SKIPPED | stage=STPSCN | square=2 | item=3 | reason=valve_closed\n',
        3: DONE + 'STPSCN_SUMMARY | squares=3 | collected=2 | skipped=0 | set_aside=1\n'
                  'STPSCN_RESULT | result=partial | collected=2 | set_aside=1\n' + ALL})
    trace = drive_until_stop(p)
    assert dif_map_folders(trace) == ['folder=g_1', 'folder=g_3']
    assert p.skipped_blocks[2]['reason'] == 'valve_closed'
    assert trace[-1] == 'STATE COMPLETED'
    assert status(root)['state'] == 'complete'


def test_a_ceta_square_set_aside_and_mapped_on_resume_is_mapped(tmp_path, monkeypatch):
    root, p = ceta_session(tmp_path, monkeypatch, {
        1: DONE,
        2: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=2 | item=3 | reason=error_move\n'})
    resumed = root / 'g_2-1.log'
    resumed.write_text('RESUME | stage=STPSCN | square=1 | action=skipped\n' + DONE
                       + 'STPSCN_RESULT | result=complete | collected=2 | set_aside=0\n' + ALL)
    os.utime(resumed, (time.time() + 200, time.time() + 200))

    trace = drive_until_stop(p, max_ticks=12)

    assert dif_map_folders(trace) == ['folder=g_1', 'folder=g_2']
    assert not p.skipped_blocks
    assert trace[-1] == 'STATE COMPLETED'


def test_a_ceta_step_scan_with_no_data_fails_the_grid(tmp_path, monkeypatch):
    root, p = ceta_session(tmp_path, monkeypatch, {
        1: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=1 | item=2 | reason=error_acquire\n'
           'STPSCN_SUMMARY | squares=1 | collected=0 | skipped=0 | set_aside=1\n'
           'STPSCN_RESULT | result=failed | collected=0 | set_aside=1\n'})
    trace = drive_until_stop(p)
    assert not dif_map_folders(trace)
    assert status(root)['state'] == 'failed'


def test_a_resumed_ceta_movie_stage_finishes_the_session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'targets').mkdir()
    (tmp_path / 'targets' / 'range_dfmv.txt').write_text('12\n14\n')
    (tmp_path / 's_targets.log').write_text(
        'Movie folder: s_mov1\nAcquired movie: s_mov1_integrated_movie.mrc\n'
        'VALVE_CLOSED | stage=DFMV | item=2 | folder=s_mov2 | action=not_marked_done\n'
        'ITEM_SKIPPED | stage=DFMV | item=2 | reason=valve_closed\n')
    (tmp_path / 's_targets-1.log').write_text(
        'RESUME | stage=DFMV | item=1 | action=skipped | folder=s_mov1\n'
        'Movie folder: s_mov3\nAcquired movie: s_mov3_integrated_movie.mrc\n'
        'Completed movie acquisition\n'
        'Completed acquiring tilt movies for all crystals!\n')
    p = TracingProcessor(working_directory=str(tmp_path), microscope=CETA,
                         start_state='WAITING_FOR_MOVIES')

    trace = drive_until_stop(p, max_ticks=5)

    assert [t for t in trace if t.startswith('MOVIE')] == [
        'MOVIE s_mov1_integrated_movie.mrc', 'MOVIE s_mov3_integrated_movie.mrc']
    assert trace[-1] == 'STATE COMPLETED'
    listed = serialem.load_set_aside(tmp_path / 'targets')
    assert [(e['stage'], e['item'], e['reason']) for e in listed] == \
        [('DFMV', 2, 'valve_closed')]


def test_ceta_positions_survive_records_between_frames(tmp_path):
    """Stage two lines after the frame's file line, with records around it."""
    (tmp_path / 'REyes_logs').mkdir()
    handler = dm.FileHandler(str(tmp_path), dm.MICROSCOPE_CONFIGS[CETA])

    def frame(index, x):
        return [f'Current open image file is: D:\\data\\g_2\\20250101_{index}_integrated_movie.mrc',
                'Saved Z = 0',
                f'Stage {x:.2f} 20.00 30.00']

    log = tmp_path / 'g_2.log'
    log.write_text('\n'.join(
        ['RESUME | stage=STPSCN | square=1 | action=skipped']
        + frame(10001, 10.0) + frame(10002, 11.0)
        + ['Completed StepScan for current grid square',
           'BLOCK_SKIPPED | stage=STPSCN | square=3 | item=4 | reason=error_move']) + '\n')
    assert handler.get_coordinates_for_file(str(log), '20250101_10001_integrated_movie.mrc') \
        == [10.0, 20.0, 30.0]
    assert handler.get_coordinates_for_file(str(log), '20250101_10002_integrated_movie.mrc') \
        == [11.0, 20.0, 30.0]
