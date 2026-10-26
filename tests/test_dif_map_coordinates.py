"""
A frame's stage position, read from the SerialEM log.

`dif-map-2` finds the line that names a frame's MRC and reads `Stage x y z` a
fixed number of lines later: one on the Apollo, two on the CETA. That makes the
step-scan macro's inner loop a contract -- nothing may be logged between Record
and ReportStageXYZ -- and when a macro broke it (e.g. a saved-frame check
echoed in between) the first frame of every grid square lost its position
without a word, and with it any chance of becoming a target.

The fixed offset is still tried first, so every log that worked reads exactly as
it did. Only when it fails is the Stage line looked for further on, and never
past the next frame: a position taken from there would be another frame's.

The log lines here have the form SerialEM writes on the DE Apollo.
"""

import pytest

import pyREyes.dif_map as dm


@pytest.fixture
def apollo(tmp_path):
    (tmp_path / 'REyes_logs').mkdir()
    return dm.FileHandler(str(tmp_path), dm.MICROSCOPE_CONFIGS['Arctica-Apollo-SM'])


@pytest.fixture
def ceta(tmp_path):
    (tmp_path / 'REyes_logs').mkdir()
    return dm.FileHandler(str(tmp_path), dm.MICROSCOPE_CONFIGS['Arctica-CETA'])


def saved(index, frames=1):
    return (f" {frames} summed frames were saved to "
            f"D:\\DEOutput\\Apollo\\sample_1\\20250101_{index}_0_movie.mrc")


def name(index):
    return f"20250101_{index}_0_movie.mrc"


FRAME_CHECK = ("STPSCN_FRAMES | square=1 | image=10001 | frames=1 | "
               "exposure=0.25 | action=none")


def write(tmp_path, lines):
    path = tmp_path / 'sample_1.log'
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


def test_stage_line_at_the_configured_offset(apollo, tmp_path):
    log = write(tmp_path, [
        saved(120833), 'Stage -169.25 99.21 66.77',
        saved(120834), 'Stage -164.25 99.21 66.77',
    ])
    assert apollo.get_coordinates_for_file(log, name(120833)) == [-169.25, 99.21, 66.77]
    assert apollo.get_coordinates_for_file(log, name(120834)) == [-164.25, 99.21, 66.77]


def test_ceta_offset_of_two_is_unchanged(ceta, tmp_path):
    log = write(tmp_path, [
        saved(1), 'Some line the CETA log puts here', 'Stage 10.00 20.00 30.00',
        saved(2), 'Some line the CETA log puts here', 'Stage 11.00 20.00 30.00',
    ])
    assert ceta.get_coordinates_for_file(log, name(1)) == [10.0, 20.0, 30.0]
    assert ceta.get_coordinates_for_file(log, name(2)) == [11.0, 20.0, 30.0]


def test_a_line_logged_before_the_stage_report_does_not_lose_the_frame(apollo, tmp_path):
    """A frame check that echoes between Record and the report."""
    log = write(tmp_path, [
        saved(120833), FRAME_CHECK, 'Stage -169.25 99.21 66.77',
        saved(120834), 'Stage -164.25 99.21 66.77',
    ])
    assert apollo.get_coordinates_for_file(log, name(120833)) == [-169.25, 99.21, 66.77]
    assert apollo.get_coordinates_for_file(log, name(120834)) == [-164.25, 99.21, 66.77]


def test_several_extra_lines_as_a_repair_writes_them(apollo, tmp_path):
    log = write(tmp_path, [
        saved(120833, frames=2),
        'Number of frames saved: 2',
        'STPSCN_FRAMES | square=1 | image=10001 | frames=2 | exposure=0.25 | '
        'action=repair_1 | next_exposure=0.25 | frame_time=0.25',
        'New frame time 0.245  exposure 0.245',
        'Stage -169.25 99.21 66.77',
    ])
    assert apollo.get_coordinates_for_file(log, name(120833)) == [-169.25, 99.21, 66.77]


def test_another_frames_position_is_never_taken(apollo, tmp_path):
    """No Stage line before the next frame's file line: no position, not the wrong one."""
    log = write(tmp_path, [
        saved(120833), FRAME_CHECK,
        saved(120834), FRAME_CHECK, 'Stage -164.25 99.21 66.77',
    ])
    assert apollo.get_coordinates_for_file(log, name(120833)) is None
    assert apollo.get_coordinates_for_file(log, name(120834)) == [-164.25, 99.21, 66.77]


def test_lines_that_only_start_with_stage_are_not_positions(apollo, tmp_path):
    """ReportStageBusy and other macro echoes also begin with the word."""
    log = write(tmp_path, [
        saved(120833), 'Stage is NOT busy', 'Stage at X=-169.2 Y=99.2 before pass',
        'Stage -169.25 99.21 66.77',
    ])
    assert apollo.get_coordinates_for_file(log, name(120833)) == [-169.25, 99.21, 66.77]


def test_the_search_is_bounded(apollo, tmp_path):
    log = write(tmp_path, [saved(120833)] + ['filler'] * dm.STAGE_LINE_LOOKAHEAD
                + ['Stage -169.25 99.21 66.77'])
    assert apollo.get_coordinates_for_file(log, name(120833)) is None


def test_a_frame_the_log_never_names_has_no_position(apollo, tmp_path):
    log = write(tmp_path, [saved(120833), 'Stage -169.25 99.21 66.77'])
    assert apollo.get_coordinates_for_file(log, name(999999)) is None
