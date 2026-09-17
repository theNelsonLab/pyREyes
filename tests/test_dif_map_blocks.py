"""
Block discovery and reyes.json in dif_map.

A block is identified from file names alone, so empty files are enough. The
cases cover blocks recorded across midnight
(the date in the name changes, the index keeps counting), a block acquired
twice into the same folder, and indices that are unpadded or outgrow their
padding.
"""

import json

import pytest

import pyREyes.dif_map as dm


@pytest.fixture
def handler(tmp_path):
    (tmp_path / 'REyes_logs').mkdir()   # created by setup_logging() in a real run
    return dm.FileHandler(str(tmp_path), dm.MICROSCOPE_CONFIGS['Arctica-CETA'])


def make_block(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).touch()
    return folder


def names(date, first, count, suffix='_integrated_movie.mrc', width=5):
    return [f"{date}_{str(i).zfill(width)}{suffix}" for i in range(first, first + count)]


def test_single_date_block(handler, tmp_path):
    folder = make_block(tmp_path / 'b_1', names('20251107', 79706, 121))

    indices = handler.determine_indices(folder)

    assert (indices.min_index, indices.max_index, indices.total_files) == (79706, 79826, 121)
    assert indices.files == names('20251107', 79706, 121)


def test_block_recorded_across_midnight_keeps_every_file(handler, tmp_path):
    files = names('20251107', 79827, 57) + names('20251108', 79884, 64)
    folder = make_block(tmp_path / 'b_36', files)

    indices = handler.determine_indices(folder)

    assert indices.total_files == 121
    assert indices.files == files
    assert indices.date_code == '20251107'


def test_repeated_acquisition_uses_the_latest(handler, tmp_path):
    first = names('20251222', 20858, 169, suffix='_0_movie.mrc')
    second = names('20251225', 45702, 169, suffix='_0_movie.mrc')
    folder = make_block(tmp_path / 'b_1', first + second)

    indices = handler.determine_indices(folder)

    assert indices.files == second


def test_missing_file_skips_the_block(handler, tmp_path):
    files = names('20250620', 454, 121)
    del files[60]
    folder = make_block(tmp_path / 'b_29', files)

    assert handler.determine_indices(folder) is None


def test_duplicate_index_skips_the_block(handler, tmp_path):
    files = names('20250620', 454, 121) + ['20250621_00500_integrated_movie.mrc']
    folder = make_block(tmp_path / 'b_29', files)

    assert handler.determine_indices(folder) is None


def test_unpadded_and_outgrown_indices(handler, tmp_path):
    files = [f"20260728_{i}_0_movie.mrc" for i in range(99995, 100004)]
    folder = make_block(tmp_path / 'b_2', files)

    indices = handler.determine_indices(folder)

    assert indices.files == files
    assert indices.index_width == 5


def test_other_files_are_ignored(handler, tmp_path):
    files = names('20251107', 1, 4)
    folder = make_block(tmp_path / 'b_3', files + ['20251107_00001_gain.mrc', 'notes.txt'])

    assert handler.determine_indices(folder).files == files


def test_reyes_json_keeps_version_1_fields_and_lists_files(handler, tmp_path):
    files = names('20251107', 333, 2) + names('20251108', 335, 2)
    folder = make_block(tmp_path / 'b_4', files)
    indices = handler.determine_indices(folder)

    handler.create_reyes_json(folder, indices)
    data = json.loads((folder / 'reyes.json').read_text())

    assert data['version'] == 1
    assert (data['min_id'], data['max_id']) == (333, 336)
    assert data['format'] == '20251107_%05d_integrated_movie.mrc'
    assert data['format'] % 333 == files[0]
    assert data['files'] == files
    assert isinstance(data['uid'], str)
