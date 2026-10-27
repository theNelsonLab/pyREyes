"""The MRC mode 0 header rule (lib/mrc_io.py).

DE detector frames are MRC mode 0 with no IMOD stamp and hold UNSIGNED counts.
mrcfile hands them back as int8, so read naively every pixel above 127 counts
wraps negative (255 -> -1). A mode 0 file is unsigned unless its header carries
the IMOD stamp with the signed-bytes flag set.
"""

import os
import struct

import numpy as np
import pytest

mrcfile = pytest.importorskip('mrcfile')

from pyREyes.lib import mrc_io
import pyREyes.dif_map as dm


def write_mode0(path, counts, imod_signed=None):
    """A mode-0 MRC holding `counts` (0..255) the way SerialEM/DE write it."""
    raw = np.asarray(counts, dtype=np.uint8)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(raw.view(np.int8))
    if imod_signed is not None:
        with open(path, 'r+b') as f:
            f.seek(152)
            f.write(struct.pack('<ii', mrc_io.IMOD_STAMP, 1 if imod_signed else 0))
    return str(path)


def bright_counts(shape=(64, 64)):
    counts = np.full(shape, 20, dtype=np.uint8)
    counts[..., 10:14, 10:14] = 255
    counts[..., 30:32, 40:42] = 171
    return counts


def library_read(path):
    with mrcfile.open(path, permissive=True) as handle:
        return np.array(handle.data)


def test_unstamped_mode0_reads_unsigned(tmp_path):
    counts = bright_counts()
    path = write_mode0(tmp_path / 'f.mrc', counts)
    assert mrc_io.mode0_signed(path) is False
    raw = library_read(path)
    assert raw.dtype == np.int8 and raw.min() < 0        # the library says int8
    data = mrc_io.mode0_view(raw, path)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, counts)


def test_imod_signed_mode0_stays_int8(tmp_path):
    counts = bright_counts()
    path = write_mode0(tmp_path / 'f.mrc', counts, imod_signed=True)
    assert mrc_io.mode0_signed(path) is True
    data = mrc_io.mode0_view(counts.copy(), path)
    assert data.dtype == np.int8
    np.testing.assert_array_equal(data, counts.view(np.int8))


def test_imod_stamp_without_the_signed_flag_is_unsigned(tmp_path):
    path = write_mode0(tmp_path / 'f.mrc', bright_counts(), imod_signed=False)
    assert mrc_io.mode0_signed(path) is False
    assert mrc_io.mode0_view(library_read(path), path).dtype == np.uint8


@pytest.mark.parametrize('dtype', [np.int16, np.uint16, np.float32])
def test_other_modes_are_untouched(tmp_path, dtype):
    values = (np.arange(64 * 64).reshape(64, 64) % 300).astype(dtype)
    path = str(tmp_path / 'f.mrc')
    with mrcfile.new(path) as mrc:
        mrc.set_data(values)
    assert mrc_io.mode0_signed(path) is None
    data = library_read(path)
    assert mrc_io.mode0_view(data, path) is data


def test_short_file_has_no_mode(tmp_path):
    path = tmp_path / 'short.mrc'
    path.write_bytes(b'\0' * 100)
    assert mrc_io.mode0_signed(str(path)) is None


def test_mode0_view_of_a_library_array_never_copies(tmp_path):
    counts = bright_counts()
    path = write_mode0(tmp_path / 'f.mrc', counts)
    as_int8 = counts.view(np.int8)
    out = mrc_io.mode0_view(as_int8, path)
    assert out.dtype == np.uint8 and np.shares_memory(out, as_int8)
    already = counts.copy()
    assert mrc_io.mode0_view(already, path) is already


def test_a_memory_mapped_stack_stays_mapped(tmp_path):
    counts = bright_counts((5, 64, 64))
    path = write_mode0(tmp_path / 'movie.mrc', counts)
    with mrcfile.mmap(path, mode='r', permissive=True) as handle:
        data = mrc_io.mode0_view(handle.data, path)
        assert isinstance(data, np.memmap)
        assert data.dtype == np.uint8 and not data.flags.owndata
        np.testing.assert_array_equal(data[3], counts[3])


def test_dif_map_shares_the_rule(tmp_path):
    assert dm.mrc_mode0_signed is mrc_io.mode0_signed
    assert dm.IMOD_STAMP == mrc_io.IMOD_STAMP
    counts = bright_counts((256, 256))
    path = write_mode0(tmp_path / 'f.mrc', counts)
    data, _ = dm.load_diffraction_frame(path)
    assert data.min() >= 0 and data.max() == 255


# --------------------------------------------------------------------------- #
# The readers: frames() and read() apply the rule to a file
# --------------------------------------------------------------------------- #
def test_read_gives_unsigned_counts_for_an_unstamped_mode0_file(tmp_path):
    counts = bright_counts()
    path = write_mode0(tmp_path / 'f.mrc', counts)
    data = mrc_io.read(path)
    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, counts)
    assert data.flags.owndata                              # a copy, usable after


def test_read_keeps_imod_signed_mode0_as_int8(tmp_path):
    counts = bright_counts()
    path = write_mode0(tmp_path / 'f.mrc', counts, imod_signed=True)
    data = mrc_io.read(path)
    assert data.dtype == np.int8
    np.testing.assert_array_equal(data, counts.view(np.int8))


def test_read_without_the_signed_flag_is_unsigned(tmp_path):
    path = write_mode0(tmp_path / 'f.mrc', bright_counts(), imod_signed=False)
    assert mrc_io.read(path).dtype == np.uint8


@pytest.mark.parametrize('dtype', [np.int16, np.uint16, np.float32])
def test_read_and_frames_leave_other_modes_untouched(tmp_path, dtype):
    values = (np.arange(64 * 64).reshape(64, 64) % 300).astype(dtype)
    path = str(tmp_path / 'f.mrc')
    with mrcfile.new(path) as mrc:
        mrc.set_data(values)
    data = mrc_io.read(path)
    assert data.dtype == dtype
    np.testing.assert_array_equal(data, values)
    with mrc_io.frames(path, mmap=True) as mapped:
        assert mapped.dtype == dtype


def test_frames_mmap_is_a_view_not_a_copy(tmp_path):
    counts = bright_counts((5, 64, 64))
    path = write_mode0(tmp_path / 'movie.mrc', counts)
    with mrc_io.frames(path, mmap=True) as data:
        assert isinstance(data, np.memmap)
        assert data.dtype == np.uint8
        assert not data.flags.owndata
        np.testing.assert_array_equal(data[3], counts[3])
