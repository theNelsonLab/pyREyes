"""dif-map-2 gives the same numbers on every machine.

The same MRC mode 0 frames used to give different Sums on different machines,
always by multiples of 256: some rosettasciio versions read mode 0 as int8,
others as uint8, so on some machines every pixel above 127 lost 256, and spot
counts, quality labels and targets followed. Mode 0 is now read from the
header, and everything summed or thresholded is float64.
"""

import struct

import mrcfile
import numpy as np
import pytest

import pyREyes.dif_map as dm

from test_dif_map_frames import spot_frame


def write_mode0(path, counts, imod_signed=None):
    """A 2048x2048 mode-0 MRC holding `counts` (0..255) as SerialEM/DE write it."""
    raw = np.asarray(counts, dtype=np.uint8)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(raw.view(np.int8))
    if imod_signed is not None:
        with open(path, 'r+b') as f:
            f.seek(152)
            f.write(struct.pack('<ii', dm.IMOD_STAMP, 1 if imod_signed else 0))
    return path


def frame_with_bright_pixels():
    frame = np.clip(spot_frame(amplitude=200.0, background=20.0), 0, 255)
    frame[1000:1004, 900:904] = 171          # behind-the-stop halo, > 127
    return frame.astype(np.uint8)


def test_mode0_is_unsigned_without_the_imod_flag(tmp_path):
    counts = frame_with_bright_pixels()
    path = write_mode0(tmp_path / 'f.mrc', counts)
    assert dm.mrc_mode0_signed(str(path)) is False
    data, _ = dm.load_diffraction_frame(str(path))
    assert data.max() == counts.max() and data.min() >= 0
    assert dm.frame_sum(data) == float(counts.astype(np.int64).sum())


def test_the_imod_signed_flag_is_honoured(tmp_path):
    counts = frame_with_bright_pixels()
    path = write_mode0(tmp_path / 'f.mrc', counts, imod_signed=True)
    assert dm.mrc_mode0_signed(str(path)) is True
    data, _ = dm.load_diffraction_frame(str(path))
    assert data.min() < 0
    path = write_mode0(tmp_path / 'g.mrc', counts, imod_signed=False)
    assert dm.mrc_mode0_signed(str(path)) is False


def test_the_reader_does_not_decide_the_sign(tmp_path, monkeypatch):
    """Whatever int8/uint8 a reader returns, the frame is the same."""
    counts = frame_with_bright_pixels()
    path = str(write_mode0(tmp_path / 'f.mrc', counts))
    sums = set()
    for dtype in (np.int8, np.uint8):
        class Fake:
            data = counts.view(dtype)
        monkeypatch.setattr(dm.hs, 'load', lambda p, Fake=Fake: Fake)
        sums.add(dm.frame_sum(dm.load_diffraction_frame(path)[0]))
    assert len(sums) == 1


def test_non_mode0_files_are_left_alone(tmp_path):
    path = tmp_path / 'f.mrc'
    with mrcfile.new(str(path)) as mrc:
        mrc.set_data(np.zeros((64, 64), dtype=np.int16))
    assert dm.mrc_mode0_signed(str(path)) is None


def test_frame_sum_is_exact_and_typed():
    ints = np.full((2048, 2048), 127, dtype=np.int16)
    assert dm.frame_sum(ints) == 127 * 2048 * 2048 and isinstance(dm.frame_sum(ints), int)
    floats = np.full((2048, 2048), 171.0, dtype=np.float32)
    assert dm.frame_sum(floats) == 171.0 * 2048 * 2048
    assert isinstance(dm.frame_sum(floats), float)


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.uint8, np.int16])
def test_detection_does_not_depend_on_the_frame_dtype(dtype):
    """Thresholds are computed in float64 whatever the frame type, so the
    spot count is a property of the counts, not of float32 rounding."""
    config = dm.MICROSCOPE_CONFIGS['Arctica-Apollo-SM']
    counts = frame_with_bright_pixels()
    processor = dm.ImageProcessor(config)
    ref = processor.create_binary_image(counts.astype(np.float64))
    got = processor.create_binary_image(counts.astype(dtype))
    assert np.array_equal(ref, got)


def test_binning_accumulates_in_float64():
    frame = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1e7)
    binned, factor = dm.bin_to_target(frame, target=2)
    expected = frame.astype(np.float64).reshape(2, 2, 2, 2).mean(axis=(1, 3))
    assert factor == 2 and binned.dtype == np.float32
    assert np.array_equal(binned, expected.astype(np.float32))
