"""
Frame loading and detection parameters in dif_map.

MRC reading is replaced by an in-memory array, so every case is a synthetic
frame whose correct answer is known: a fixed pattern of Gaussian spots that the
detector finds at 2048x2048. The larger, stacked and differently typed versions
of that frame must give the same answer.
"""

import numpy as np
import pytest

import pyREyes.dif_map as dm


SPOT_ROWS = (400, 700, 1300, 1600)
SPOT_COLS = (350, 650, 1400, 1700)


def spot_frame(size=2048, amplitude=400.0, background=20.0, seed=0):
    """A float frame with a 4x4 lattice of spots away from the excluded centre."""
    rng = np.random.default_rng(seed)
    frame = rng.poisson(background, (size, size)).astype(np.float64)
    scale = size / 2048
    yy, xx = np.mgrid[0:size, 0:size]
    for r in SPOT_ROWS:
        for c in SPOT_COLS:
            frame += amplitude * np.exp(-((yy - r * scale) ** 2 + (xx - c * scale) ** 2)
                                        / (2 * (2.0 * scale) ** 2))
    return frame


def upsample(frame, factor):
    """Replicate pixels so that mean binning by `factor` restores `frame` exactly."""
    return np.kron(frame, np.ones((factor, factor)))


@pytest.fixture
def config():
    return dm.MICROSCOPE_CONFIGS['Arctica-CETA']


@pytest.fixture
def fake_mrc(monkeypatch):
    """Serve arrays from a dict instead of reading MRC files."""
    arrays = {}
    monkeypatch.setattr(dm, '_read_mrc', lambda path: arrays[path])
    return arrays


def count_peaks(config, data):
    processor = dm.ImageProcessor(config)
    binary = processor.create_binary_image(data)
    return processor.get_centroids(binary, exclude_center=25)[1]


@pytest.fixture
def reference(config):
    return count_peaks(config, spot_frame().astype(np.float32))


def test_reference_frame_finds_every_spot(reference):
    assert reference == len(SPOT_ROWS) * len(SPOT_COLS)


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16, np.int32, np.float16, np.float32, np.float64])
def test_oversized_frames_of_any_type_bin_to_the_reference(config, fake_mrc, reference, dtype):
    frame = spot_frame(amplitude=100.0, background=5.0)
    fake_mrc['f'] = upsample(frame, 2).astype(dtype)

    data, info = dm.load_diffraction_frame('f')

    assert data.shape == (2048, 2048)
    assert data.dtype == np.float32
    assert info['binning'] == 2
    assert count_peaks(config, data) == reference


@pytest.mark.parametrize('size, factor', [(4096, 2), (6144, 3), (8192, 4)])
def test_binning_uses_one_integer_factor(fake_mrc, size, factor):
    fake_mrc['f'] = upsample(spot_frame(), factor).astype(np.float32)

    data, info = dm.load_diffraction_frame('f')

    assert info['binning'] == factor
    np.testing.assert_allclose(data, spot_frame(), rtol=1e-5)


def test_int16_overflow_is_avoided_when_binning(fake_mrc):
    fake_mrc['f'] = np.full((4096, 4096), 30000, dtype=np.int16)

    data, _ = dm.load_diffraction_frame('f')

    assert data.min() == data.max() == 30000


def test_int16_frames_at_target_size_are_left_unchanged(fake_mrc):
    frame = spot_frame().astype(np.int16)
    fake_mrc['f'] = frame

    data, info = dm.load_diffraction_frame('f')

    assert data.dtype == np.int16
    assert info['binning'] == 1
    np.testing.assert_array_equal(data, frame)


@pytest.mark.parametrize('frames', [1, 2, 3])
@pytest.mark.parametrize('size', [2048, 4096, 8192])
def test_stacks_are_summed(config, fake_mrc, reference, frames, size):
    single = upsample(spot_frame(), size // 2048).astype(np.float32)
    fake_mrc['f'] = np.stack([single] * frames)

    data, info = dm.load_diffraction_frame('f')

    assert data.shape == (2048, 2048)
    assert info['frames'] == (frames if frames > 1 else 1)
    np.testing.assert_allclose(data, spot_frame() * frames, rtol=1e-5)
    assert count_peaks(config, data) == reference


def test_stack_with_frames_on_the_last_axis(fake_mrc):
    fake_mrc['f'] = np.stack([spot_frame(size=1024)] * 2, axis=-1).astype(np.float32)[:, :1024, :]
    data, info = dm.load_diffraction_frame('f')
    assert info['frames'] == 2
    assert data.shape == (1024, 1024)


def test_non_finite_pixels_are_zeroed(fake_mrc):
    frame = spot_frame().astype(np.float32)
    frame[10, 10] = np.nan
    frame[20, 20] = np.inf
    fake_mrc['f'] = frame

    data, info = dm.load_diffraction_frame('f')

    assert info['non_finite'] == 2
    assert np.isfinite(data).all()


def test_unsupported_types_and_shapes_raise(fake_mrc):
    fake_mrc['complex'] = np.zeros((64, 64), dtype=np.complex64)
    fake_mrc['4d'] = np.zeros((2, 2, 64, 64), dtype=np.float32)
    with pytest.raises(ValueError):
        dm.load_diffraction_frame('complex')
    with pytest.raises(ValueError):
        dm.load_diffraction_frame('4d')


def test_bad_column_is_removed_before_binning(config, fake_mrc, reference):
    frame = upsample(spot_frame(amplitude=100.0, background=5.0), 4)
    frame[4000:, 7912:7914] += 60.0   # hot pair of columns over half the frame
    fake_mrc['f'] = frame.astype(np.float32)

    data, info = dm.load_diffraction_frame('f')

    assert info['bad_lines'] == {'columns': [(7912, 7913)]}
    assert count_peaks(config, data) == reference


def test_spots_and_wide_structures_are_not_bad_lines(fake_mrc):
    frame = upsample(spot_frame(amplitude=5000.0), 2)
    frame[3000:3200, :] *= 0.2        # a wide dark band, e.g. a beam stop arm
    fake_mrc['f'] = frame.astype(np.float32)

    _, info = dm.load_diffraction_frame('f')

    assert info['bad_lines'] == {}


def test_bad_lines_are_not_removed_at_target_size(fake_mrc):
    frame = spot_frame()
    frame[:, 1000] += 500.0
    fake_mrc['f'] = frame.astype(np.float32)

    data, info = dm.load_diffraction_frame('f')

    assert info['bad_lines'] == {}
    assert data[5, 1000] == np.float32(frame[5, 1000])


def test_small_frames_use_small_frame_parameters(config):
    processor = dm.ImageProcessor(config)
    assert processor.detection_params((2048, 2048))['min_pixels_prim'] == config.min_pixels_prim
    assert processor.detection_params((1024, 1024)) == dm.SMALL_FRAME_PARAMS[1024]
    assert processor.detection_params((512, 512)) == dm.SMALL_FRAME_PARAMS[512]
    assert processor.detection_params((256, 256)) == dm.SMALL_FRAME_PARAMS[512]


def test_small_frames_find_spots(config):
    frame = spot_frame()
    small = frame.reshape(512, 4, 512, 4).mean(axis=(1, 3)).astype(np.float32)
    assert count_peaks(config, small) == len(SPOT_ROWS) * len(SPOT_COLS)


def test_legacy_parse_keeps_configuration_parameters(config):
    processor = dm.ImageProcessor(config, legacy_parse=True)
    assert processor.detection_params((512, 512))['min_pixels_prim'] == config.min_pixels_prim


def test_border_mask_on_tiny_frames(config):
    processor = dm.ImageProcessor(config)
    binary = processor.create_binary_image(spot_frame(size=32).astype(np.float32))
    assert binary.shape == (32, 32)
    assert not binary[:, 0].any() and not binary[:, -1].any()
