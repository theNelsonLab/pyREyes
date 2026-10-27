"""
MRC mode 0 (8-bit) signedness: one rule for every pyREyes frame reader.

MRC2014 calls mode 0 signed int8. IMOD, and the DE/SerialEM writers used with
DE detectors such as the Apollo, store UNSIGNED counts (0-255) in it and mark
signed bytes with the IMOD stamp plus flag bit 0. Reader libraries disagree:
mrcfile always hands back int8, and different rosettasciio versions return int8
or uint8. Read as int8, every pixel above 127 counts wraps to a negative number
(255 -> -1), which on a diffraction frame turns the brightest spot into a hole
ringed by false peaks.

Rule: a mode 0 file is unsigned unless its header carries the IMOD stamp with
the signed-bytes flag set. Detector counts are never negative.

`frames` and `read` open an MRC file through mrcfile and apply the rule, for
code that needs the data rather than a view of an array it already holds.
"""

import contextlib
from typing import Iterator, Optional

import numpy as np

__version__ = '3.6.1'

# IMOD header stamp and its "bytes are signed" flag (words 39 and 40, 1-based).
IMOD_STAMP = 1146047817
IMOD_FLAG_SIGNED_BYTES = 1


def mode0_signed(mrc_file: str) -> Optional[bool]:
    """
    Whether an MRC file's 8-bit pixels are signed, from its header alone.

    Returns None for a file that is not MRC mode 0 (or is too short to have a
    header); True only for mode 0 with the IMOD stamp and signed-bytes flag.
    """
    with open(mrc_file, 'rb') as handle:
        header = handle.read(224)
    if len(header) < 224:
        return None
    # Machine stamp 0x11 = big-endian; anything else is read as little-endian
    words = np.frombuffer(header, dtype='>i4' if header[212] == 0x11 else '<i4')
    mode, stamp, flags = int(words[3]), int(words[38]), int(words[39])
    if mode != 0:
        return None
    return stamp == IMOD_STAMP and bool(flags & IMOD_FLAG_SIGNED_BYTES)


def mode0_view(data: np.ndarray, mrc_file: str) -> np.ndarray:
    """
    ``data`` (read from ``mrc_file`` by any library) with the header's signedness.

    A VIEW, never a copy, so a memory-mapped stack stays memory-mapped. Data
    that is not 8-bit integer, or a file that is not mode 0, is returned as is.
    """
    if data is None or data.dtype.itemsize != 1 or data.dtype.kind not in 'iu':
        return data
    signed = mode0_signed(mrc_file)
    if signed is None:
        return data
    want = np.dtype(np.int8) if signed else np.dtype(np.uint8)
    return data if data.dtype == want else data.view(want)


@contextlib.contextmanager
def frames(mrc_file: str, mmap: bool = False) -> Iterator[np.ndarray]:
    """
    The data of an MRC file, mode 0 read by the header rule.

    ``mmap=True`` keeps the file memory-mapped (only the frames touched are
    read). The array is valid inside the ``with`` block only.
    """
    import mrcfile
    opener = mrcfile.mmap if mmap else mrcfile.open
    with opener(str(mrc_file), mode='r', permissive=True) as handle:
        yield mode0_view(handle.data, str(mrc_file))


def read(mrc_file: str) -> np.ndarray:
    """The whole MRC file as an in-memory array (a copy), mode 0 by the header rule."""
    with frames(mrc_file) as data:
        if data is None:
            raise ValueError(f"No image data in {mrc_file}")
        return data.copy()
