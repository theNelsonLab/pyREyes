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
"""

from typing import Optional

import numpy as np

__version__ = '3.6.0'

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
