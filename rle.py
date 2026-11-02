"""
rle.py — run-length encoding used in submission.csv

Pixels of a 384x384 tile are numbered row by row (left to right, then top to
bottom), starting at 1. A binary mask is written as "start length start length ...";
a mask with no pixels is written as "-" (Kaggle rejects empty cells).

    >>> m = np.zeros((2, 3), bool); m[0, 1:] = True; m[1, 0] = True
    >>> rle_encode(m)
    '2 3'
    >>> rle_decode('2 3', (2, 3)).astype(int).tolist()
    [[0, 1, 1], [1, 0, 0]]
    >>> rle_encode(np.zeros((2, 3), bool))
    '-'
"""

import numpy as np

EMPTY = "-"


def rle_encode(mask: np.ndarray) -> str:
    m = np.concatenate([[0], np.asarray(mask, bool).ravel().astype(np.int8), [0]])
    runs = np.flatnonzero(m[1:] != m[:-1]) + 1     # 1-based start / end+1 of each run
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs)) or EMPTY


def rle_decode(rle, shape: tuple) -> np.ndarray:
    n = shape[0] * shape[1]
    m = np.zeros(n, bool)
    if not isinstance(rle, str) or rle.strip() in ("", EMPTY):     # "-", or empty / NaN
        return m.reshape(shape)
    try:
        s = np.array(rle.split(), dtype=np.int64)
    except ValueError:
        raise ValueError(f"RLE must contain integers only: {rle[:50]!r}")
    if len(s) % 2:
        raise ValueError("RLE must have an even number of values (start length pairs)")
    starts, lengths = s[0::2] - 1, s[1::2]
    if (starts < 0).any() or (lengths < 1).any() or (starts + lengths > n).any():
        raise ValueError(f"RLE run outside the image (pixels are numbered 1..{n})")
    for a, k in zip(starts, lengths):
        m[a:a + k] = True
    return m.reshape(shape)
