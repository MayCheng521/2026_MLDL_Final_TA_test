"""
make_submission.py — turn a folder of predicted masks into submission.csv

Each prediction is a PNG named after its test tile (t001.png ... t113.png), the
same size as the tile (384x384), with pixel values
    0 = wall, 1 = small, 2 = medium, 3 = large
(the same values as data/train/labels, which also use 255 for ignore; a prediction
must not contain 255).

submission.csv has 3 rows per tile, one per cell class:
    id,rle
    t001_1,<RLE of the small-cell pixels>
    t001_2,<RLE of the medium-cell pixels>
    t001_3,<RLE of the large-cell pixels>
A class with no pixel is written as "-". Pixels in none of the three masks
count as wall. See rle.py for the encoding.

Usage:
  python make_submission.py PRED_DIR                      # writes submission.csv
  python make_submission.py PRED_DIR -o my_sub.csv --test-dir data/test/images
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from rle import rle_encode

HERE = Path(__file__).resolve().parent


def load_pred(p: Path) -> np.ndarray:
    im = Image.open(p)
    a = np.array(im if im.mode in ("L", "P") else im.convert("L"))    # "P": keep palette indices
    if a.ndim != 2 or a.max() > 3:
        raise ValueError(f"{p.name}: pixel values must be 0..3 (one channel), got {np.unique(a)[:10]}")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("submission.csv"))
    default_test = next((d for d in (HERE / "data" / "test" / "images", HERE / "test" / "images") if d.is_dir()),
                        Path("test/images"))
    ap.add_argument("--test-dir", type=Path, default=default_test,
                    help="test images; every tile here must have a prediction")
    args = ap.parse_args()

    ids = sorted(p.stem for p in args.test_dir.glob("*.png"))
    if not ids:
        raise SystemExit(f"no test images in {args.test_dir}")
    rows = []                                   # check every tile before writing anything
    for tid in ids:
        pp = args.pred_dir / f"{tid}.png"
        if not pp.exists():
            raise SystemExit(f"missing prediction for {tid}: {pp}")
        try:
            pred = load_pred(pp)
        except ValueError as e:
            raise SystemExit(str(e))
        h, w = Image.open(args.test_dir / f"{tid}.png").size[::-1]
        if pred.shape != (h, w):
            raise SystemExit(f"{tid}: prediction {pred.shape} != tile {(h, w)}")
        rows += [(f"{tid}_{c}", rle_encode(pred == c)) for c in (1, 2, 3)]
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows([("id", "rle")] + rows)
    print(f"{len(ids)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
