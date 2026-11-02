"""
evaluate.py — score test predictions against the answers (TA only)

Score per tile = mean IoU over the cell classes 1..3 (small / medium / large)
that appear in the GT or in the prediction of that tile; a class absent from both
is skipped, one present in only one of them has IoU 0. Wall (class 0) is not
scored. GT pixels labelled ignore (255) are removed before scoring, whatever
the prediction is there. A tile where no class appears at all scores 1.
Final score = mean over tiles, reported for public, private and all tiles.

This is the same score as the Kaggle metric.

Predictions: a folder of PNGs (t001.png ..., values 0..3) or a
submission.csv (see make_submission.py).

Usage:
  python evaluate.py PRED_DIR                  # public / private / all
  python evaluate.py submission.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from make_submission import load_pred
from rle import rle_decode

HERE = Path(__file__).resolve().parent
IGNORE = 255
CLASS_NAMES = ["wall", "small", "medium", "large"]


def load_gt(p: Path) -> np.ndarray:
    return np.array(Image.open(p))           # 0..3, ignore = 255


def tile_score(pred: np.ndarray, gt: np.ndarray) -> float:
    valid = gt != IGNORE
    pred, gt = pred[valid], gt[valid]
    ious = []
    for c in (1, 2, 3):
        p, g = pred == c, gt == c
        union = np.logical_or(p, g).sum()
        if union:
            ious.append(np.logical_and(p, g).sum() / union)
    return float(np.mean(ious)) if ious else 1.0


def read_submission(path: Path, shape: tuple) -> dict:
    """submission.csv -> {tile id: label map}; pixels in no class are wall."""
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["id"] in rows:
            raise ValueError(f"duplicate id {r['id']}")
        rows[r["id"]] = r["rle"]
    preds = {}
    for rid, rle in rows.items():
        tid, c = rid.rsplit("_", 1)
        if c not in ("1", "2", "3"):
            raise ValueError(f"bad id {rid}: must end in _1, _2 or _3")
        lab = preds.setdefault(tid, np.zeros(shape, np.uint8))
        m = rle_decode(rle, shape)
        if (lab[m] != 0).any():
            raise ValueError(f"{tid}: a pixel is assigned to more than one class")
        lab[m] = int(c)
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred", type=Path, help="folder of PNG predictions or a submission.csv")
    ap.add_argument("--answers", type=Path, default=HERE / "answers")
    args = ap.parse_args()

    split = {r["id"]: r["split"] for r in csv.DictReader(open(args.answers / "test_split.csv", encoding="utf-8"))}
    gts = {tid: load_gt(args.answers / "labels" / f"{tid}.png") for tid in split}
    shape = next(iter(gts.values())).shape
    preds = read_submission(args.pred, shape) if args.pred.suffix.lower() == ".csv" else None

    scores = {"public": [], "private": []}
    inter, union = np.zeros(4), np.zeros(4)
    for tid, gt in sorted(gts.items()):
        if preds is not None:
            if tid not in preds:
                raise SystemExit(f"missing prediction for {tid}")
            pred = preds[tid]
        else:
            pp = args.pred / f"{tid}.png"
            if not pp.exists():
                raise SystemExit(f"missing prediction for {tid}: {pp}")
            pred = load_pred(pp)
        if pred.shape != gt.shape:
            raise SystemExit(f"{tid}: prediction {pred.shape} != GT {gt.shape}")
        scores[split[tid]].append(tile_score(pred, gt))
        valid = gt != IGNORE
        p, g = pred[valid], gt[valid]
        for c in range(4):
            inter[c] += np.logical_and(p == c, g == c).sum()
            union[c] += np.logical_or(p == c, g == c).sum()

    for k, v in scores.items():
        print(f"{k:8s}{np.mean(v):.4f}  ({len(v)} tiles)")
    print(f"{'all':8s}{np.mean(scores['public'] + scores['private']):.4f}")
    print("pooled IoU per class: " + "  ".join(
        f"{CLASS_NAMES[c]}={inter[c] / union[c]:.3f}" for c in range(4) if union[c]))


if __name__ == "__main__":
    main()
