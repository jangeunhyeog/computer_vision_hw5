from __future__ import annotations

import numpy as np


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute intersection-over-union for two [x, y, w, h] boxes."""
    ax, ay, aw, ah = [float(v) for v in box_a[:4]]
    bx, by, bw, bh = [float(v) for v in box_b[:4]]

    ax2, ay2 = ax + max(aw, 0.0), ay + max(ah, 0.0)
    bx2, by2 = bx + max(bw, 0.0), by + max(bh, 0.0)

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(aw, 0.0) * max(ah, 0.0)
    area_b = max(bw, 0.0) * max(bh, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compute_ious(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """Compute IoU for every frame."""
    n = min(len(pred_boxes), len(gt_boxes))
    return np.asarray([bbox_iou(pred_boxes[i], gt_boxes[i]) for i in range(n)], dtype=np.float64)
