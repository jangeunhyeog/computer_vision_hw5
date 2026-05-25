from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bbox(image_rgb: np.ndarray, bbox_xywh: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Draw a bounding box on an RGB image and return a copy."""
    out = image_rgb.copy()
    x, y, w, h = [int(round(v)) for v in bbox_xywh[:4]]
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), thickness)
    return out


def save_debug_frame(path: str | Path, image_rgb: np.ndarray, bbox_xywh: np.ndarray) -> None:
    """Save an RGB image with a predicted bounding box."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = draw_bbox(image_rgb, bbox_xywh)
    bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def plot_ious(ious: np.ndarray, out_dir: str | Path) -> None:
    """Save IoU-over-time and IoU histogram plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(np.arange(len(ious)), ious)
    plt.xlabel("Frame")
    plt.ylabel("IoU")
    plt.ylim(0.0, 1.0)
    plt.title("Tracking IoU over time")
    plt.tight_layout()
    plt.savefig(out_dir / "iou_over_time.png", dpi=150)
    plt.close()

    plt.figure()
    plt.hist(ious, bins=20, range=(0.0, 1.0))
    plt.xlabel("IoU")
    plt.ylabel("Number of frames")
    plt.title("IoU histogram")
    plt.tight_layout()
    plt.savefig(out_dir / "iou_histogram.png", dpi=150)
    plt.close()
