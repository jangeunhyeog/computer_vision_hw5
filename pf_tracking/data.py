from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

BBox = np.ndarray  # [x, y, w, h], float32/float64


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def _frame_dir(sequence_dir: Path) -> Path:
    sequence_dir = Path(sequence_dir)
    img_dir = sequence_dir / "img"
    return img_dir if img_dir.exists() else sequence_dir


def list_frames(sequence_dir: str | Path) -> List[Path]:
    """Return sorted image paths for a sequence.

    Expected layout: sequence/img/0001.jpg, ...
    If sequence/img does not exist, images are read directly from sequence/.
    """
    root = _frame_dir(Path(sequence_dir))
    frames: List[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        frames.extend(root.glob(pattern))
    frames = sorted(frames)
    if not frames:
        raise FileNotFoundError(f"No image frames found under {root}")
    return frames


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_ground_truth(sequence_dir: str | Path) -> np.ndarray:
    """Load OTB-style ground truth boxes as an array of shape (T, 4).

    Accepts comma-separated or whitespace-separated rows:
        x,y,w,h
        x y w h
    """
    sequence_dir = Path(sequence_dir)
    candidates = [
        sequence_dir / "groundtruth.txt",
        sequence_dir / "groundtruth_rect.txt",
        sequence_dir / "gt.txt",
    ]
    gt_path = next((p for p in candidates if p.exists()), None)
    if gt_path is None:
        raise FileNotFoundError(
            f"Could not find a ground-truth file in {sequence_dir}. "
            "Expected groundtruth.txt, groundtruth_rect.txt, or gt.txt."
        )

    boxes = []
    for line_no, line in enumerate(gt_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        line = line.replace(",", " ")
        vals = [float(v) for v in line.split()]
        if len(vals) < 4:
            raise ValueError(f"Invalid GT line {line_no} in {gt_path}: {line!r}")
        boxes.append(vals[:4])
    if not boxes:
        raise ValueError(f"Ground-truth file is empty: {gt_path}")
    return np.asarray(boxes, dtype=np.float64)


def save_bboxes(path: str | Path, boxes: np.ndarray) -> None:
    """Save boxes as x,y,w,h with six decimal places."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for box in np.asarray(boxes):
            f.write("{:.6f},{:.6f},{:.6f},{:.6f}\n".format(*box[:4]))


def load_bboxes(path: str | Path) -> np.ndarray:
    """Load a predicted bbox file in x,y,w,h format."""
    path = Path(path)
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) < 4:
            raise ValueError(f"Invalid bbox line {line_no}: {line!r}")
        boxes.append(vals[:4])
    return np.asarray(boxes, dtype=np.float64)


def load_sequence(sequence_dir: str | Path) -> Tuple[List[Path], np.ndarray]:
    """Load frame paths and ground truth boxes."""
    frames = list_frames(sequence_dir)
    gt = load_ground_truth(sequence_dir)
    if len(gt) < len(frames):
        raise ValueError(f"Ground truth has {len(gt)} boxes but sequence has {len(frames)} frames")
    return frames, gt[: len(frames)]
