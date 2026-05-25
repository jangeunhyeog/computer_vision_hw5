from __future__ import annotations

import numpy as np


def clip_bbox_xywh(bbox: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]) -> np.ndarray:
    """Clip [x, y, w, h] to image bounds and keep non-negative size."""
    h_img, w_img = image_shape[:2]
    x, y, w, h = [float(v) for v in bbox[:4]]
    x1 = np.clip(x, 0, max(w_img - 1, 0))
    y1 = np.clip(y, 0, max(h_img - 1, 0))
    x2 = np.clip(x + max(w, 0), 0, w_img)
    y2 = np.clip(y + max(h, 0), 0, h_img)
    return np.asarray([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float64)


def bbox_to_int_slices(bbox: np.ndarray, image_shape: tuple[int, int] | tuple[int, int, int]):
    """Convert [x, y, w, h] to integer image slices.

    The slices are safe for array indexing. Empty boxes return zero-area slices.
    """
    x, y, w, h = clip_bbox_xywh(bbox, image_shape)
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = int(np.ceil(x + w))
    y2 = int(np.ceil(y + h))
    return slice(y1, y2), slice(x1, x2)


def state_to_bbox(state: np.ndarray, initial_size: tuple[float, float], aspect_ratio: float | None = None) -> np.ndarray:
    """Convert state [cx, cy, scale] to bbox [x, y, w, h].

    `initial_size` is (width, height). Scale 1.0 means the initial box size.
    """
    cx, cy, scale = [float(v) for v in state[:3]]
    scale = max(scale, 1e-6)
    w0, h0 = [float(v) for v in initial_size]
    w = w0 * scale
    h = h0 * scale
    if aspect_ratio is not None:
        # Keep the same area approximately while enforcing the given aspect ratio w / h.
        area = max(w * h, 1e-6)
        w = np.sqrt(area * aspect_ratio)
        h = w / aspect_ratio
    return np.asarray([cx - 0.5 * w, cy - 0.5 * h, w, h], dtype=np.float64)


def bbox_to_state(bbox: np.ndarray, initial_size: tuple[float, float]) -> np.ndarray:
    """Convert bbox [x, y, w, h] to state [cx, cy, scale]."""
    x, y, w, h = [float(v) for v in bbox[:4]]
    w0, h0 = [float(v) for v in initial_size]
    # Average width and height scale for stability.
    scale = 0.5 * (w / max(w0, 1e-6) + h / max(h0, 1e-6))
    return np.asarray([x + 0.5 * w, y + 0.5 * h, scale], dtype=np.float64)
