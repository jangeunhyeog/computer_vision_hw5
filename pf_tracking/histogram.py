from __future__ import annotations

import cv2
import numpy as np

from .geometry import bbox_to_int_slices


def compute_hsv_histogram(
    image_rgb: np.ndarray,
    bbox_xywh: np.ndarray,
    hs_bins: tuple[int, int] = (10, 10),
    v_bins: int = 10,
    min_h: float = 0.1,
    min_s: float = 0.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the assignment HSV histogram inside a bounding box.

    Expected behavior:
    - Convert RGB to HSV using OpenCV.
    - OpenCV uses H in [0, 179], S in [0, 255], V in [0, 255]. Normalize each channel to [0, 1].
    - Populate a 2D H-S histogram for pixels satisfying H >= min_h and S >= min_s.
    - Populate an additional 1D V histogram for the remaining pixels.
    - Concatenate the flattened HS histogram and the V histogram.
    - Normalize the final vector so that its sum is 1.

    Returns:
        Array of length hs_bins[0] * hs_bins[1] + v_bins.
    """
    # Crop the image region defined by the bounding box.
    row_slice, col_slice = bbox_to_int_slices(bbox_xywh, image_rgb.shape)
    crop = image_rgb[row_slice, col_slice]

    # Handle empty crop.
    total_bins = hs_bins[0] * hs_bins[1] + v_bins
    if crop.size == 0:
        return np.full(total_bins, 1.0 / total_bins, dtype=np.float64)

    # Convert RGB to HSV using OpenCV.
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Normalize OpenCV HSV channels to [0, 1]:
    #   H: [0, 179] -> [0, 1], S: [0, 255] -> [0, 1], V: [0, 255] -> [0, 1]
    h_ch = hsv[:, :, 0].astype(np.float64) / 179.0
    s_ch = hsv[:, :, 1].astype(np.float64) / 255.0
    v_ch = hsv[:, :, 2].astype(np.float64) / 255.0

    # Flatten channels for pixel-level filtering.
    h_flat = h_ch.ravel()
    s_flat = s_ch.ravel()
    v_flat = v_ch.ravel()

    # Mask: pixels satisfying H >= min_h AND S >= min_s go to HS histogram.
    hs_mask = (h_flat >= min_h) & (s_flat >= min_s)
    v_mask = ~hs_mask  # remaining pixels go to V histogram.

    # Build 2D H-S histogram.
    hs_hist, _, _ = np.histogram2d(
        h_flat[hs_mask], s_flat[hs_mask],
        bins=hs_bins, range=[[0.0, 1.0], [0.0, 1.0]],
    )

    # Build 1D V histogram for the remaining pixels.
    v_hist, _ = np.histogram(
        v_flat[v_mask], bins=v_bins, range=(0.0, 1.0),
    )

    # Concatenate and normalize.
    hist = np.concatenate([hs_hist.ravel(), v_hist]).astype(np.float64)
    total = hist.sum()
    if total < eps:
        return np.full(total_bins, 1.0 / total_bins, dtype=np.float64)
    return hist / total


def bhattacharyya_coefficient(hist_a: np.ndarray, hist_b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Bhattacharyya coefficient between two normalized histograms.

    The coefficient is sum_i sqrt(a_i * b_i), and should lie in [0, 1]
    for valid probability histograms.
    """
    # Defensively normalize both histograms.
    a = np.asarray(hist_a, dtype=np.float64)
    b = np.asarray(hist_b, dtype=np.float64)
    a_sum = a.sum()
    b_sum = b.sum()
    if a_sum > eps:
        a = a / a_sum
    if b_sum > eps:
        b = b / b_sum
    return float(np.sum(np.sqrt(a * b)))


def histogram_likelihood(
    candidate_hist: np.ndarray,
    target_hist: np.ndarray,
    lambda_: float = 20.0,
) -> float:
    """Convert histogram similarity into a particle likelihood.

    Use the squared Bhattacharyya distance:
        D^2 = 1 - sum_i sqrt(target_i * candidate_i)
        likelihood = exp(-lambda_ * D^2)
    """
    bc = bhattacharyya_coefficient(candidate_hist, target_hist)
    d_sq = 1.0 - bc
    return float(np.exp(-lambda_ * d_sq))
