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
    # TODO(STUDENT): implement the HSV histogram described above.
    # Hint 1: crop the image using bbox_to_int_slices(bbox_xywh, image_rgb.shape).
    # Hint 2: use cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).
    # Hint 3: np.histogram2d and np.histogram are acceptable.
    raise NotImplementedError("compute_hsv_histogram is a student TODO")


def bhattacharyya_coefficient(hist_a: np.ndarray, hist_b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute the Bhattacharyya coefficient between two normalized histograms.

    The coefficient is sum_i sqrt(a_i * b_i), and should lie in [0, 1]
    for valid probability histograms.
    """
    # TODO(STUDENT): implement the Bhattacharyya coefficient.
    # Hint: normalize both inputs defensively before computing the coefficient.
    raise NotImplementedError("bhattacharyya_coefficient is a student TODO")


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
    # TODO(STUDENT): compute D^2 from the Bhattacharyya coefficient and return exp(-lambda_ * D^2).
    raise NotImplementedError("histogram_likelihood is a student TODO")
