from __future__ import annotations

import numpy as np

from .geometry import bbox_to_state, state_to_bbox


def normalize_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize nonnegative weights; fall back to a uniform distribution if needed."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= eps:
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
    return weights / total


def initialize_particles(
    init_bbox_xywh: np.ndarray,
    num_particles: int,
    rng: np.random.Generator,
    sigma_xy: float = 5.0,
    sigma_scale: float = 0.02,
) -> np.ndarray:
    """Initialize particles around the first-frame bounding box.

    State format is [cx, cy, scale]. Scale 1.0 corresponds to the first-frame box size.
    """
    # TODO(STUDENT): initialize particles from a Gaussian distribution around the initial state.
    # The center coordinates should use sigma_xy; the scale should use sigma_scale.
    # Return an array of shape (num_particles, 3).
    raise NotImplementedError("initialize_particles is a student TODO")


def systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample particles according to normalized weights using systematic resampling.

    After resampling, all particles have equal implicit weights.
    """
    # TODO(STUDENT): implement systematic resampling.
    # Hint: build cumulative_sum = np.cumsum(weights), then use regularly spaced positions.
    raise NotImplementedError("systematic_resample is a student TODO")


def transition(
    particles: np.ndarray,
    rng: np.random.Generator,
    initial_size: tuple[float, float],
    sigma_xy: float = 0.15,
    sigma_scale: float = 0.05,
    min_scale: float = 0.2,
    max_scale: float = 4.0,
) -> np.ndarray:
    """Apply scale-adaptive random-walk dynamics to particles.

    `initial_size` is the first-frame bounding box size (w0, h0). Per the assignment
    equation, the per-particle noise standard deviations are scale-adaptive:

        std_x = sigma_xy * w0 * s_{t-1}
        std_y = sigma_xy * h0 * s_{t-1}      (= sigma_xy * w0 * s * gamma, gamma = h0 / w0)
        std_s = sigma_scale * s_{t-1}

    Add zero-mean Gaussian noise with these std values to each particle's
    (cx, cy, s). Clip scale to [min_scale, max_scale] after the update.
    """
    # TODO(STUDENT): implement scale-adaptive random-walk prediction.
    raise NotImplementedError("transition is a student TODO")


def estimate_state(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Estimate target state from weighted particles.

    Use the weighted mean of particles as the posterior estimate.
    """
    # TODO(STUDENT): return the weighted average state.
    raise NotImplementedError("estimate_state is a student TODO")


def particles_to_bboxes(particles: np.ndarray, initial_size: tuple[float, float]) -> np.ndarray:
    """Convert an array of particle states to bounding boxes."""
    return np.asarray([state_to_bbox(p, initial_size) for p in particles], dtype=np.float64)
