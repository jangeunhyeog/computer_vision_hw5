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
    # Convert the initial bounding box to state [cx, cy, scale=1.0].
    initial_size = (float(init_bbox_xywh[2]), float(init_bbox_xywh[3]))
    init_state = bbox_to_state(init_bbox_xywh, initial_size)  # [cx, cy, 1.0]

    # Create particles with Gaussian noise around the initial state.
    particles = np.empty((num_particles, 3), dtype=np.float64)
    particles[:, 0] = init_state[0] + rng.normal(0.0, sigma_xy, size=num_particles)   # cx
    particles[:, 1] = init_state[1] + rng.normal(0.0, sigma_xy, size=num_particles)   # cy
    particles[:, 2] = init_state[2] + rng.normal(0.0, sigma_scale, size=num_particles) # scale
    # Ensure scale is positive.
    particles[:, 2] = np.maximum(particles[:, 2], 1e-6)
    return particles


def systematic_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample particles according to normalized weights using systematic resampling.

    After resampling, all particles have equal implicit weights.
    """
    n = len(weights)
    # Normalize weights defensively.
    w = normalize_weights(weights)
    # Build cumulative distribution.
    cumsum = np.cumsum(w)
    cumsum[-1] = 1.0  # ensure exact 1.0 at the end to avoid numerical issues.

    # Generate systematically spaced positions.
    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n, dtype=np.float64) / n

    # Find indices using searchsorted.
    indices = np.searchsorted(cumsum, positions)
    indices = np.clip(indices, 0, n - 1)

    return particles[indices].copy()


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
    w0, h0 = initial_size
    n = len(particles)
    new_particles = particles.copy()

    # Per-particle scale (s_{t-1}) for adaptive noise.
    s_prev = np.abs(particles[:, 2])  # ensure positive

    # Scale-adaptive noise standard deviations.
    std_x = sigma_xy * w0 * s_prev
    std_y = sigma_xy * h0 * s_prev
    std_s = sigma_scale * s_prev

    # Add zero-mean Gaussian noise.
    new_particles[:, 0] += rng.normal(0.0, 1.0, size=n) * std_x
    new_particles[:, 1] += rng.normal(0.0, 1.0, size=n) * std_y
    new_particles[:, 2] += rng.normal(0.0, 1.0, size=n) * std_s

    # Clip scale to valid range.
    new_particles[:, 2] = np.clip(new_particles[:, 2], min_scale, max_scale)

    return new_particles


def estimate_state(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Estimate target state from weighted particles.

    Use the weighted mean of particles as the posterior estimate.
    """
    w = normalize_weights(weights)
    return np.average(particles, axis=0, weights=w)


def particles_to_bboxes(particles: np.ndarray, initial_size: tuple[float, float]) -> np.ndarray:
    """Convert an array of particle states to bounding boxes."""
    return np.asarray([state_to_bbox(p, initial_size) for p in particles], dtype=np.float64)
