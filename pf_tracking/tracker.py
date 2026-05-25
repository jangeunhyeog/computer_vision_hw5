from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import state_to_bbox
from .histogram import compute_hsv_histogram, histogram_likelihood
from .particles import (
    estimate_state,
    initialize_particles,
    normalize_weights,
    systematic_resample,
    transition,
)


@dataclass
class TrackerConfig:
    num_particles: int = 100
    init_sigma_xy: float = 4.0
    init_sigma_scale: float = 0.02
    transition_sigma_xy: float = 0.15
    transition_sigma_scale: float = 0.05
    lambda_: float = 20.0
    hs_bins: tuple[int, int] = (10, 10)
    v_bins: int = 10
    min_h: float = 0.1
    min_s: float = 0.2


class ParticleFilterTracker:
    """Color-histogram particle-filter tracker.

    Usage:
        tracker = ParticleFilterTracker(init_frame, init_bbox, rng, config)
        pred_bbox = tracker.update(next_frame)
    """

    def __init__(
        self,
        init_frame_rgb: np.ndarray,
        init_bbox_xywh: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.rng = rng or np.random.default_rng()
        self.init_bbox = np.asarray(init_bbox_xywh, dtype=np.float64)
        self.initial_size = (float(self.init_bbox[2]), float(self.init_bbox[3]))

        self.target_hist = compute_hsv_histogram(
            init_frame_rgb,
            self.init_bbox,
            hs_bins=self.config.hs_bins,
            v_bins=self.config.v_bins,
            min_h=self.config.min_h,
            min_s=self.config.min_s,
        )
        self.particles = initialize_particles(
            self.init_bbox,
            self.config.num_particles,
            self.rng,
            sigma_xy=self.config.init_sigma_xy,
            sigma_scale=self.config.init_sigma_scale,
        )
        self.weights = np.full(self.config.num_particles, 1.0 / self.config.num_particles, dtype=np.float64)
        self.current_state = estimate_state(self.particles, self.weights)

    def update(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Process one frame and return predicted bbox [x, y, w, h]."""
        # 1. Resampling from previous posterior.
        self.particles = systematic_resample(self.particles, self.weights, self.rng)
        self.weights = np.full(len(self.particles), 1.0 / len(self.particles), dtype=np.float64)

        # 2. State transition / prediction.
        self.particles = transition(
            self.particles,
            self.rng,
            initial_size=self.initial_size,
            sigma_xy=self.config.transition_sigma_xy,
            sigma_scale=self.config.transition_sigma_scale,
        )

        # 3. Observation likelihood for each particle.
        likelihoods = np.zeros(len(self.particles), dtype=np.float64)
        for i, state in enumerate(self.particles):
            bbox = state_to_bbox(state, self.initial_size)
            cand_hist = compute_hsv_histogram(
                frame_rgb,
                bbox,
                hs_bins=self.config.hs_bins,
                v_bins=self.config.v_bins,
                min_h=self.config.min_h,
                min_s=self.config.min_s,
            )
            likelihoods[i] = histogram_likelihood(cand_hist, self.target_hist, lambda_=self.config.lambda_)

        self.weights = normalize_weights(likelihoods)

        # 4. Posterior estimate.
        self.current_state = estimate_state(self.particles, self.weights)
        return state_to_bbox(self.current_state, self.initial_size)
