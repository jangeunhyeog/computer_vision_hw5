import numpy as np

from pf_tracking.geometry import bbox_to_state, state_to_bbox
from pf_tracking.metrics import bbox_iou
from pf_tracking.particles import normalize_weights


def test_bbox_iou_identical():
    box = np.array([10, 20, 30, 40], dtype=float)
    assert bbox_iou(box, box) == 1.0


def test_bbox_iou_no_overlap():
    assert bbox_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 5, 5])) == 0.0


def test_normalize_weights_fallback():
    w = normalize_weights(np.array([0.0, 0.0, 0.0]))
    assert np.allclose(w, np.ones(3) / 3)


def test_state_bbox_roundtrip_initial_scale():
    bbox = np.array([10, 20, 30, 40], dtype=float)
    state = bbox_to_state(bbox, initial_size=(30, 40))
    out = state_to_bbox(state, initial_size=(30, 40))
    assert np.allclose(out, bbox)
