from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pf_tracking.data import load_bboxes, load_ground_truth
from pf_tracking.metrics import compute_ious
from pf_tracking.visualize import plot_ious


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tracking boxes using IoU.")
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gt = load_ground_truth(args.sequence)
    pred = load_bboxes(args.result)
    ious = compute_ious(pred, gt)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "ious.txt", ious, fmt="%.6f")
    plot_ious(ious, args.output)
    print(f"Mean IoU: {ious.mean():.4f}")
    print(f"Saved plots to {args.output}")


if __name__ == "__main__":
    main()
