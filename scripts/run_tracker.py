from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pf_tracking.data import load_sequence, read_image, save_bboxes
from pf_tracking.tracker import ParticleFilterTracker, TrackerConfig
from pf_tracking.visualize import save_debug_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run particle-filter tracking on one sequence.")
    parser.add_argument("--sequence", type=Path, required=True, help="Sequence directory with img/ and groundtruth.txt")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--save-debug-every", type=int, default=10)
    args = parser.parse_args()

    frames, gt = load_sequence(args.sequence)
    rng = np.random.default_rng(args.seed)
    config = TrackerConfig(num_particles=args.num_particles)

    first_frame = read_image(frames[0])
    tracker = ParticleFilterTracker(first_frame, gt[0], rng=rng, config=config)

    preds = [gt[0].astype(float)]
    args.output.mkdir(parents=True, exist_ok=True)
    save_debug_frame(args.output / "debug" / f"{0:04d}.jpg", first_frame, preds[0])

    for t, frame_path in enumerate(frames[1:], start=1):
        frame = read_image(frame_path)
        pred = tracker.update(frame)
        preds.append(pred)
        if args.save_debug_every > 0 and (t % args.save_debug_every == 0 or t == len(frames) - 1):
            save_debug_frame(args.output / "debug" / f"{t:04d}.jpg", frame, pred)

    pred_arr = np.asarray(preds, dtype=np.float64)
    save_bboxes(args.output / "predicted_bboxes.txt", pred_arr)
    print(f"Saved predicted boxes to {args.output / 'predicted_bboxes.txt'}")


if __name__ == "__main__":
    main()
