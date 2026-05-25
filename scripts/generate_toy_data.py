from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a toy moving-square tracking sequence.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    args = parser.parse_args()

    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_lines = []
    box_w, box_h = 24, 18

    for t in range(args.frames):
        image = np.full((args.height, args.width, 3), 35, dtype=np.uint8)
        x = 15 + 3 * t
        y = 35 + int(8 * np.sin(t / 4.0))
        x = min(x, args.width - box_w - 2)
        y = min(max(y, 1), args.height - box_h - 2)
        image[y : y + box_h, x : x + box_w] = np.asarray([220, 35, 35], dtype=np.uint8)

        # Add mild clutter.
        image[20:45, 105:135] = np.asarray([35, 220, 35], dtype=np.uint8)
        gt_lines.append(f"{x},{y},{box_w},{box_h}\n")
        cv2.imwrite(str(img_dir / f"{t+1:04d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    (args.out / "groundtruth.txt").write_text("".join(gt_lines))
    print(f"Wrote toy sequence to {args.out}")


if __name__ == "__main__":
    main()
