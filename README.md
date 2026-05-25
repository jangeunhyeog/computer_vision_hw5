# Assignment 5: Particle Filter Tracking (Python Skeleton)

This project is the Python skeleton for **Visual Tracking by Particle Filtering**.
Students complete a lightweight implementation of a color-histogram particle-filter tracker.

## What students implement

Core TODOs are marked with `TODO(STUDENT)` in the source code.

## Directory structure

```text
assignment5_particle_filter_tracking_project/
  pf_tracking/             # library code
  scripts/                 # command-line scripts
  tests/                   # public smoke tests
  data/                    # put sequences here; not committed
  outputs/                 # generated results; not submitted unless requested
```

The assigned sequence is **Bolt** (OTB-2015). 

```text
data/Bolt/
  img/
    0001.jpg
    0002.jpg
    ...
  groundtruth_rect.txt     # each line: x,y,w,h (comma-separated)
```


## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Optional sanity data

The toy data generator creates a short synthetic moving-square sequence so students can debug their implementation before using real videos.

```bash
python scripts/generate_toy_data.py --out data/toy_red_square --frames 30
```

## Running the tracker

Quick sanity check on the toy sequence:

```bash
python scripts/run_tracker.py --sequence data/toy_red_square --output outputs/toy_red_square --seed 0
python scripts/evaluate_tracker.py --sequence data/toy_red_square --result outputs/toy_red_square/predicted_bboxes.txt --output outputs/toy_red_square
```

Graded run on the assigned sequence (Bolt):

```bash
python scripts/run_tracker.py --sequence data/Bolt --output outputs/Bolt --seed 2026
python scripts/evaluate_tracker.py --sequence data/Bolt --result outputs/Bolt/predicted_bboxes.txt --output outputs/Bolt
```

## Submission

Submit a zip file named `hw5_<STUDENT_ID>.zip` containing:

- `pf_tracking/` with your completed implementation
- `scripts/` if modified
- generated plots and a report


