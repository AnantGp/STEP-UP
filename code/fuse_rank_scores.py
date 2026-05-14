#!/usr/bin/env python3
"""Fuse two StepUP submission score files by empirical rank.

The top ArogyaPandit submission used a weighted rank fusion of two calibrated
S-norm submission candidates. Inputs may be submission zips containing
``scores.txt`` or plain score text files.
"""

from __future__ import annotations

import argparse
import pathlib
import zipfile

import numpy as np


def load_scores(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            scores = np.loadtxt(zf.open("scores.txt"))
    else:
        scores = np.loadtxt(path)
    if scores.ndim == 2:
        scores = scores.mean(axis=1)
    return np.asarray(scores, dtype=np.float64)


def empirical_rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks / float(len(scores))


def save_submission(scores: np.ndarray, threshold: float, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_path = output.with_suffix(".scores.txt")
    threshold_path = output.with_suffix(".threshold.txt")
    np.savetxt(scores_path, scores)
    threshold_path.write_text(f"{threshold:g}")
    with zipfile.ZipFile(output, mode="w") as zf:
        zf.write(scores_path, arcname="scores.txt")
        zf.write(threshold_path, arcname="threshold.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=pathlib.Path, required=True)
    parser.add_argument("--secondary", type=pathlib.Path, required=True)
    parser.add_argument("--secondary-weight", type=float, default=0.115)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    primary = load_scores(args.primary)
    secondary = load_scores(args.secondary)
    if len(primary) != len(secondary):
        raise SystemExit(
            f"Score length mismatch: primary={len(primary)} secondary={len(secondary)}"
        )
    if not 0.0 <= args.secondary_weight <= 1.0:
        raise SystemExit("--secondary-weight must be in [0, 1]")

    w = args.secondary_weight
    fused = (1.0 - w) * empirical_rank(primary) + w * empirical_rank(secondary)
    save_submission(fused, args.threshold, args.output)
    print(f"Wrote {args.output} with secondary_weight={w:.3f}")


if __name__ == "__main__":
    main()
