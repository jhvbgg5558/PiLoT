#!/usr/bin/env python3
"""Evaluate per-frame render input poses against refined PiLoT outputs."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.eval import _euler_to_rotation_ecef


Pose = Tuple[List[float], List[float]]


def _pose_from_row(row: Dict[str, str], prefix: str) -> Pose:
    trans = [
        float(row[f"{prefix}_lon"]),
        float(row[f"{prefix}_lat"]),
        float(row[f"{prefix}_alt"]),
    ]
    euler_file = [
        float(row[f"{prefix}_roll"]),
        float(row[f"{prefix}_pitch"]),
        float(row[f"{prefix}_yaw"]),
    ]
    return trans, euler_file


def _euler_file_to_eval(euler_file: Iterable[float]) -> List[float]:
    roll, pitch, yaw = [float(v) for v in euler_file]
    return [pitch, roll, yaw]


def _pose_errors(pred: Pose, gt: Pose) -> Tuple[float, float, float]:
    pred_trans, pred_euler_file = pred
    gt_trans, gt_euler_file = gt

    pred_euler = _euler_file_to_eval(pred_euler_file)
    gt_euler = _euler_file_to_eval(gt_euler_file)

    pred_R, pred_t = _euler_to_rotation_ecef(pred_euler, pred_trans)
    gt_R, gt_t = _euler_to_rotation_ecef(gt_euler, gt_trans)

    trans_error = float(np.linalg.norm(pred_t - gt_t))
    cos = np.clip((np.trace(gt_R.T @ pred_R) - 1) / 2, -1.0, 1.0)
    rot_error = float(np.rad2deg(abs(np.arccos(cos))))
    yaw_error = float(abs((pred_euler[-1] - gt_euler[-1] + 180) % 360 - 180))
    return trans_error, rot_error, yaw_error


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _recall(values: List[float], thresholds: Iterable[float]) -> Dict[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {float(th): float(np.mean(arr < th)) for th in thresholds}


def _format_improvement(initial: float, refined: float, unit: str) -> str:
    delta = initial - refined
    if initial == 0:
        return f"{delta:.3f} {unit} (n/a)"
    pct = delta / initial * 100.0
    return f"{delta:.3f} {unit} ({pct:.2f}%)"


def evaluate_trace(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    render_errors = {"translation": [], "rotation": [], "yaw": []}
    refined_errors = {"translation": [], "rotation": [], "yaw": []}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = _pose_from_row(row, "gt")
            render = _pose_from_row(row, "render")
            refined = _pose_from_row(row, "refined")

            rt, rr, ry = _pose_errors(render, gt)
            ft, fr, fy = _pose_errors(refined, gt)

            render_errors["translation"].append(rt)
            render_errors["rotation"].append(rr)
            render_errors["yaw"].append(ry)
            refined_errors["translation"].append(ft)
            refined_errors["rotation"].append(fr)
            refined_errors["yaw"].append(fy)

    if not render_errors["translation"]:
        raise ValueError(f"No pose rows found in {path}")

    return {
        "render": {
            key: _stats(values) for key, values in render_errors.items()
        },
        "refined": {
            key: _stats(values) for key, values in refined_errors.items()
        },
        "render_recall": {
            "translation": _recall(render_errors["translation"], [1, 3, 5]),
            "rotation": _recall(render_errors["rotation"], [1, 3, 5]),
        },
        "refined_recall": {
            "translation": _recall(refined_errors["translation"], [1, 3, 5]),
            "rotation": _recall(refined_errors["rotation"], [1, 3, 5]),
        },
        "count": {"frames": len(render_errors["translation"])},
    }


def _print_summary(stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    render = stats["render"]
    refined = stats["refined"]
    frames = int(stats["count"]["frames"])

    rows = [
        ("Translation median", "translation", "m"),
        ("Rotation median", "rotation", "deg"),
        ("Yaw median", "yaw", "deg"),
    ]

    print(f"Frames: {frames}")
    print()
    print("| Metric | Per-frame input/render pose | Refined output | Improvement |")
    print("|---|---:|---:|---:|")
    for label, key, unit in rows:
        initial = render[key]["median"]
        final = refined[key]["median"]
        improvement = _format_improvement(initial, final, unit)
        print(
            f"| {label} | {initial:.3f} {unit} | "
            f"{final:.3f} {unit} | {improvement} |"
        )

    print()
    print("Detailed stats:")
    print("| Pose | Metric | Median | Mean | Std | Min | Max |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for pose_name, pose_stats in [("Render", render), ("Refined", refined)]:
        for key, unit in [
            ("translation", "m"),
            ("rotation", "deg"),
            ("yaw", "deg"),
        ]:
            values = pose_stats[key]
            print(
                f"| {pose_name} | {key} | "
                f"{values['median']:.3f} {unit} | "
                f"{values['mean']:.3f} {unit} | "
                f"{values['std']:.3f} {unit} | "
                f"{values['min']:.3f} {unit} | "
                f"{values['max']:.3f} {unit} |"
            )

    print()
    print("Recall:")
    print("| Pose | Threshold | Success Rate |")
    print("|---|---|---:|")
    for pose_name, recall_stats in [
        ("Render", stats["render_recall"]),
        ("Refined", stats["refined_recall"]),
    ]:
        for key, unit in [("translation", "m"), ("rotation", "deg")]:
            for threshold, rate in recall_stats[key].items():
                print(
                    f"| {pose_name} | {key} < {threshold:g}{unit} | "
                    f"{rate * 100:.2f}% |"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-frame render input pose and refined pose trace.",
    )
    parser.add_argument(
        "trace_csv",
        type=Path,
        help="Path to outputs/<name>_pose_trace.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = evaluate_trace(args.trace_csv)
    _print_summary(stats)


if __name__ == "__main__":
    main()
