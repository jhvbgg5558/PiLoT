#!/usr/bin/env python3
"""Analyze P11 synthetic sequence localization ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.pose_adapter import normalize_angle_deg


DEFAULT_SEQUENCE_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11"
DEFAULT_P11_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11"
DEFAULT_GATE_V2_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_gate_v2"
DEFAULT_FIRSTPOSE_GATE_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_firstpose_gate"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_ablation_p11"


def angular_diff_deg(a: float, b: float) -> float:
    return abs(normalize_angle_deg(float(a) - float(b)))


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _f(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    if val is None or val == "":
        return default
    return float(val)


def _pose_from_row(row: Dict[str, str], prefix: str) -> Dict[str, Any]:
    return {
        "frame_id": row["frame_id"],
        "east": _f(row, f"{prefix}_east"),
        "north": _f(row, f"{prefix}_north"),
        "alt": _f(row, f"{prefix}_alt"),
        "yaw": _f(row, f"{prefix}_yaw"),
        "method": prefix,
    }


def _gt_from_row(row: Dict[str, str]) -> Dict[str, Any]:
    return _pose_from_row(row, "gt")


def _error(gt: Dict[str, Any], pose: Dict[str, Any]) -> Dict[str, float]:
    dx = float(pose["east"]) - float(gt["east"])
    dy = float(pose["north"]) - float(gt["north"])
    dz = float(pose["alt"]) - float(gt["alt"])
    return {
        "xy_error_m": float(math.hypot(dx, dy)),
        "alt_error_m": float(abs(dz)),
        "yaw_error_deg": float(angular_diff_deg(pose["yaw"], gt["yaw"])),
    }


def _stats(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"rmse": None, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _load_odom_edges(path: Path) -> Dict[str, Dict[str, Any]]:
    edges: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src, dst, dx, dy, dz, dyaw, dist = line.split()[:7]
        edges[src] = {
            "src": src,
            "dst": dst,
            "dx_m": float(dx),
            "dy_m": float(dy),
            "dz_m": float(dz),
            "dyaw_deg": float(dyaw),
            "distance_m": float(dist),
        }
    return edges


def _temporal_pred_only(rows: List[Dict[str, str]], odom: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    prev: Optional[Dict[str, Any]] = None
    prev_name: Optional[str] = None
    for row in rows:
        name = row["frame_id"]
        if prev is None or prev_name not in odom:
            pose = _pose_from_row(row, "init")
            pose["method"] = "init_seed"
        else:
            edge = odom[prev_name]
            if edge["dst"] != name:
                pose = _pose_from_row(row, "init")
                pose["method"] = "init_seed_missing_odom"
            else:
                pose = {
                    "frame_id": name,
                    "east": float(prev["east"]) + edge["dx_m"],
                    "north": float(prev["north"]) + edge["dy_m"],
                    "alt": float(prev["alt"]) + edge["dz_m"],
                    "yaw": normalize_angle_deg(float(prev["yaw"]) + edge["dyaw_deg"]),
                    "method": "temporal_pred_only",
                }
        out.append(pose)
        prev = pose
        prev_name = name
    return out


def _best_gate_no_temporal_for_frame(frame_id: str, init_pose: Dict[str, Any], candidates: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    rows = [
        r for r in candidates.get(frame_id, [])
        if r.get("method") != "temporal_predicted"
        and r.get("method") != "init"
        and r.get("visual_gate_pass") == "True"
        and r.get("pose_gate_pass") == "True"
    ]
    if not rows:
        out = dict(init_pose)
        out["method"] = "init"
        return out
    row = min(rows, key=lambda r: _f(r, "score", float("inf")))
    return {
        "frame_id": frame_id,
        "east": _f(row, "east"),
        "north": _f(row, "north"),
        "alt": _f(row, "alt"),
        "yaw": _f(row, "yaw"),
        "method": row["method"],
    }


def _strategy_poses(
    strategy: str,
    p11_rows: List[Dict[str, str]],
    gate_rows: List[Dict[str, str]],
    candidates: Dict[str, List[Dict[str, str]]],
    odom: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if strategy == "init_only":
        return [_pose_from_row(r, "init") for r in gate_rows]
    if strategy == "raw_refined_only":
        return [_pose_from_row(r, "refined") for r in gate_rows]
    if strategy == "corrected_refined_only":
        out = []
        for r in gate_rows:
            pose = _pose_from_row(r, "refined")
            pose["yaw"] = _f(r, "corrected_refined_downward_yaw")
            pose["method"] = "corrected_refined_only"
            out.append(pose)
        return out
    if strategy == "old_visual_gate":
        return [
            {
                "frame_id": r["frame_id"],
                "east": _f(r, "refined_east") if r.get("selected_method") == "refined" else _f(r, "init_east"),
                "north": _f(r, "refined_north") if r.get("selected_method") == "refined" else _f(r, "init_north"),
                "alt": _f(r, "refined_alt") if r.get("selected_method") == "refined" else _f(r, "init_alt"),
                "yaw": _f(r, "refined_yaw") if r.get("selected_method") == "refined" else _f(r, "init_yaw"),
                "method": r.get("selected_method") or "unknown",
            }
            for r in p11_rows
        ]
    if strategy == "temporal_pred_only":
        return _temporal_pred_only(gate_rows, odom)
    if strategy == "gate_v2_no_temporal":
        return [_best_gate_no_temporal_for_frame(r["frame_id"], _pose_from_row(r, "init"), candidates) for r in gate_rows]
    if strategy == "gate_v2_current":
        return [
            {
                "frame_id": r["frame_id"],
                "east": _f(r, "selected_east"),
                "north": _f(r, "selected_north"),
                "alt": _f(r, "selected_alt"),
                "yaw": _f(r, "selected_yaw"),
                "method": r.get("selected_method") or "unknown",
            }
            for r in gate_rows
        ]
    raise ValueError(f"Unknown strategy: {strategy}")


def _evaluate_strategy(strategy: str, poses: List[Dict[str, Any]], gt_rows: List[Dict[str, str]], init_rows: List[Dict[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_frame: List[Dict[str, Any]] = []
    xy_errors: List[float] = []
    yaw_errors: List[float] = []
    init_xy_errors: List[float] = []
    counts = Counter()
    for pose, gt_row, init_row in zip(poses, gt_rows, init_rows):
        gt = _gt_from_row(gt_row)
        init_pose = _pose_from_row(init_row, "init")
        err = _error(gt, pose)
        init_err = _error(gt, init_pose)
        xy_errors.append(err["xy_error_m"])
        yaw_errors.append(err["yaw_error_deg"])
        init_xy_errors.append(init_err["xy_error_m"])
        counts[pose.get("method", strategy)] += 1
        per_frame.append(
            {
                "frame_id": pose["frame_id"],
                "strategy": strategy,
                "method": pose.get("method", strategy),
                "east": pose["east"],
                "north": pose["north"],
                "alt": pose["alt"],
                "yaw": pose["yaw"],
                **err,
                "init_xy_error_m": init_err["xy_error_m"],
            }
        )
    xy = _stats(xy_errors)
    yaw = _stats(yaw_errors)
    end_gt = _gt_from_row(gt_rows[-1])
    end_err = _error(end_gt, poses[-1])
    improved = np.asarray(xy_errors) < np.asarray(init_xy_errors)
    worse = np.asarray(xy_errors) > np.asarray(init_xy_errors)
    summary = {
        "strategy": strategy,
        "xy_rmse": xy["rmse"],
        "xy_mean": xy["mean"],
        "xy_median": xy["median"],
        "xy_p90": xy["p90"],
        "yaw_rmse": yaw["rmse"],
        "yaw_mean": yaw["mean"],
        "yaw_median": yaw["median"],
        "yaw_p90": yaw["p90"],
        "improved_frame_ratio_vs_init": float(np.mean(improved)),
        "worse_frame_ratio_vs_init": float(np.mean(worse)),
        "max_xy_error": xy["max"],
        "max_yaw_error": yaw["max"],
        "drift_end_error": end_err["xy_error_m"],
        "selected_method_counts": dict(counts),
    }
    return summary, per_frame


def _plot_bar(rows: List[Dict[str, Any]], key: str, path: Path, ylabel: str) -> None:
    labels = [r["strategy"] for r in rows]
    vals = [r[key] for r in rows]
    plt.figure(figsize=(12, 5))
    plt.bar(labels, vals)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_curves(per_frame: List[Dict[str, Any]], strategies: List[str], key: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(12, 5))
    for strategy in strategies:
        rows = [r for r in per_frame if r["strategy"] == strategy]
        rows = sorted(rows, key=lambda r: r["frame_id"])
        plt.plot(np.arange(len(rows)), [r[key] for r in rows], label=strategy)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_trajectory(per_frame: List[Dict[str, Any]], gt_rows: List[Dict[str, str]], strategies: List[str], path: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.plot([_f(r, "gt_east") for r in gt_rows], [_f(r, "gt_north") for r in gt_rows], "k-", label="GT", linewidth=2)
    for strategy in strategies:
        rows = sorted([r for r in per_frame if r["strategy"] == strategy], key=lambda r: r["frame_id"])
        plt.plot([r["east"] for r in rows], [r["north"] for r in rows], marker=".", linewidth=1, markersize=2, label=strategy)
    plt.axis("equal")
    plt.xlabel("Raster X / Easting (m)")
    plt.ylabel("Raster Y / Northing (m)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(path: Path, summaries: List[Dict[str, Any]]) -> None:
    by_name = {s["strategy"]: s for s in summaries}
    gate = by_name.get("gate_v2_current")
    temporal = by_name.get("temporal_pred_only")
    no_temporal = by_name.get("gate_v2_no_temporal")
    corrected = by_name.get("corrected_refined_only")
    raw = by_name.get("raw_refined_only")
    init = by_name.get("init_only")
    first_init = by_name.get("firstpose_odom_init_only")
    first_gate = by_name.get("firstpose_odom_safe_selected")
    lines = [
        "# P11.4 Synthetic Sequence Ablation Check",
        "",
        "## Answers",
        f"1. Old gate_v2 accuracy: XY RMSE `{gate['xy_rmse'] if gate else None}` m, yaw RMSE `{gate['yaw_rmse'] if gate else None}` deg.",
        f"2. temporal_pred_only accuracy: XY RMSE `{temporal['xy_rmse'] if temporal else None}` m, yaw RMSE `{temporal['yaw_rmse'] if temporal else None}` deg.",
        f"3. firstpose_odom init accuracy: XY RMSE `{first_init['xy_rmse'] if first_init else None}` m, yaw RMSE `{first_init['yaw_rmse'] if first_init else None}` deg.",
        f"4. firstpose_odom safe selected accuracy: XY RMSE `{first_gate['xy_rmse'] if first_gate else None}` m, yaw RMSE `{first_gate['yaw_rmse'] if first_gate else None}` deg.",
        f"5. firstpose safe vs old gate_v2 XY delta `{(first_gate['xy_rmse'] - gate['xy_rmse']) if first_gate and gate else None}` m; lower is better.",
        f"6. gate_v2_no_temporal vs init: `{no_temporal['xy_rmse'] if no_temporal else None}` m vs `{init['xy_rmse'] if init else None}` m.",
        f"7. corrected_refined_only vs raw_refined_only: XY `{corrected['xy_rmse'] if corrected else None}` vs `{raw['xy_rmse'] if raw else None}`, yaw `{corrected['yaw_rmse'] if corrected else None}` vs `{raw['yaw_rmse'] if raw else None}`.",
        "",
        "## Recommendation",
    ]
    if first_gate and first_init and first_gate["xy_rmse"] <= first_init["xy_rmse"]:
        lines.append("Use first-pose+odom safe gate as the P11 sequence baseline; it improves or preserves the propagated prior without trusting feature refinement blindly.")
    elif gate and temporal and no_temporal and init and gate["xy_rmse"] <= temporal["xy_rmse"] * 0.95 and no_temporal["xy_rmse"] < init["xy_rmse"]:
        lines.append("Proceed to P12 sliding window; PiLoT candidates add value beyond odometry.")
    elif gate and temporal and temporal["xy_rmse"] <= gate["xy_rmse"] * 1.05:
        lines.append("The current gain is dominated by odom/temporal prediction. Do P12 as an odometry baseline, but return to single-frame scorer/feature loss before claiming image-based correction.")
    else:
        lines.append("Optimize single-frame scorer/feature loss before relying on temporal smoothing.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--p11-result-dir", default=DEFAULT_P11_DIR)
    parser.add_argument("--gate-v2-dir", default=DEFAULT_GATE_V2_DIR)
    parser.add_argument("--firstpose-gate-dir", default=DEFAULT_FIRSTPOSE_GATE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = _resolve(args.sequence_dir)
    p11_dir = _resolve(args.p11_result_dir)
    gate_dir = _resolve(args.gate_v2_dir)
    firstpose_dir = _resolve(args.firstpose_gate_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    p11_rows = _read_csv(p11_dir / "per_frame_metrics.csv")
    gate_rows = _read_csv(gate_dir / "per_frame_metrics.csv")
    cand_rows = _read_csv(gate_dir / "candidate_gate_metrics.csv")
    firstpose_rows: List[Dict[str, str]] = []
    if (firstpose_dir / "per_frame_metrics.csv").exists():
        firstpose_rows = _read_csv(firstpose_dir / "per_frame_metrics.csv")
    odom = _load_odom_edges(sequence_dir / "poses" / "odom_edges.txt")
    candidates: Dict[str, List[Dict[str, str]]] = {}
    for row in cand_rows:
        candidates.setdefault(row["frame_id"], []).append(row)

    strategies = [
        "init_only",
        "raw_refined_only",
        "corrected_refined_only",
        "old_visual_gate",
        "temporal_pred_only",
        "gate_v2_no_temporal",
        "gate_v2_current",
    ]
    summaries: List[Dict[str, Any]] = []
    all_per_frame: List[Dict[str, Any]] = []
    method_counts: Dict[str, Any] = {}
    for strategy in strategies:
        poses = _strategy_poses(strategy, p11_rows, gate_rows, candidates, odom)
        summary, per_frame = _evaluate_strategy(strategy, poses, gate_rows, gate_rows)
        summaries.append(summary)
        all_per_frame.extend(per_frame)
        method_counts[strategy] = summary["selected_method_counts"]

    if firstpose_rows:
        firstpose_strategies = [
            ("firstpose_odom_init_only", [_pose_from_row(r, "init") for r in firstpose_rows]),
            (
                "firstpose_odom_safe_selected",
                [
                    {
                        "frame_id": r["frame_id"],
                        "east": _f(r, "selected_east"),
                        "north": _f(r, "selected_north"),
                        "alt": _f(r, "selected_alt"),
                        "yaw": _f(r, "selected_yaw"),
                        "method": r.get("selected_method") or "unknown",
                    }
                    for r in firstpose_rows
                ],
            ),
        ]
        for strategy, poses in firstpose_strategies:
            summary, per_frame = _evaluate_strategy(strategy, poses, firstpose_rows, firstpose_rows)
            summaries.append(summary)
            all_per_frame.extend(per_frame)
            method_counts[strategy] = summary["selected_method_counts"]

    best_xy = min(summaries, key=lambda r: r["xy_rmse"])
    best_yaw = min(summaries, key=lambda r: r["yaw_rmse"])
    ablation_summary = {
        "strategies": summaries,
        "best_xy_strategy": best_xy["strategy"],
        "best_xy_rmse": best_xy["xy_rmse"],
        "best_yaw_strategy": best_yaw["strategy"],
        "best_yaw_rmse": best_yaw["yaw_rmse"],
        "interpretation": (
            "Compare gate_v2_current with temporal_pred_only to determine whether gains come from odometry/temporal prior or PiLoT image refinement."
        ),
    }
    _write_json(output_dir / "ablation_summary.json", ablation_summary)
    _write_csv(output_dir / "ablation_metrics.csv", summaries)
    _write_csv(output_dir / "ablation_per_frame.csv", all_per_frame)
    _write_json(output_dir / "method_counts.json", method_counts)

    _plot_trajectory(all_per_frame, gate_rows, strategies, output_dir / "trajectory_comparison.png")
    _plot_bar(summaries, "xy_rmse", output_dir / "xy_rmse_bar.png", "XY RMSE (m)")
    _plot_bar(summaries, "yaw_rmse", output_dir / "yaw_rmse_bar.png", "Yaw RMSE (deg)")
    _plot_curves(all_per_frame, strategies, "xy_error_m", output_dir / "xy_error_curves.png", "XY error (m)")
    _plot_curves(all_per_frame, strategies, "yaw_error_deg", output_dir / "yaw_error_curves.png", "Yaw error (deg)")
    _write_report(output_dir.parent / "sim_sequence_ablation_p11_check.md", summaries)
    print(json.dumps(_jsonable(ablation_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
