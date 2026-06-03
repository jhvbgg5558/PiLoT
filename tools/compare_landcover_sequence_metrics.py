#!/usr/bin/env python3
"""Compare P11 first-pose gate metrics across landcover-biased sequences."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BUILDING_SEQUENCE_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11"
DEFAULT_BUILDING_RUN_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_firstpose_gate"
DEFAULT_LANDCOVER_SEQUENCE_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover"
DEFAULT_LANDCOVER_RUN_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_landcover_firstpose_gate"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_ablation_p11_landcover"


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_json(path: Path) -> Dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key)
    if val is None or val == "":
        return None
    return float(val)


def _stats(rows: List[Dict[str, str]], key: str) -> Dict[str, Optional[float]]:
    vals = np.asarray([v for v in (_f(row, key) for row in rows) if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(vals.max()),
    }


def _case_summary(label: str, sequence_dir: Path, run_dir: Path) -> Dict[str, Any]:
    seq_summary = _maybe_json(sequence_dir / "summary.json")
    run_summary = _read_json(run_dir / "summary_metrics.json")
    gate = _maybe_json(run_dir / "gate_decision_breakdown.json")
    rows = _read_csv(run_dir / "per_frame_metrics.csv")
    return {
        "label": label,
        "sequence_dir": sequence_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "landcover_summary": seq_summary.get("landcover_summary", {}),
        "init_xy_rmse": run_summary.get("init_xy_rmse"),
        "selected_xy_rmse": run_summary.get("selected_xy_rmse"),
        "raw_refined_xy_rmse": run_summary.get("raw_refined_xy_rmse"),
        "init_yaw_rmse": run_summary.get("init_yaw_rmse"),
        "selected_yaw_rmse": run_summary.get("selected_yaw_rmse"),
        "raw_refined_yaw_rmse": run_summary.get("raw_refined_yaw_rmse"),
        "safe_gate_accept_ratio": run_summary.get("safe_gate_accept_ratio"),
        "worse_frame_ratio": run_summary.get("worse_frame_ratio"),
        "selected_method_counts": run_summary.get("selected_method_counts", {}),
        "visual_reject_reasons": gate.get("visual_reject_reasons", {}),
        "pose_reject_reasons": gate.get("pose_reject_reasons", {}),
        "temporal_reject_reasons": gate.get("temporal_reject_reasons", {}),
        "selected_xy_error_stats": _stats(rows, "selected_xy_error_m"),
        "selected_yaw_error_stats": _stats(rows, "selected_yaw_error_deg"),
    }


def _write_csv(path: Path, cases: List[Dict[str, Any]]) -> None:
    fields = [
        "label",
        "vegetation_ratio",
        "road_like_ratio",
        "edge_density",
        "height_variation",
        "building_density",
        "landcover_score",
        "init_xy_rmse",
        "selected_xy_rmse",
        "raw_refined_xy_rmse",
        "init_yaw_rmse",
        "selected_yaw_rmse",
        "raw_refined_yaw_rmse",
        "safe_gate_accept_ratio",
        "worse_frame_ratio",
        "selected_xy_mean",
        "selected_xy_median",
        "selected_xy_p90",
        "selected_xy_max",
        "selected_yaw_mean",
        "selected_yaw_median",
        "selected_yaw_p90",
        "selected_yaw_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            lc = case.get("landcover_summary", {})
            xy = case.get("selected_xy_error_stats", {})
            yaw = case.get("selected_yaw_error_stats", {})
            writer.writerow(
                {
                    "label": case["label"],
                    "vegetation_ratio": lc.get("vegetation_ratio"),
                    "road_like_ratio": lc.get("road_like_ratio"),
                    "edge_density": lc.get("edge_density"),
                    "height_variation": lc.get("height_variation"),
                    "building_density": lc.get("building_density"),
                    "landcover_score": lc.get("landcover_score"),
                    "init_xy_rmse": case.get("init_xy_rmse"),
                    "selected_xy_rmse": case.get("selected_xy_rmse"),
                    "raw_refined_xy_rmse": case.get("raw_refined_xy_rmse"),
                    "init_yaw_rmse": case.get("init_yaw_rmse"),
                    "selected_yaw_rmse": case.get("selected_yaw_rmse"),
                    "raw_refined_yaw_rmse": case.get("raw_refined_yaw_rmse"),
                    "safe_gate_accept_ratio": case.get("safe_gate_accept_ratio"),
                    "worse_frame_ratio": case.get("worse_frame_ratio"),
                    "selected_xy_mean": xy.get("mean"),
                    "selected_xy_median": xy.get("median"),
                    "selected_xy_p90": xy.get("p90"),
                    "selected_xy_max": xy.get("max"),
                    "selected_yaw_mean": yaw.get("mean"),
                    "selected_yaw_median": yaw.get("median"),
                    "selected_yaw_p90": yaw.get("p90"),
                    "selected_yaw_max": yaw.get("max"),
                }
            )


def _write_report(path: Path, cases: List[Dict[str, Any]]) -> None:
    by_label = {case["label"]: case for case in cases}
    building = by_label.get("building_firstpose_gate")
    landcover = by_label.get("landcover_firstpose_gate")
    lines = [
        "# P11 Landcover vs Building First-Pose Gate Comparison",
        "",
        "| Case | Veg | Road-like | Building density | XY RMSE | Yaw RMSE | Safe accept | Worse ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lc = case.get("landcover_summary", {})
        lines.append(
            "| {label} | {veg} | {road} | {building_density} | {xy} | {yaw} | {accept} | {worse} |".format(
                label=case["label"],
                veg=lc.get("vegetation_ratio"),
                road=lc.get("road_like_ratio"),
                building_density=lc.get("building_density"),
                xy=case.get("selected_xy_rmse"),
                yaw=case.get("selected_yaw_rmse"),
                accept=case.get("safe_gate_accept_ratio"),
                worse=case.get("worse_frame_ratio"),
            )
        )
    lines.extend(["", "## Delta"])
    if building and landcover:
        lines.append(f"- selected XY RMSE delta landcover - building: `{landcover['selected_xy_rmse'] - building['selected_xy_rmse']}` m")
        lines.append(f"- selected yaw RMSE delta landcover - building: `{landcover['selected_yaw_rmse'] - building['selected_yaw_rmse']}` deg")
        lines.append(f"- raw refined XY RMSE delta landcover - building: `{landcover['raw_refined_xy_rmse'] - building['raw_refined_xy_rmse']}` m")
        lines.append(f"- safe gate accept ratio delta: `{landcover['safe_gate_accept_ratio'] - building['safe_gate_accept_ratio']}`")
    lines.extend(["", "## Gate Rejection Reasons"])
    for case in cases:
        lines.append(f"- `{case['label']}` visual={case.get('visual_reject_reasons')} pose={case.get('pose_reject_reasons')} temporal={case.get('temporal_reject_reasons')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--building-sequence-dir", default=DEFAULT_BUILDING_SEQUENCE_DIR)
    parser.add_argument("--building-run-dir", default=DEFAULT_BUILDING_RUN_DIR)
    parser.add_argument("--landcover-sequence-dir", default=DEFAULT_LANDCOVER_SEQUENCE_DIR)
    parser.add_argument("--landcover-run-dir", default=DEFAULT_LANDCOVER_RUN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _case_summary("building_firstpose_gate", _resolve(args.building_sequence_dir), _resolve(args.building_run_dir)),
        _case_summary("landcover_firstpose_gate", _resolve(args.landcover_sequence_dir), _resolve(args.landcover_run_dir)),
    ]
    payload = {"cases": cases}
    (output_dir / "landcover_vs_building_summary.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "landcover_vs_building_metrics.csv", cases)
    _write_report(output_dir / "landcover_vs_building_report.md", cases)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
