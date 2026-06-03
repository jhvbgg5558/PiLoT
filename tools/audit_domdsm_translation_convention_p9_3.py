#!/usr/bin/env python3
"""Audit simple DOM/DSM translation convention transforms for raw refined delta."""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset  # noqa: E402
from tools.run_raw_refine_diagnostics_exif_test_0000 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_POSE_FILE,
    DEFAULT_QUERY_IMAGE,
    DEFAULT_SOURCE_EXPERIMENT,
    _cosine,
    _load_json,
    _norm,
    _read_source_metrics,
    _render_candidate,
    _safe_jsonable,
    _setup_render_context,
    _write_csv,
    _write_json,
)


DEFAULT_P9_2_DIR = "docs/experiments/dom_dsm_prepare/raw_refine_diagnostics_exif_test_0000"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/p9_3_translation_convention_audit_0000"
DEFAULT_WIDTH = 512
XY_TRANSFORMS = {
    "identity": lambda e, n: (e, n),
    "flip_east": lambda e, n: (-e, n),
    "flip_north": lambda e, n: (e, -n),
    "flip_both": lambda e, n: (-e, -n),
    "swap": lambda e, n: (n, e),
    "neg_swap": lambda e, n: (-n, -e),
}
ALT_MODES = {
    "raw_alt": lambda a: a,
    "neg_alt": lambda a: -a,
    "freeze_alt": lambda _a: 0.0,
    "small_alt": lambda a: 0.25 * a,
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _row_by_candidate(rows: Sequence[Dict[str, str]], candidate: str) -> Dict[str, str]:
    for row in rows:
        if row.get("candidate") == candidate:
            return row
    raise KeyError(f"candidate not found in delta comparison: {candidate}")


def _axis_match(a: float, b: float) -> bool:
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return abs(a) < 1e-9 and abs(b) < 1e-9
    return bool(np.sign(a) == np.sign(b))


def _candidate_name(xy_transform: str, alt_mode: str) -> str:
    return f"{xy_transform}__{alt_mode}"


def _load_p9_2_inputs(p9_2_dir: Path) -> Dict[str, Any]:
    line_search_path = p9_2_dir / "experiment_2_raw_delta_line_search" / "line_search.csv"
    delta_comparison_path = p9_2_dir / "experiment_3_local_best_vs_raw_delta" / "delta_comparison.csv"
    if not line_search_path.exists():
        raise FileNotFoundError(line_search_path)
    if not delta_comparison_path.exists():
        raise FileNotFoundError(delta_comparison_path)
    line_rows = _read_csv(line_search_path)
    delta_rows = _read_csv(delta_comparison_path)
    raw = _row_by_candidate(delta_rows, "raw_translation_initial_rotation")
    p13 = _row_by_candidate(delta_rows, "p13_selected")
    p12 = _row_by_candidate(delta_rows, "p12_selected")
    initial_line = next(row for row in line_rows if abs(_float(row, "scale")) < 1e-9)
    return {
        "line_search_path": os.fspath(line_search_path),
        "delta_comparison_path": os.fspath(delta_comparison_path),
        "line_rows": line_rows,
        "delta_rows": delta_rows,
        "raw_delta": [_float(raw, "east_m"), _float(raw, "north_m"), _float(raw, "alt_m")],
        "p13": p13,
        "p12": p12,
        "initial_line": initial_line,
    }


def _make_row(
    xy_transform: str,
    alt_mode: str,
    offsets: Sequence[float],
    metrics: Dict[str, Any],
    initial: Dict[str, Any],
    p12: Dict[str, str],
    p13: Dict[str, str],
) -> Dict[str, Any]:
    vec3 = [float(offsets[0]), float(offsets[1]), float(offsets[2])]
    vec2 = vec3[:2]
    p13_vec = [_float(p13, "east_m"), _float(p13, "north_m"), _float(p13, "alt_m")]
    p12_vec = [_float(p12, "east_m"), _float(p12, "north_m"), _float(p12, "alt_m")]
    return {
        "candidate": _candidate_name(xy_transform, alt_mode),
        "xy_transform": xy_transform,
        "alt_mode": alt_mode,
        "east_m": vec3[0],
        "north_m": vec3[1],
        "alt_m": vec3[2],
        "delta_norm_xy_m": _norm(vec2),
        "delta_norm_3d_m": _norm(vec3),
        "cosine_vs_p13_xy": _cosine(vec2, p13_vec[:2]),
        "cosine_vs_p12_xy": _cosine(vec2, p12_vec[:2]),
        "distance_to_p13_xy_m": _norm([vec3[0] - p13_vec[0], vec3[1] - p13_vec[1]]),
        "distance_to_p12_xy_m": _norm([vec3[0] - p12_vec[0], vec3[1] - p12_vec[1]]),
        "distance_to_p13_3d_m": _norm([vec3[i] - p13_vec[i] for i in range(3)]),
        "distance_to_p12_3d_m": _norm([vec3[i] - p12_vec[i] for i in range(3)]),
        "axis_sign_east_matches_p13": _axis_match(vec3[0], p13_vec[0]),
        "axis_sign_north_matches_p13": _axis_match(vec3[1], p13_vec[1]),
        "axis_sign_alt_matches_p13": _axis_match(vec3[2], p13_vec[2]),
        "axis_sign_east_matches_p12": _axis_match(vec3[0], p12_vec[0]),
        "axis_sign_north_matches_p12": _axis_match(vec3[1], p12_vec[1]),
        "axis_sign_alt_matches_p12": _axis_match(vec3[2], p12_vec[2]),
        "chamfer": metrics["edge_chamfer"],
        "overlap": metrics["edge_overlap_ratio"],
        "valid_depth_ratio": metrics["valid_depth_ratio"],
        "beats_initial_chamfer": float(metrics["edge_chamfer"]) < float(initial["edge_chamfer"]),
        "beats_initial_overlap": float(metrics["edge_overlap_ratio"]) > float(initial["edge_overlap_ratio"]),
        "objective_chamfer_minus_overlap": float(metrics["edge_chamfer"]) - float(metrics["edge_overlap_ratio"]),
    }


def _passes_adapter_rule(row: Dict[str, Any], initial: Dict[str, Any], p12: Dict[str, str], p13: Dict[str, str]) -> Tuple[bool, List[str]]:
    reasons = []
    beats_initial = bool(row["beats_initial_chamfer"] and row["beats_initial_overlap"])
    if not beats_initial:
        reasons.append("does_not_beat_initial_on_both_visual_metrics")
    p12_chamfer = _float(p12, "chamfer")
    p13_chamfer = _float(p13, "chamfer")
    comparable_visual = float(row["chamfer"]) <= p12_chamfer or float(row["chamfer"]) <= 1.1 * p13_chamfer
    if not comparable_visual:
        reasons.append("not_as_good_as_p12_or_within_10pct_of_p13_chamfer")
    direction_distance = (
        float(row["cosine_vs_p13_xy"]) > 0.5
        or float(row["cosine_vs_p12_xy"]) > 0.5
        or float(row["distance_to_p13_xy_m"]) <= 3.0
        or float(row["distance_to_p12_xy_m"]) <= 3.0
    )
    if not direction_distance:
        reasons.append("not_direction_or_distance_consistent_with_local_best")
    return beats_initial and comparable_visual and direction_distance, reasons


def _copy_top_images(out_dir: Path, row: Dict[str, Any], subdir: str) -> None:
    src_dir = out_dir / row["candidate"]
    dst_dir = out_dir / subdir
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["rendered_rgb.png", "overlay.png", "edge_overlay.png", "checkerboard.png", "metrics.json"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    _write_json(dst_dir / "selected_candidate.json", row)


def _draw_delta_vectors(rows: Sequence[Dict[str, Any]], p12: Dict[str, str], p13: Dict[str, str], output_path: Path) -> None:
    canvas = np.full((820, 900, 3), 255, dtype=np.uint8)
    center = np.array([450, 410], dtype=np.float64)
    vectors = [(row["candidate"], row["east_m"], row["north_m"], (120, 120, 120)) for row in rows]
    vectors.extend(
        [
            ("p13_selected", _float(p13, "east_m"), _float(p13, "north_m"), (40, 170, 40)),
            ("p12_selected", _float(p12, "east_m"), _float(p12, "north_m"), (210, 120, 40)),
        ]
    )
    max_abs = max(max(abs(float(e)), abs(float(n))) for _name, e, n, _color in vectors)
    scale = 310.0 / max(max_abs, 1.0)
    cv2.line(canvas, (90, 410), (810, 410), (190, 190, 190), 1)
    cv2.line(canvas, (450, 70), (450, 750), (190, 190, 190), 1)
    cv2.putText(canvas, "East +", (740, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.putText(canvas, "North +", (465, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 70), 2, cv2.LINE_AA)
    for row in rows:
        color = (80, 80, 220) if row["xy_transform"] in {"identity", "flip_both"} else (120, 120, 120)
        end = center + np.array([float(row["east_m"]) * scale, -float(row["north_m"]) * scale])
        cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(end.astype(int)), color, 1, tipLength=0.08)
    for name, east, north, color in vectors[-2:]:
        end = center + np.array([float(east) * scale, -float(north) * scale])
        cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(end.astype(int)), color, 4, tipLength=0.12)
    cv2.putText(canvas, "P13 green, P12 orange, transform candidates gray/red", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.imwrite(os.fspath(output_path), canvas)


def _draw_metric_heatmap(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    xy_names = list(XY_TRANSFORMS.keys())
    alt_names = list(ALT_MODES.keys())
    cell_w, cell_h = 190, 88
    canvas = np.full((110 + len(xy_names) * cell_h, 170 + len(alt_names) * cell_w, 3), 255, dtype=np.uint8)
    chamfers = np.asarray([float(row["chamfer"]) for row in rows], dtype=np.float64)
    overlaps = np.asarray([float(row["overlap"]) for row in rows], dtype=np.float64)
    cmin, cmax = chamfers.min(), chamfers.max()
    omin, omax = overlaps.min(), overlaps.max()
    row_map = {(row["xy_transform"], row["alt_mode"]): row for row in rows}
    for j, alt in enumerate(alt_names):
        x = 150 + j * cell_w
        cv2.putText(canvas, alt, (x + 8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 2, cv2.LINE_AA)
    for i, xy in enumerate(xy_names):
        y = 90 + i * cell_h
        cv2.putText(canvas, xy, (12, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 2, cv2.LINE_AA)
        for j, alt in enumerate(alt_names):
            row = row_map[(xy, alt)]
            x = 150 + j * cell_w
            c_norm = (float(row["chamfer"]) - cmin) / max(cmax - cmin, 1e-9)
            o_norm = (float(row["overlap"]) - omin) / max(omax - omin, 1e-9)
            color = (
                int(80 + 150 * c_norm),
                int(80 + 150 * o_norm),
                int(230 - 120 * c_norm),
            )
            cv2.rectangle(canvas, (x, y), (x + cell_w - 8, y + cell_h - 8), color, -1)
            text = f"c {float(row['chamfer']):.2f} o {float(row['overlap']):.2f}"
            cv2.putText(canvas, text, (x + 8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.fspath(output_path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--p9-2-dir", default=DEFAULT_P9_2_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = {
        "config": args.config,
        "query_image": args.query_image,
        "pose_file": args.pose_file,
        "source_experiment": args.source_experiment,
        "p9_2_dir": args.p9_2_dir,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
    started = time.perf_counter()
    try:
        run_log["failure_stage"] = "load_inputs"
        p9_2 = _load_p9_2_inputs(REPO_ROOT / args.p9_2_dir)
        source_metrics = _read_source_metrics(REPO_ROOT / args.source_experiment)
        initial = source_metrics["initial"]
        ctx = _setup_render_context(args)
        raw_e, raw_n, raw_a = p9_2["raw_delta"]
        rows: List[Dict[str, Any]] = []
        initial_trans = [float(x) for x in initial["translation_lon_lat_alt"]]
        initial_euler = [float(x) for x in initial["euler_pitch_roll_yaw"]]

        run_log["failure_stage"] = "render_candidates"
        for xy_name, xy_fn in XY_TRANSFORMS.items():
            east, north = xy_fn(raw_e, raw_n)
            for alt_name, alt_fn in ALT_MODES.items():
                alt = alt_fn(raw_a)
                name = _candidate_name(xy_name, alt_name)
                trans = apply_enu_offset(
                    initial_trans,
                    east,
                    north,
                    alt,
                    ctx["to_raster"],
                    ctx["from_raster"],
                )
                metrics = _render_candidate(
                    output_dir,
                    name,
                    ctx["renderer"],
                    ctx["query_rgb"],
                    trans,
                    initial_euler,
                    args.checker_tile,
                    {
                        "xy_transform": xy_name,
                        "alt_mode": alt_name,
                        "east_m": float(east),
                        "north_m": float(north),
                        "alt_m": float(alt),
                        "source": "p9_3_simple_translation_transform",
                    },
                )
                rows.append(_make_row(xy_name, alt_name, [east, north, alt], metrics, initial, p9_2["p12"], p9_2["p13"]))

        run_log["failure_stage"] = "summarize"
        fields = [
            "candidate",
            "xy_transform",
            "alt_mode",
            "east_m",
            "north_m",
            "alt_m",
            "delta_norm_xy_m",
            "delta_norm_3d_m",
            "cosine_vs_p13_xy",
            "cosine_vs_p12_xy",
            "distance_to_p13_xy_m",
            "distance_to_p12_xy_m",
            "distance_to_p13_3d_m",
            "distance_to_p12_3d_m",
            "axis_sign_east_matches_p13",
            "axis_sign_north_matches_p13",
            "axis_sign_alt_matches_p13",
            "axis_sign_east_matches_p12",
            "axis_sign_north_matches_p12",
            "axis_sign_alt_matches_p12",
            "chamfer",
            "overlap",
            "valid_depth_ratio",
            "beats_initial_chamfer",
            "beats_initial_overlap",
            "objective_chamfer_minus_overlap",
        ]
        _write_csv(output_dir / "transform_candidates.csv", rows, fields)
        _write_json(output_dir / "transform_candidates.json", {"rows": rows})

        best_by_chamfer = min(rows, key=lambda row: float(row["chamfer"]))
        best_by_overlap = max(rows, key=lambda row: float(row["overlap"]))
        closest_to_p13 = min(rows, key=lambda row: float(row["distance_to_p13_xy_m"]))
        closest_to_p12 = min(rows, key=lambda row: float(row["distance_to_p12_xy_m"]))
        passing = []
        failed_rule_summary = {}
        for row in rows:
            ok, reasons = _passes_adapter_rule(row, initial, p9_2["p12"], p9_2["p13"])
            if ok:
                passing.append(row)
            failed_rule_summary[row["candidate"]] = reasons
        recommended = min(passing, key=lambda row: float(row["objective_chamfer_minus_overlap"])) if passing else None
        summary = {
            "experiment": "p9_3_domdsm_translation_convention_audit",
            "query_image": args.query_image,
            "config": args.config,
            "pose_file": args.pose_file,
            "source_experiment": args.source_experiment,
            "p9_2_inputs": {
                "line_search_csv": p9_2["line_search_path"],
                "delta_comparison_csv": p9_2["delta_comparison_path"],
            },
            "raw_delta_east_north_alt_m": p9_2["raw_delta"],
            "initial_metrics": {
                "chamfer": initial["edge_chamfer"],
                "overlap": initial["edge_overlap_ratio"],
            },
            "local_best_reference": {
                "p13": p9_2["p13"],
                "p12": p9_2["p12"],
            },
            "num_candidates": len(rows),
            "best_by_chamfer": best_by_chamfer,
            "best_by_overlap": best_by_overlap,
            "closest_to_p13": closest_to_p13,
            "closest_to_p12": closest_to_p12,
            "recommend_adapter": recommended is not None,
            "recommended_transform": recommended,
            "diagnosis": "simple_translation_transform_candidate_found" if recommended else "objective_mismatch_not_simple_translation_convention",
            "failed_rule_summary": failed_rule_summary,
            "sanity_checks": {
                "expected_candidate_count": 24,
                "candidate_count_ok": len(rows) == 24,
                "identity_raw_alt_chamfer": next(row["chamfer"] for row in rows if row["candidate"] == "identity__raw_alt"),
                "p9_2_raw_translation_initial_rotation_chamfer": source_metrics["raw_refined_translation_initial_rotation"]["edge_chamfer"],
                "identity_raw_alt_overlap": next(row["overlap"] for row in rows if row["candidate"] == "identity__raw_alt"),
                "p9_2_raw_translation_initial_rotation_overlap": source_metrics["raw_refined_translation_initial_rotation"]["edge_overlap_ratio"],
            },
            "total_time_sec": time.perf_counter() - started,
        }
        _write_json(output_dir / "summary.json", summary)
        _copy_top_images(output_dir, best_by_chamfer, "top_by_chamfer")
        _copy_top_images(output_dir, best_by_overlap, "top_by_overlap")
        closest_local = min(rows, key=lambda row: min(float(row["distance_to_p13_xy_m"]), float(row["distance_to_p12_xy_m"])))
        _copy_top_images(output_dir, closest_local, "top_by_local_delta_distance")
        _draw_delta_vectors(rows, p9_2["p12"], p9_2["p13"], output_dir / "delta_vector_comparison.png")
        _draw_metric_heatmap(rows, output_dir / "transform_metric_heatmap.png")
        _write_json(output_dir / "run_log.json", {**run_log, "failure_stage": None, "total_time_sec": summary["total_time_sec"]})
        print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
        return 0
    except Exception:
        run_log["traceback"] = traceback.format_exc()
        run_log["total_time_sec"] = time.perf_counter() - started
        _write_json(output_dir / "run_log.json", run_log)
        print(run_log["traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
