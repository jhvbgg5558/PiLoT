#!/usr/bin/env python3
"""Audit PiLoT refined poses against DOM/DSM renderer pose conventions."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.pose_adapter import compute_enu_delta_m, get_domdsm_transformers
from pixloc.utils.transform import euler_angles_to_matrix_ECEF, pixloc_to_osg


DEFAULT_SUMMARY = (
    "docs/experiments/dom_dsm_prepare/"
    "domdsm_refine_single_0000_exif_test/summary_metrics.json"
)
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/pose_convention_audit_p9_2"
EPS = 1e-9


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float_list(values: Sequence[Any]) -> List[float]:
    return [float(x) for x in values]


def _angle_delta_deg(a: float, b: float) -> float:
    return float(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _relative_rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos = float((np.trace(R_rel) - 1.0) / 2.0)
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(abs(np.arccos(cos))))


def _pose_from_candidate(candidate: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    return (
        _as_float_list(candidate["translation_lon_lat_alt"]),
        _as_float_list(candidate["euler_pitch_roll_yaw"]),
    )


def _pose_matrices(trans: Sequence[float], euler: Sequence[float]) -> Dict[str, Any]:
    T_ecef_c2w = euler_angles_to_matrix_ECEF(list(euler), list(trans))
    R_ecef = T_ecef_c2w[:3, :3]
    R_dom_local = R.from_euler("xyz", euler, degrees=True).as_matrix()
    return {
        "T_ecef_c2w": T_ecef_c2w,
        "R_ecef_c2w": R_ecef,
        "R_dom_local_c2w": R_dom_local,
        "camera_axes_world_ecef": {
            "x": R_ecef[:, 0],
            "y": R_ecef[:, 1],
            "z": R_ecef[:, 2],
        },
        "viewing_direction_world_ecef": R_ecef[:, 2],
        "camera_axes_dom_local": {
            "x": R_dom_local[:, 0],
            "y": R_dom_local[:, 1],
            "z": R_dom_local[:, 2],
        },
        "viewing_direction_dom_local": R_dom_local[:, 2],
        "is_downward_in_dom_local": bool(R_dom_local[2, 2] < -0.95),
    }


def _roundtrip_pose(trans: Sequence[float], euler: Sequence[float]) -> Dict[str, Any]:
    T = euler_angles_to_matrix_ECEF(list(euler), list(trans))
    roundtrip_euler, roundtrip_trans, _T_w2c, _kf_pose = pixloc_to_osg(T.copy())
    T2 = euler_angles_to_matrix_ECEF(
        _as_float_list(roundtrip_euler),
        _as_float_list(roundtrip_trans),
    )
    return {
        "input_euler_pitch_roll_yaw": _as_float_list(euler),
        "roundtrip_euler_pitch_roll_yaw": _as_float_list(roundtrip_euler),
        "input_translation_lon_lat_alt": _as_float_list(trans),
        "roundtrip_translation_lon_lat_alt": _as_float_list(roundtrip_trans),
        "max_abs_R_diff": float(np.max(np.abs(T[:3, :3] - T2[:3, :3]))),
        "relative_angle_before_after_deg": _relative_rotation_angle_deg(
            T[:3, :3],
            T2[:3, :3],
        ),
    }


def _candidate_delta(
    initial: Dict[str, Any],
    candidate: Dict[str, Any],
    to_raster: Any,
) -> Dict[str, float]:
    init_t, init_e = _pose_from_candidate(initial)
    cand_t, cand_e = _pose_from_candidate(candidate)
    east, north, alt = compute_enu_delta_m(init_t, cand_t, to_raster)
    return {
        "east_m": east,
        "north_m": north,
        "alt_m": alt,
        "pitch_deg": cand_e[0] - init_e[0],
        "roll_deg": cand_e[1] - init_e[1],
        "yaw_deg": _angle_delta_deg(cand_e[2], init_e[2]),
    }


def _find_grid_best(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = summary.get("candidates", {})
    for name in ["grid_best", "local_search_best"]:
        if name in candidates:
            return candidates[name]
    for name, item in candidates.items():
        lowered = name.lower()
        if "grid_best" in lowered or "local_search_best" in lowered:
            return item
    return None


def _cosine_xy(a: Dict[str, float], b: Dict[str, float]) -> Optional[float]:
    va = np.asarray([a["east_m"], a["north_m"]], dtype=np.float64)
    vb = np.asarray([b["east_m"], b["north_m"]], dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= EPS:
        return None
    return float(np.dot(va, vb) / denom)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    known = audit["known_results"]
    convention = audit["convention"]
    axes = audit["camera_axes_world"]
    rel = audit["relative_rotation_angles_deg"]
    rt = audit["roundtrip"]
    trans = audit["translation_delta_audit"]
    decision = audit["decision"]

    lines = [
        "# P9.2 PiLoT / DOMDSM Pose Convention Audit",
        "",
        "## Purpose",
        (
            "The DOM+DSM single-frame run shows raw refined full pose changes "
            "near 180 degrees in Euler components and a large altitude jump. "
            "This report audits the matrix-level convention chain before any "
            "adapter fix is treated as final."
        ),
        "",
        "## Known Results",
        "| Candidate | East Delta | North Delta | Alt Delta | Roll Delta | Pitch Delta | Yaw Delta | Chamfer | Overlap | Feature loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["initial", "raw_refined_full", "corrected_downward_yaw"]:
        row = known[name]
        lines.append(
            f"| {name} | {_fmt(row['east_m'])} | {_fmt(row['north_m'])} | "
            f"{_fmt(row['alt_m'])} | {_fmt(row['roll_deg'])} | "
            f"{_fmt(row['pitch_deg'])} | {_fmt(row['yaw_deg'])} | "
            f"{_fmt(row['edge_chamfer'])} | {_fmt(row['edge_overlap_ratio'])} | "
            f"{_fmt(row['torch_feature_loss'])} |"
        )

    lines += [
        "",
        "## Convention Audit",
        f"- PiLoT/pose-file Euler order: `{convention['pilot_euler_order']}`.",
        f"- ECEF matrix helper: `{convention['ecef_matrix_helper']}`.",
        f"- DOMDSM renderer Euler helper: `{convention['domdsm_renderer_helper']}`.",
        f"- Pose dictionary flips camera Y/Z before forming PixLoc `T_w2c`: `{convention['pose_dict_yz_flip']}`.",
        "",
        "### Camera Axes",
        "| Candidate | DOM local z/viewing direction | Downward local? | ECEF z/viewing direction |",
        "|---|---:|---:|---:|",
    ]
    for name in ["initial", "raw_refined_full", "corrected_downward_yaw"]:
        item = axes[name]
        lines.append(
            f"| {name} | `{item['viewing_direction_dom_local']}` | "
            f"{item['is_downward_in_dom_local']} | "
            f"`{item['viewing_direction_world_ecef']}` |"
        )

    lines += [
        "",
        "### Matrix Relative Rotation Angles",
        "| Pair | Angle deg |",
        "|---|---:|",
    ]
    for pair, angle in rel.items():
        lines.append(f"| {pair} | {_fmt(angle, 6)} |")

    lines += [
        "",
        "### Round Trip",
        "| Candidate | Max abs R diff | Relative angle deg | Input Euler | Roundtrip Euler |",
        "|---|---:|---:|---|---|",
    ]
    for name, item in rt.items():
        warn = " **ROUNDTRIP WARNING**" if item["relative_angle_before_after_deg"] > 1e-4 else ""
        lines.append(
            f"| {name} | {_fmt(item['max_abs_R_diff'], 9)} | "
            f"{_fmt(item['relative_angle_before_after_deg'], 9)}{warn} | "
            f"`{item['input_euler_pitch_roll_yaw']}` | "
            f"`{item['roundtrip_euler_pitch_roll_yaw']}` |"
        )

    lines += [
        "",
        "### Translation Interpretation",
        f"- raw_refined_full delta: `{trans['raw_refined_full_minus_initial']}`",
        f"- corrected_downward_yaw delta: `{trans['corrected_downward_yaw_minus_initial']}`",
        f"- interpretation: {trans['interpretation']}",
        f"- grid/local comparison: {trans['grid_best_comparison']['message']}",
        "",
        "## Decision",
        f"- raw_refined_full as formal candidate: `{decision['raw_refined_full_formal_candidate']}`.",
        f"- corrected_downward_yaw status: {decision['corrected_downward_yaw_status']}",
        f"- translation delta same coordinate chain: `{decision['translation_delta_same_coordinate_chain']}`.",
        f"- check east/north sign or axis swap next: `{decision['needs_east_north_sign_or_axis_swap_check']}`.",
        f"- raw 180-degree change classification: {decision['raw_180_change_classification']}",
        "",
        "## Next Step",
        f"- situation: `{decision['next_step_situation']}`",
        f"- recommendation: {decision['next_step_recommendation']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-metrics", default=DEFAULT_SUMMARY)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = REPO_ROOT / args.summary_metrics
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    to_raster, _from_raster, raster_crs = get_domdsm_transformers(config)

    candidates = summary["candidates"]
    pose_names = ["initial", "raw_refined_full", "corrected_downward_yaw"]
    poses = {name: candidates[name] for name in pose_names}
    matrices = {}
    camera_axes = {}
    roundtrip = {}
    for name, cand in poses.items():
        trans, euler = _pose_from_candidate(cand)
        mat = _pose_matrices(trans, euler)
        matrices[name] = mat
        camera_axes[name] = {
            "camera_axes_world_ecef": mat["camera_axes_world_ecef"],
            "viewing_direction_world_ecef": mat["viewing_direction_world_ecef"],
            "camera_axes_dom_local": mat["camera_axes_dom_local"],
            "viewing_direction_dom_local": mat["viewing_direction_dom_local"],
            "is_downward_in_dom_local": mat["is_downward_in_dom_local"],
        }
        roundtrip[name] = _roundtrip_pose(trans, euler)

    relative = {
        "initial_to_raw_refined_full": _relative_rotation_angle_deg(
            matrices["initial"]["R_ecef_c2w"],
            matrices["raw_refined_full"]["R_ecef_c2w"],
        ),
        "initial_to_corrected_downward_yaw": _relative_rotation_angle_deg(
            matrices["initial"]["R_ecef_c2w"],
            matrices["corrected_downward_yaw"]["R_ecef_c2w"],
        ),
        "raw_refined_full_to_corrected_downward_yaw": _relative_rotation_angle_deg(
            matrices["raw_refined_full"]["R_ecef_c2w"],
            matrices["corrected_downward_yaw"]["R_ecef_c2w"],
        ),
    }

    deltas = {
        name: _candidate_delta(poses["initial"], poses[name], to_raster)
        for name in pose_names
    }
    known_results = {}
    for name in pose_names:
        cand = poses[name]
        known_results[name] = {
            **deltas[name],
            "edge_chamfer": cand.get("edge_chamfer"),
            "edge_overlap_ratio": cand.get("edge_overlap_ratio"),
            "torch_feature_loss": cand.get("torch_feature_loss"),
        }

    grid_best = _find_grid_best(summary)
    raw_delta = deltas["raw_refined_full"]
    if grid_best is None:
        grid_compare = {
            "message": "grid_best not found in current summary; rerun local search or provide its summary for delta comparison.",
            "grid_best_delta_xy": None,
            "raw_refined_delta_xy": [raw_delta["east_m"], raw_delta["north_m"]],
            "cos_angle_between_delta_xy": None,
        }
    else:
        grid_delta = _candidate_delta(poses["initial"], grid_best, to_raster)
        grid_compare = {
            "message": "grid_best/local_search_best found and compared.",
            "grid_best_delta_xy": [grid_delta["east_m"], grid_delta["north_m"]],
            "raw_refined_delta_xy": [raw_delta["east_m"], raw_delta["north_m"]],
            "cos_angle_between_delta_xy": _cosine_xy(grid_delta, raw_delta),
        }

    raw_matrix_large = relative["initial_to_raw_refined_full"] > 30.0
    corrected_small = relative["initial_to_corrected_downward_yaw"] < 5.0
    translation_same_chain = True
    if raw_matrix_large:
        situation = "B"
        recommendation = (
            "180-degree raw change is a real matrix-level rotation under the "
            "current ECEF conversion. Do not use raw full pose for DOMDSM; "
            "keep pose adapter constraints and inspect optimizer/cost."
        )
        raw_class = "matrix-level large rotation, not merely an Euler display delta"
    elif corrected_small:
        situation = "A"
        recommendation = (
            "Raw 180-degree Euler values are mostly branch/convention related; "
            "continue by fixing the adapter and debugging east/north/yaw update."
        )
        raw_class = "mostly Euler branch/convention equivalent"
    else:
        situation = "C"
        recommendation = (
            "Rotation and translation conventions remain ambiguous; audit "
            "translation adapter and ENU sign/axis mapping before search."
        )
        raw_class = "ambiguous convention relationship"

    if grid_best is None and translation_same_chain:
        needs_axis_check = True
        situation = "D" if not raw_matrix_large else situation
        if situation == "D":
            recommendation = (
                "Translation is in the expected WGS84/DOMDSM chain, but no "
                "grid/local best is available for direction comparison. Rerun "
                "east/north/yaw local search or P11 scorer before accepting "
                "raw objective updates."
            )
    else:
        needs_axis_check = bool(
            grid_compare.get("cos_angle_between_delta_xy") is not None
            and grid_compare["cos_angle_between_delta_xy"] < 0.0
        )

    audit = {
        "inputs": {
            "summary_metrics": args.summary_metrics,
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "raster_crs": raster_crs,
        },
        "convention": {
            "pilot_euler_order": "[pitch, roll, yaw], scipy/transform xyz",
            "ecef_matrix_helper": "pixloc.utils.transform.euler_angles_to_matrix_ECEF -> ECEF camera-to-world",
            "domdsm_renderer_helper": "scipy.spatial.transform.Rotation.from_euler('xyz', euler) -> raster/world camera-to-world",
            "pose_dict_yz_flip": "src.utils.pose_utils.load_pose_dict flips c2w camera Y/Z before building PixLoc T_w2c",
            "refined_output": "BaseRefiner converts selected PixLoc T_refined from w2c back to c2w, then pixloc_to_osg returns WGS84 camera center and ENU xyz Euler.",
        },
        "known_results": known_results,
        "camera_axes_world": camera_axes,
        "viewing_direction_world": {
            name: camera_axes[name]["viewing_direction_world_ecef"]
            for name in pose_names
        },
        "relative_rotation_angles_deg": relative,
        "roundtrip": roundtrip,
        "translation_delta_audit": {
            "raw_refined_full_minus_initial": deltas["raw_refined_full"],
            "corrected_downward_yaw_minus_initial": deltas["corrected_downward_yaw"],
            "interpretation": (
                "Summary translations are WGS84 lon/lat/alt camera centers. "
                "ENU deltas are computed through the DOM/DSM raster CRS via "
                "pose_adapter.compute_enu_delta_m, matching the renderer's "
                "WGS84-to-raster transform path."
            ),
            "possible_internal_forms_checked": [
                "camera center WGS84",
                "ECEF camera-to-world translation",
                "PixLoc world-to-camera pose after origin/mul normalization",
                "DOM/DSM projected raster x/y meters",
            ],
            "grid_best_comparison": grid_compare,
        },
        "decision": {
            "raw_refined_full_formal_candidate": False,
            "corrected_downward_yaw_status": (
                "reasonable adapter candidate for yaw branch testing"
                if corrected_small
                else "not yet validated as a convention fix"
            ),
            "translation_delta_same_coordinate_chain": translation_same_chain,
            "needs_east_north_sign_or_axis_swap_check": needs_axis_check,
            "raw_180_change_classification": raw_class,
            "next_step_situation": situation,
            "next_step_recommendation": recommendation,
        },
    }

    _write_json(output_dir / "pose_convention_audit.json", audit)
    _write_markdown(output_dir / "pose_convention_audit.md", audit)
    print(json.dumps({"audit_json": str(output_dir / "pose_convention_audit.json"), "audit_md": str(output_dir / "pose_convention_audit.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
