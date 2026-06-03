#!/usr/bin/env python3
"""Run PiLoT refinement on a synthetic DOM/DSM flight sequence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.localization.tracker import SimpleTracker
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Camera
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import compute_enu_delta_m, domxy_to_wgs84, get_domdsm_transformers, normalize_angle_deg
from pixloc.utils.get_depth import generate_render_camera, pad_to_multiple, sample_3d_points
from pixloc.utils.transform import WGS84_to_ECEF, euler_angles_to_matrix_ECEF


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_SEQUENCE_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_firstpose_gate"
DEFAULT_DOC = "docs/experiments/dom_dsm_prepare/sim_sequence_pilot_p11_check.md"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
INIT_POSE_POLICIES = {"first_pose_odom", "per_frame_file"}


def angular_diff_deg(a: float, b: float) -> float:
    return abs(normalize_angle_deg(float(a) - float(b)))


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_jsonable(value.tolist())
    if torch.is_tensor(value):
        return _safe_jsonable(value.detach().cpu().numpy())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_id",
        "gt_east",
        "gt_north",
        "gt_alt",
        "gt_yaw",
        "init_east",
        "init_north",
        "init_alt",
        "init_yaw",
        "refined_east",
        "refined_north",
        "refined_alt",
        "refined_yaw",
        "selected_method",
        "init_xy_error_m",
        "refined_xy_error_m",
        "selected_xy_error_m",
        "init_alt_error_m",
        "refined_alt_error_m",
        "selected_alt_error_m",
        "init_yaw_error_deg",
        "refined_yaw_error_deg",
        "selected_yaw_error_deg",
        "init_chamfer",
        "refined_chamfer",
        "gt_chamfer",
        "init_overlap",
        "refined_overlap",
        "gt_overlap",
        "feature_loss_initial",
        "feature_loss_refined",
        "feature_loss_drop",
        "safe_gate_accepted",
        "refine_success",
        "renderer_backend_init",
        "renderer_backend_refined",
        "renderer_backend_gt",
    ]
    extra = sorted({k for row in rows for k in row.keys()} - set(fields))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_pose_file(path: Path) -> Dict[str, Dict[str, Any]]:
    poses: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Invalid pose line in {path}: {line}")
        name = parts[0]
        lon, lat, alt, roll, pitch, yaw = map(float, parts[1:7])
        poses[name] = {
            "name": name,
            "trans": [lon, lat, alt],
            "euler": [pitch, roll, normalize_angle_deg(yaw)],
            "roll_pitch_yaw": [roll, pitch, normalize_angle_deg(yaw)],
        }
    return poses


def _load_odom_edges(path: Path) -> Dict[str, Dict[str, Any]]:
    edges: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return edges
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"Invalid odom edge line in {path}: {line}")
        src, dst = parts[0], parts[1]
        dx, dy, dz, dyaw, dist = map(float, parts[2:7])
        edges[src] = {
            "src": src,
            "dst": dst,
            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
            "dyaw_deg": dyaw,
            "distance_m": dist,
        }
    return edges


def _scale_camera(config: Dict[str, Any], width: int, height: Optional[int]) -> Tuple[Dict[str, Any], np.ndarray]:
    cam = deepcopy(config["default_confs"]["cam_query"])
    src_w = float(cam["width"])
    src_h = float(cam["height"])
    if height is None:
        height = int(round(float(width) * src_h / src_w))
    fx, fy, cx, cy = [float(x) for x in cam["params"]]
    sx = float(width) / src_w
    sy = float(height) / src_h
    cam["width"] = int(width)
    cam["height"] = int(height)
    cam["max_size"] = int(width)
    cam["params"] = np.asarray([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float64)
    render_camera = np.asarray([width, height, cx * sx, cy * sy, fx * sx, fy * sy], dtype=np.float64)
    return cam, render_camera


def _prepare_render_config(config: Dict[str, Any], render_camera: np.ndarray, output_dir: Path) -> Dict[str, Any]:
    render_config = deepcopy(config["render_config"])
    render_config["render_camera"] = render_camera
    render_config.setdefault("dom_dsm", {})
    render_config["dom_dsm"]["debug_dir"] = os.fspath(output_dir / "render_debug")
    render_config["dom_dsm"]["debug_every"] = 0
    return render_config


def _edge_metrics(query_rgb: np.ndarray, render_rgb: np.ndarray) -> Dict[str, float]:
    if query_rgb.shape[:2] != render_rgb.shape[:2]:
        render_rgb = cv2.resize(render_rgb, (query_rgb.shape[1], query_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    q_gray = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY) if query_rgb.ndim == 3 else query_rgb
    r_gray = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY) if render_rgb.ndim == 3 else render_rgb
    q_edges = cv2.Canny(q_gray, 80, 160)
    r_edges = cv2.Canny(r_gray, 80, 160)
    q_count = int(np.count_nonzero(q_edges))
    r_count = int(np.count_nonzero(r_edges))
    if q_count == 0 or r_count == 0:
        return {
            "chamfer": float("inf"),
            "overlap": 0.0,
            "query_edge_count": q_count,
            "render_edge_count": r_count,
        }
    inv_r = (r_edges == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv_r, cv2.DIST_L2, 3)
    chamfer = float(dist[q_edges > 0].mean())
    kernel = np.ones((3, 3), dtype=np.uint8)
    r_dilated = cv2.dilate(r_edges, kernel, iterations=1)
    overlap_count = int(np.count_nonzero((q_edges > 0) & (r_dilated > 0)))
    overlap = float(overlap_count / max(q_count, 1))
    return {
        "chamfer": chamfer,
        "overlap": overlap,
        "query_edge_count": q_count,
        "render_edge_count": r_count,
        "overlap_count": overlap_count,
    }


def _pose_xy_alt_yaw(pose: Dict[str, Any], to_raster: Any) -> Tuple[float, float, float, float]:
    lon, lat, alt = pose["trans"]
    east, north, _ = compute_enu_delta_m([lon, lat, alt], [lon, lat, alt], to_raster)
    del east, north
    x, y = to_raster.transform(float(lon), float(lat))
    return float(x), float(y), float(alt), float(pose["euler"][2])


def _pose_error(gt: Dict[str, Any], pred: Dict[str, Any], to_raster: Any) -> Dict[str, float]:
    de, dn, da = compute_enu_delta_m(gt["trans"], pred["trans"], to_raster)
    yaw_err = angular_diff_deg(float(pred["euler"][2]), float(gt["euler"][2]))
    return {
        "xy_error_m": float(math.hypot(de, dn)),
        "alt_error_m": float(abs(da)),
        "yaw_error_deg": float(yaw_err),
    }


def _pose_from_raster_xy(name: str, x: float, y: float, alt: float, yaw: float, from_raster: Any) -> Dict[str, Any]:
    lon, lat, _ = domxy_to_wgs84([float(x), float(y), float(alt)], from_raster)
    yaw = normalize_angle_deg(yaw)
    return {
        "name": name,
        "trans": [float(lon), float(lat), float(alt)],
        "euler": [0.0, 180.0, yaw],
    }


def _pose_with_yaw(name: str, pose: Dict[str, Any], yaw: float) -> Dict[str, Any]:
    return {
        "name": name,
        "trans": [float(x) for x in pose["trans"]],
        "euler": [0.0, 180.0, normalize_angle_deg(yaw)],
    }


def _temporal_prediction(
    prev_selected: Optional[Dict[str, Any]],
    prev_frame_name: Optional[str],
    current_name: str,
    odom_edges: Dict[str, Dict[str, Any]],
    to_raster: Any,
    from_raster: Any,
) -> Optional[Dict[str, Any]]:
    if prev_selected is None or prev_frame_name is None:
        return None
    edge = odom_edges.get(prev_frame_name)
    if edge is None or edge.get("dst") != current_name:
        return None
    x, y, alt, yaw = _pose_xy_alt_yaw(prev_selected, to_raster)
    return _pose_from_raster_xy(
        current_name,
        x + float(edge["dx_m"]),
        y + float(edge["dy_m"]),
        alt + float(edge["dz_m"]),
        normalize_angle_deg(yaw + float(edge["dyaw_deg"])),
        from_raster,
    )


def _propagate_pose_from_edge(
    pose: Dict[str, Any],
    edge: Dict[str, Any],
    name: str,
    to_raster: Any,
    from_raster: Any,
) -> Dict[str, Any]:
    x, y, alt, yaw = _pose_xy_alt_yaw(pose, to_raster)
    return _pose_from_raster_xy(
        name,
        x + float(edge["dx_m"]),
        y + float(edge["dy_m"]),
        alt + float(edge["dz_m"]),
        normalize_angle_deg(yaw + float(edge["dyaw_deg"])),
        from_raster,
    )


def _build_init_sequence(
    image_paths: Sequence[Path],
    init_poses: Dict[str, Dict[str, Any]],
    odom_edges: Dict[str, Dict[str, Any]],
    policy: str,
    to_raster: Any,
    from_raster: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    if policy not in INIT_POSE_POLICIES:
        raise ValueError(f"Unknown init pose policy: {policy}")
    if not image_paths:
        return {}, {}
    missing = [p.name for p in image_paths if p.name not in init_poses]
    if missing:
        raise KeyError(f"Missing init pose for frames: {missing[:5]}")
    if policy == "per_frame_file":
        return (
            {p.name: deepcopy(init_poses[p.name]) for p in image_paths},
            {p.name: "init_noisy_poses.txt" for p in image_paths},
        )

    first_name = image_paths[0].name
    poses = {first_name: deepcopy(init_poses[first_name])}
    poses[first_name]["name"] = first_name
    sources = {first_name: "first_frame_init_noisy_pose"}
    prev_name = first_name
    for path in image_paths[1:]:
        name = path.name
        edge = odom_edges.get(prev_name)
        if edge is None:
            raise ValueError(f"Missing odom edge from {prev_name} to {name}")
        if edge.get("dst") != name:
            raise ValueError(f"Invalid odom edge from {prev_name}: expected dst {name}, got {edge.get('dst')}")
        poses[name] = _propagate_pose_from_edge(poses[prev_name], edge, name, to_raster, from_raster)
        sources[name] = f"first_pose_odom:{prev_name}->{name}"
        prev_name = name
    return poses, sources


def _sample_points_from_depth(depth: np.ndarray, num_samples: int, rng: np.random.Generator) -> torch.Tensor:
    valid = np.argwhere(np.isfinite(depth) & (depth > 0))
    if valid.size == 0:
        raise ValueError("No valid depth pixels for back-projection")
    take = min(num_samples, len(valid))
    ids = rng.choice(len(valid), size=take, replace=len(valid) < take)
    ys = valid[ids, 0]
    xs = valid[ids, 1]
    return torch.as_tensor(np.stack([xs, ys], axis=1), dtype=torch.float32, device="cuda")


def _back_project_for_pose(
    depth: np.ndarray,
    euler: Sequence[float],
    trans: Sequence[float],
    render_camera: Any,
    origin: torch.Tensor,
    mul: float,
    rng: np.random.Generator,
    num_samples: int = 500,
) -> Tuple[torch.Tensor, Any, Any, torch.Tensor]:
    points2d = _sample_points_from_depth(depth, num_samples, rng)
    T_c2w = torch.as_tensor(
        euler_angles_to_matrix_ECEF(list(euler), list(trans)),
        device="cuda",
        dtype=torch.float32,
    )
    return sample_3d_points(
        points2d,
        depth,
        T_c2w,
        render_camera,
        list(euler),
        list(trans),
        origin=origin,
        mul=mul,
        is_init_frame=True,
    )


def _feature_losses_from_tracker(tracker: SimpleTracker) -> Tuple[Optional[float], Optional[float]]:
    vals: List[np.ndarray] = []
    for level_costs in tracker.costs:
        for cost in level_costs:
            arr = np.asarray(cost, dtype=np.float64).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
    if not vals:
        return None, None
    initial = float(np.min(vals[0]))
    refined = float(np.min(vals[-1]))
    return initial, refined


def _dense_feature_image_loss(localizer: RenderLocalizer, query_rgb: np.ndarray, render_rgb: np.ndarray, width: int) -> Optional[float]:
    try:
        with torch.no_grad():
            q_img = localizer.refiner.dense_feature_extraction(
                np.ascontiguousarray(_zero_pad_for_loss(query_rgb, width))
            )[0]
            r_img = localizer.refiner.dense_feature_extraction(
                np.ascontiguousarray(_zero_pad_for_loss(render_rgb, width))
            )[0]
            losses: List[float] = []
            for fq, fr in zip(q_img, r_img):
                if localizer.refiner.conf.compute_uncertainty:
                    fq = fq[:-1]
                    fr = fr[:-1]
                if localizer.refiner.conf.normalize_descriptors:
                    fq = torch.nn.functional.normalize(fq, dim=0)
                    fr = torch.nn.functional.normalize(fr, dim=0)
                diff = (fq - fr).float()
                losses.append(float(torch.mean(diff * diff).detach().cpu().item()))
            return float(np.mean(losses)) if losses else None
    except Exception:
        return None


def _zero_pad_for_loss(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    size = max(int(width), h, w)
    padded = np.zeros((size, size) + image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image
    return padded


def _pose_from_ret(ret: Dict[str, Any], fallback: Dict[str, Any], name: str) -> Dict[str, Any]:
    if ret.get("success") and "translation" in ret and "euler_angles" in ret:
        return {
            "name": name,
            "trans": [float(x) for x in ret["translation"]],
            "euler": [float(x) for x in ret["euler_angles"]],
        }
    return deepcopy(fallback)


def _render_metric(renderer: DOMDSMRenderer, query_rgb: np.ndarray, pose: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, float]]:
    color, depth = renderer.render(pose["trans"], pose["euler"])
    meta = deepcopy(getattr(renderer, "last_render_metadata", {}))
    metrics = _edge_metrics(query_rgb, color)
    return color, depth, meta, metrics


def _visual_gate(candidate_visual: Dict[str, float], init_visual: Dict[str, float]) -> Tuple[bool, str]:
    max_chamfer = float(init_visual["chamfer"]) * 1.05 + 0.05
    min_overlap = float(init_visual["overlap"]) - 0.03
    if float(candidate_visual["chamfer"]) > max_chamfer:
        return False, "visual_chamfer_worse"
    if float(candidate_visual["overlap"]) < min_overlap:
        return False, "visual_overlap_worse"
    return True, "ok"


def _pose_gate(candidate_pose: Dict[str, Any], init_pose: Dict[str, Any], to_raster: Any, args: argparse.Namespace) -> Tuple[bool, str]:
    de, dn, da = compute_enu_delta_m(init_pose["trans"], candidate_pose["trans"], to_raster)
    if math.hypot(de, dn) > float(args.max_xy_jump_m):
        return False, "pose_xy_jump"
    if abs(da) > float(args.max_alt_jump_m):
        return False, "pose_alt_jump"
    if angular_diff_deg(candidate_pose["euler"][2], init_pose["euler"][2]) > float(args.max_yaw_jump_deg):
        return False, "pose_yaw_jump"
    return True, "ok"


def _temporal_gate(candidate_pose: Dict[str, Any], predicted_pose: Optional[Dict[str, Any]], to_raster: Any, args: argparse.Namespace) -> Tuple[bool, str]:
    if predicted_pose is None:
        return True, "no_prediction"
    de, dn, _da = compute_enu_delta_m(predicted_pose["trans"], candidate_pose["trans"], to_raster)
    if math.hypot(de, dn) > float(args.temporal_xy_threshold_m):
        return False, "temporal_xy"
    if angular_diff_deg(candidate_pose["euler"][2], predicted_pose["euler"][2]) > float(args.temporal_yaw_threshold_deg):
        return False, "temporal_yaw"
    return True, "ok"


def _candidate_is_selectable(method: str, args: argparse.Namespace) -> bool:
    return method != "temporal_predicted" or bool(args.allow_temporal_candidate_selection)


def _candidate_score(
    candidate: Dict[str, Any],
    init_visual: Dict[str, float],
    init_pose: Dict[str, Any],
    predicted_pose: Optional[Dict[str, Any]],
    to_raster: Any,
) -> float:
    visual = candidate["visual"]
    chamfer_term = float(visual["chamfer"]) / max(float(init_visual["chamfer"]), 1e-6)
    overlap_term = 1.0 - float(visual["overlap"])
    de_i, dn_i, da_i = compute_enu_delta_m(init_pose["trans"], candidate["pose"]["trans"], to_raster)
    pose_term = 0.05 * math.hypot(de_i, dn_i) + 0.02 * abs(da_i)
    yaw_term = 0.02 * angular_diff_deg(candidate["pose"]["euler"][2], init_pose["euler"][2])
    temporal_term = 0.0
    if predicted_pose is not None:
        de_t, dn_t, _da_t = compute_enu_delta_m(predicted_pose["trans"], candidate["pose"]["trans"], to_raster)
        temporal_term = 0.05 * math.hypot(de_t, dn_t) + 0.02 * angular_diff_deg(candidate["pose"]["euler"][2], predicted_pose["euler"][2])
    feature_term = 0.0
    if candidate.get("feature_loss") is not None and candidate.get("init_feature_loss") is not None:
        denom = max(abs(float(candidate["init_feature_loss"])), 1e-9)
        feature_term = 0.05 * max(0.0, (float(candidate["feature_loss"]) - float(candidate["init_feature_loss"])) / denom)
    return float(chamfer_term + overlap_term + pose_term + yaw_term + temporal_term + feature_term)


def _evaluate_candidates(
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    init_pose: Dict[str, Any],
    init_color: np.ndarray,
    init_visual: Dict[str, float],
    refined_pose: Dict[str, Any],
    refined_color: np.ndarray,
    refined_visual: Dict[str, float],
    predicted_pose: Optional[Dict[str, Any]],
    to_raster: Any,
    from_raster: Any,
    args: argparse.Namespace,
    frame_name: str,
    feature_loss_initial: Optional[float],
    feature_loss_refined: Optional[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    refined_x, refined_y, _refined_alt, raw_refined_yaw = _pose_xy_alt_yaw(refined_pose, to_raster)
    _init_x, _init_y, init_alt, init_yaw = _pose_xy_alt_yaw(init_pose, to_raster)
    corrected_yaw = normalize_angle_deg(raw_refined_yaw + 180.0)

    pose_specs: List[Tuple[str, Dict[str, Any], Optional[np.ndarray], Optional[Dict[str, float]], Optional[float]]] = [
        ("init", init_pose, init_color, init_visual, feature_loss_initial),
        ("raw_refined_full", refined_pose, refined_color, refined_visual, feature_loss_refined),
        ("refined_corrected_yaw", _pose_with_yaw(frame_name, refined_pose, corrected_yaw), None, None, None),
        (
            "refined_freeze_alt_corrected_yaw",
            _pose_from_raster_xy(frame_name, refined_x, refined_y, init_alt, corrected_yaw, from_raster),
            None,
            None,
            None,
        ),
        (
            "refined_freeze_alt_init_yaw",
            _pose_from_raster_xy(frame_name, refined_x, refined_y, init_alt, init_yaw, from_raster),
            None,
            None,
            None,
        ),
    ]
    if args.use_temporal_gate and predicted_pose is not None:
        pose_specs.append(("temporal_predicted", predicted_pose, None, None, None))

    candidates: List[Dict[str, Any]] = []
    for method, pose, color, visual, feature_loss in pose_specs:
        if color is None or visual is None:
            color, _depth, meta, visual = _render_metric(renderer, query_rgb, pose)
        else:
            meta = {}
        visual_pass, visual_reason = (True, "init") if method == "init" else _visual_gate(visual, init_visual)
        pose_pass, pose_reason = (True, "disabled")
        if args.use_pose_consistency_gate and method != "init":
            pose_pass, pose_reason = _pose_gate(pose, init_pose, to_raster, args)
        temporal_pass, temporal_reason = (True, "disabled")
        if args.use_temporal_gate and method != "init":
            temporal_pass, temporal_reason = _temporal_gate(pose, predicted_pose, to_raster, args)
        final_pass = bool(visual_pass and pose_pass and temporal_pass)
        cand = {
            "method": method,
            "pose": pose,
            "color": color,
            "visual": visual,
            "renderer_metadata": meta,
            "feature_loss": feature_loss,
            "init_feature_loss": feature_loss_initial,
            "visual_gate_pass": bool(visual_pass),
            "visual_gate_reason": visual_reason,
            "pose_gate_pass": bool(pose_pass),
            "pose_gate_reason": pose_reason,
            "temporal_gate_pass": bool(temporal_pass),
            "temporal_gate_reason": temporal_reason,
            "selectable": _candidate_is_selectable(method, args),
            "final_gate_pass": final_pass,
        }
        cand["score"] = _candidate_score(cand, init_visual, init_pose, predicted_pose, to_raster)
        candidates.append(cand)

    passed = [c for c in candidates if c["final_gate_pass"] and c["selectable"]]
    selected = min(passed, key=lambda c: c["score"]) if passed else candidates[0]
    return selected, candidates


def _make_debug_frame(query: np.ndarray, init: np.ndarray, refined: np.ndarray, selected: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = query.shape[:2]
    panels = []
    for title, img in [("query", query), ("init render", init), ("refined render", refined), ("selected render", selected)]:
        panel = img.copy()
        cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        panels.append(panel)
    canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = panels[0]
    canvas[0:h, w : 2 * w] = panels[1]
    canvas[h : 2 * h, 0:w] = panels[2]
    canvas[h : 2 * h, w : 2 * w] = panels[3]
    cv2.imwrite(os.fspath(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _rmse(vals: Sequence[float]) -> Optional[float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else None


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _corr(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    if len(pairs) < 3:
        return None
    arr = np.asarray(pairs, dtype=np.float64)
    if np.std(arr[:, 0]) < 1e-9 or np.std(arr[:, 1]) < 1e-9:
        return None
    return float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])


def _plot_trajectory(rows: List[Dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(8, 8))
    series = [("gt", "GT"), ("init", "Init"), ("refined", "Refined")]
    if rows and "selected_east" in rows[0]:
        series.append(("selected", "Selected"))
    for prefix, label in series:
        xs = [row[f"{prefix}_east"] for row in rows]
        ys = [row[f"{prefix}_north"] for row in rows]
        plt.plot(xs, ys, marker=".", linewidth=1, markersize=3, label=label)
    plt.axis("equal")
    plt.xlabel("Raster X / Easting (m)")
    plt.ylabel("Raster Y / Northing (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_error_curve(rows: List[Dict[str, Any]], path: Path, key_suffix: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 4))
    x = np.arange(len(rows))
    for prefix, label in [("init", "Init"), ("refined", "Refined"), ("selected", "Selected")]:
        plt.plot(x, [row[f"{prefix}_{key_suffix}"] for row in rows], label=label)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_visual_curve(rows: List[Dict[str, Any]], path: Path) -> None:
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, [row["init_chamfer"] for row in rows], label="Init")
    axes[0].plot(x, [row["refined_chamfer"] for row in rows], label="Refined")
    axes[0].plot(x, [row.get("gt_chamfer", np.nan) for row in rows], label="GT")
    axes[0].set_ylabel("Chamfer")
    axes[0].legend()
    axes[1].plot(x, [row["init_overlap"] for row in rows], label="Init")
    axes[1].plot(x, [row["refined_overlap"] for row in rows], label="Refined")
    axes[1].plot(x, [row.get("gt_overlap", np.nan) for row in rows], label="GT")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Overlap")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(path: Path, sequence_meta: Dict[str, Any], summary: Dict[str, Any], sanity: Dict[str, Any]) -> None:
    if "selected_v2_xy_rmse" in summary:
        lines = [
            "# P11.3 Pose-Consistency and Temporal Safe Gate Check",
            "",
            "## Questions",
            f"1. P11.2 selected yaw RMSE issue: raw refined yaw RMSE is `{summary.get('raw_refined_yaw_rmse')}`, corrected refined yaw RMSE is `{summary.get('corrected_refined_yaw_rmse')}`, so this run distinguishes yaw convention from true pose jumps.",
            f"2. Corrected refined yaw improvement: `{summary.get('corrected_refined_yaw_rmse')}` vs raw `{summary.get('raw_refined_yaw_rmse')}`.",
            f"3. Pose gate accept ratio: `{summary.get('pose_gate_accept_ratio')}`.",
            f"4. Temporal gate accept ratio: `{summary.get('temporal_gate_accept_ratio')}`.",
            f"5. selected_v2 XY RMSE: `{summary.get('selected_v2_xy_rmse')}`, init `{summary.get('init_xy_rmse')}`, old selected `{summary.get('old_selected_xy_rmse')}`.",
            f"6. selected_v2 yaw RMSE: `{summary.get('selected_v2_yaw_rmse')}`, old selected yaw RMSE `{summary.get('old_selected_yaw_rmse')}`.",
            f"7. Recommendation: {summary.get('recommended_next_step')}",
            "",
            "## Interpretation",
            str(summary.get("interpretation")),
            "",
            "## Renderer Note",
            f"- Requested renderer: `{sequence_meta.get('renderer', {}).get('backend')}`.",
            "- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.3.",
            "",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines = [
        "# P11.2 Synthetic Sequence PiLoT Refinement Check",
        "",
        "## P11.1 Sequence",
        f"- Sequence frames: `{sequence_meta.get('num_frames')}`",
        f"- Renderer backend requested: `{sequence_meta.get('renderer', {}).get('backend')}`",
        "- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.2 and is recorded in metrics.",
        "",
        "## Purpose",
        "- Run PiLoT single-frame refinement on synthetic query frames generated from GT pose.",
        "- Compare init, refined, and safe-gate selected poses against GT.",
        "- Check whether feature loss drop aligns with GT error drop.",
        "",
        "## Quantitative Results",
        f"- Frames: `{summary.get('num_frames')}`",
        f"- Init XY RMSE: `{summary.get('init_xy_rmse')}` m",
        f"- Refined XY RMSE: `{summary.get('refined_xy_rmse')}` m",
        f"- Selected XY RMSE: `{summary.get('selected_xy_rmse')}` m",
        f"- Init yaw RMSE: `{summary.get('init_yaw_rmse')}` deg",
        f"- Refined yaw RMSE: `{summary.get('refined_yaw_rmse')}` deg",
        f"- Selected yaw RMSE: `{summary.get('selected_yaw_rmse')}` deg",
        f"- Improved frame ratio: `{summary.get('improved_frame_ratio')}`",
        f"- Worse frame ratio: `{summary.get('worse_frame_ratio')}`",
        f"- Safe gate accept ratio: `{summary.get('safe_gate_accept_ratio')}`",
        f"- Mean feature loss drop: `{summary.get('mean_feature_loss_drop')}`",
        f"- Correlation loss drop vs XY error drop: `{summary.get('correlation_feature_loss_drop_vs_xy_error_drop')}`",
        "",
        "## Sanity Check",
        f"- Mean GT chamfer: `{sanity.get('mean_gt_chamfer')}`",
        f"- Mean init chamfer: `{sanity.get('mean_init_chamfer')}`",
        f"- Mean GT overlap: `{sanity.get('mean_gt_overlap')}`",
        f"- Mean init overlap: `{sanity.get('mean_init_overlap')}`",
        f"- Passed: `{sanity.get('passed')}`",
        "",
        "## Interpretation",
        str(summary.get("interpretation")),
        "",
        "## Next Step",
        str(summary.get("recommended_next_step")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sequence-dir", default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--use-safe-gate", action="store_true")
    parser.add_argument("--save-debug-frames", action="store_true")
    parser.add_argument("--sanity-frames", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--init-pose-policy", choices=sorted(INIT_POSE_POLICIES), default="first_pose_odom")
    parser.add_argument("--audit-yaw", action="store_true")
    parser.add_argument("--use-pose-consistency-gate", action="store_true")
    parser.add_argument("--use-temporal-gate", action="store_true")
    parser.add_argument("--allow-temporal-candidate-selection", action="store_true")
    parser.add_argument("--max-yaw-jump-deg", type=float, default=10.0)
    parser.add_argument("--max-xy-jump-m", type=float, default=8.0)
    parser.add_argument("--max-alt-jump-m", type=float, default=2.0)
    parser.add_argument("--temporal-xy-threshold-m", type=float, default=2.0)
    parser.add_argument("--temporal-yaw-threshold-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = _resolve(args.sequence_dir)
    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "sample_debug_frames"
    if args.save_debug_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)

    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cam_cfg, render_camera_np = _scale_camera(config, args.width, args.height)
    render_config = _prepare_render_config(config, render_camera_np, output_dir)
    query_camera = Camera.from_colmap(cam_cfg).to("cuda")
    render_camera = generate_render_camera(render_camera_np).float().to("cuda")
    renderer = DOMDSMRenderer(render_config)
    localizer = RenderLocalizer(config["default_confs"]["from_render_test"])

    gt_poses = _load_pose_file(sequence_dir / "poses" / "gt_poses.txt")
    init_poses = _load_pose_file(sequence_dir / "poses" / "init_noisy_poses.txt")
    sequence_meta = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    image_paths = sorted(p for p in (sequence_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_frames and args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError(f"No sequence images found in {sequence_dir / 'images'}")

    to_raster, from_raster, _crs = get_domdsm_transformers(config)
    odom_edges = _load_odom_edges(sequence_dir / "poses" / "odom_edges.txt")
    init_sequence, init_sources = _build_init_sequence(
        image_paths,
        init_poses,
        odom_edges,
        args.init_pose_policy,
        to_raster,
        from_raster,
    )
    first_gt = gt_poses[image_paths[0].name]
    origin_np = np.asarray(WGS84_to_ECEF(first_gt["trans"]), dtype=np.float64)
    origin = torch.as_tensor(origin_np, device="cuda", dtype=torch.float32)
    refine_conf = deepcopy(config["default_confs"]["refine"])
    refine_conf["origin"] = origin_np
    mul = float(refine_conf.get("mul", 1.0))
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    sanity_rows: List[Dict[str, Any]] = []
    run_logs: List[Dict[str, Any]] = []
    yaw_audit_rows: List[Dict[str, Any]] = []
    gate_candidate_rows: List[Dict[str, Any]] = []
    prev_selected_pose: Optional[Dict[str, Any]] = None
    prev_frame_name: Optional[str] = None
    gate_v2_enabled = bool(args.audit_yaw or args.use_pose_consistency_gate or args.use_temporal_gate)

    for idx, image_path in enumerate(image_paths):
        name = image_path.name
        if name not in gt_poses or name not in init_sequence:
            raise KeyError(f"Missing pose for frame {name}")
        gt_pose = gt_poses[name]
        init_pose = init_sequence[name]
        query_rgb = read_image(image_path, scale=None)

        gt_color = gt_depth = None
        gt_meta: Dict[str, Any] = {}
        gt_visual = {"chamfer": None, "overlap": None}
        if idx < args.sanity_frames:
            gt_color, gt_depth, gt_meta, gt_visual = _render_metric(renderer, query_rgb, gt_pose)

        init_color, init_depth, init_meta, init_visual = _render_metric(renderer, query_rgb, init_pose)
        p3d, T_w2c, T_init, dd = _back_project_for_pose(
            init_depth,
            init_pose["euler"],
            init_pose["trans"],
            render_camera,
            origin,
            mul,
            rng,
            num_samples=args.num_samples,
        )

        tracker = SimpleTracker(localizer.refiner)
        t0 = time.perf_counter()
        try:
            ret = localizer.run_query(
                os.fspath(image_path),
                query_camera,
                render_camera,
                pad_to_multiple(init_color, 16),
                query_T=T_init,
                render_T=T_w2c,
                Points_3D_ECEF=p3d,
                dd=dd,
                last_frame_info={"refine_conf": refine_conf, "observations": []},
                query_resize_ratio=1.0,
                image_query=query_rgb,
            )
        except Exception as exc:
            ret = {"success": False, "error": repr(exc)}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        optimizer_loss_initial, optimizer_loss_refined = _feature_losses_from_tracker(tracker)

        refined_pose = _pose_from_ret(ret, init_pose, name)
        refined_color, refined_depth, refined_meta, refined_visual = _render_metric(renderer, query_rgb, refined_pose)
        feature_loss_initial = _dense_feature_image_loss(localizer, query_rgb, init_color, int(render_camera_np[0]))
        feature_loss_refined = _dense_feature_image_loss(localizer, query_rgb, refined_color, int(render_camera_np[0]))
        feature_loss_drop = (
            feature_loss_initial - feature_loss_refined
            if feature_loss_initial is not None and feature_loss_refined is not None
            else None
        )

        init_err = _pose_error(gt_pose, init_pose, to_raster)
        refined_err = _pose_error(gt_pose, refined_pose, to_raster)
        predicted_pose = _temporal_prediction(prev_selected_pose, prev_frame_name, name, odom_edges, to_raster, from_raster) if args.use_temporal_gate else None

        if gate_v2_enabled:
            selected_candidate, candidates = _evaluate_candidates(
                renderer,
                query_rgb,
                init_pose,
                init_color,
                init_visual,
                refined_pose,
                refined_color,
                refined_visual,
                predicted_pose,
                to_raster,
                from_raster,
                args,
                name,
                feature_loss_initial,
                feature_loss_refined,
            )
            selected_pose = selected_candidate["pose"]
            selected_method = str(selected_candidate["method"])
            selected_color = selected_candidate["color"]
            safe_gate_accepted = selected_method != "init"
            for cand in candidates:
                cand_pose = cand["pose"]
                cand_x, cand_y, cand_alt, cand_yaw = _pose_xy_alt_yaw(cand_pose, to_raster)
                cand_err = _pose_error(gt_pose, cand_pose, to_raster)
                gate_candidate_rows.append(
                    {
                        "frame_id": name,
                        "method": cand["method"],
                        "east": cand_x,
                        "north": cand_y,
                        "alt": cand_alt,
                        "yaw": cand_yaw,
                        "xy_error_m": cand_err["xy_error_m"],
                        "yaw_error_deg": cand_err["yaw_error_deg"],
                        "chamfer": cand["visual"]["chamfer"],
                        "overlap": cand["visual"]["overlap"],
                        "visual_gate_pass": cand["visual_gate_pass"],
                        "visual_gate_reason": cand["visual_gate_reason"],
                        "pose_gate_pass": cand["pose_gate_pass"],
                        "pose_gate_reason": cand["pose_gate_reason"],
                        "temporal_gate_pass": cand["temporal_gate_pass"],
                        "temporal_gate_reason": cand["temporal_gate_reason"],
                        "selectable": cand["selectable"],
                        "final_gate_pass": cand["final_gate_pass"],
                        "score": cand["score"],
                        "selected": cand["method"] == selected_method,
                    }
                )
        else:
            safe_gate_accepted = bool(
                ret.get("success")
                and refined_visual["chamfer"] <= init_visual["chamfer"]
                and refined_visual["overlap"] >= init_visual["overlap"]
            )
            if args.use_safe_gate and not safe_gate_accepted:
                selected_pose = init_pose
                selected_method = "init"
                selected_color = init_color
            else:
                selected_pose = refined_pose
                selected_method = "refined"
                selected_color = refined_color
        selected_err = _pose_error(gt_pose, selected_pose, to_raster)

        gt_x, gt_y, gt_alt, gt_yaw = _pose_xy_alt_yaw(gt_pose, to_raster)
        init_x, init_y, init_alt, init_yaw = _pose_xy_alt_yaw(init_pose, to_raster)
        refined_x, refined_y, refined_alt, refined_yaw = _pose_xy_alt_yaw(refined_pose, to_raster)
        selected_x, selected_y, selected_alt, selected_yaw = _pose_xy_alt_yaw(selected_pose, to_raster)
        corrected_refined_yaw = normalize_angle_deg(refined_yaw + 180.0)
        corrected_refined_yaw_error = angular_diff_deg(corrected_refined_yaw, gt_yaw)
        raw_refined_yaw_error = angular_diff_deg(refined_yaw, gt_yaw)

        row: Dict[str, Any] = {
            "frame_id": name,
            "init_pose_policy": args.init_pose_policy,
            "init_source": init_sources[name],
            "gt_east": gt_x,
            "gt_north": gt_y,
            "gt_alt": gt_alt,
            "gt_yaw": gt_yaw,
            "init_east": init_x,
            "init_north": init_y,
            "init_alt": init_alt,
            "init_yaw": init_yaw,
            "refined_east": refined_x,
            "refined_north": refined_y,
            "refined_alt": refined_alt,
            "refined_yaw": refined_yaw,
            "raw_refined_yaw": refined_yaw,
            "corrected_refined_downward_yaw": corrected_refined_yaw,
            "selected_east": selected_x,
            "selected_north": selected_y,
            "selected_alt": selected_alt,
            "selected_yaw": selected_yaw,
            "selected_method": selected_method,
            "init_xy_error_m": init_err["xy_error_m"],
            "refined_xy_error_m": refined_err["xy_error_m"],
            "selected_xy_error_m": selected_err["xy_error_m"],
            "init_alt_error_m": init_err["alt_error_m"],
            "refined_alt_error_m": refined_err["alt_error_m"],
            "selected_alt_error_m": selected_err["alt_error_m"],
            "init_yaw_error_deg": init_err["yaw_error_deg"],
            "refined_yaw_error_deg": refined_err["yaw_error_deg"],
            "raw_yaw_error_deg": raw_refined_yaw_error,
            "corrected_yaw_error_deg": corrected_refined_yaw_error,
            "selected_yaw_error_deg": selected_err["yaw_error_deg"],
            "init_chamfer": init_visual["chamfer"],
            "refined_chamfer": refined_visual["chamfer"],
            "gt_chamfer": gt_visual["chamfer"],
            "init_overlap": init_visual["overlap"],
            "refined_overlap": refined_visual["overlap"],
            "gt_overlap": gt_visual["overlap"],
            "feature_loss_initial": feature_loss_initial,
            "feature_loss_refined": feature_loss_refined,
            "feature_loss_drop": feature_loss_drop,
            "safe_gate_accepted": safe_gate_accepted,
            "refine_success": bool(ret.get("success")),
            "elapsed_ms": elapsed_ms,
            "renderer_backend_init": init_meta.get("backend_used"),
            "renderer_backend_refined": refined_meta.get("backend_used"),
            "renderer_backend_gt": gt_meta.get("backend_used"),
            "renderer_fallback_init": init_meta.get("fallback_reason"),
            "renderer_fallback_refined": refined_meta.get("fallback_reason"),
            "renderer_fallback_gt": gt_meta.get("fallback_reason"),
        }
        rows.append(row)
        if args.audit_yaw:
            yaw_audit_rows.append(
                {
                    "frame_id": name,
                    "gt_yaw": gt_yaw,
                    "init_yaw": init_yaw,
                    "raw_refined_yaw": refined_yaw,
                    "corrected_refined_downward_yaw": corrected_refined_yaw,
                    "selected_yaw": selected_yaw,
                    "raw_yaw_error_deg": raw_refined_yaw_error,
                    "corrected_yaw_error_deg": corrected_refined_yaw_error,
                    "selected_yaw_error_deg": selected_err["yaw_error_deg"],
                    "selected_method": selected_method,
                }
            )
        if idx < args.sanity_frames:
            sanity_rows.append(row)
        run_logs.append(
            {
                "frame_id": name,
                "success": bool(ret.get("success")),
                "error": ret.get("error"),
                "elapsed_ms": elapsed_ms,
                "feature_loss_initial": feature_loss_initial,
                "feature_loss_refined": feature_loss_refined,
                "optimizer_loss_initial": optimizer_loss_initial,
                "optimizer_loss_refined": optimizer_loss_refined,
                "ret_keys": sorted(ret.keys()),
                "overall_loss": _safe_jsonable(ret.get("overall_loss")),
                "fail_list": _safe_jsonable(ret.get("fail_list")),
                "diff_R": _safe_jsonable(ret.get("diff_R")),
                "diff_t": _safe_jsonable(ret.get("diff_t")),
            }
        )
        if args.save_debug_frames and (idx < 10 or idx in {len(image_paths) - 1}):
            _make_debug_frame(query_rgb, init_color, refined_color, selected_color, debug_dir / f"{idx:06d}_{selected_method}.jpg")
        print(
            f"[{idx + 1}/{len(image_paths)}] {name} "
            f"init_xy={init_err['xy_error_m']:.3f} refined_xy={refined_err['xy_error_m']:.3f} "
            f"selected={selected_method} success={bool(ret.get('success'))}"
        )
        prev_selected_pose = selected_pose
        prev_frame_name = name

    _write_csv(output_dir / "per_frame_metrics.csv", rows)
    if args.audit_yaw:
        _write_csv(output_dir / "yaw_audit.csv", yaw_audit_rows)
    if gate_candidate_rows:
        _write_csv(output_dir / "candidate_gate_metrics.csv", gate_candidate_rows)
    _write_json(output_dir / "run_logs.json", {"frames": run_logs})

    init_xy = [row["init_xy_error_m"] for row in rows]
    refined_xy = [row["refined_xy_error_m"] for row in rows]
    selected_xy = [row["selected_xy_error_m"] for row in rows]
    init_yaw = [row["init_yaw_error_deg"] for row in rows]
    refined_yaw = [row["refined_yaw_error_deg"] for row in rows]
    corrected_yaw = [row["corrected_yaw_error_deg"] for row in rows]
    selected_yaw = [row["selected_yaw_error_deg"] for row in rows]
    xy_error_drop = [row["init_xy_error_m"] - row["refined_xy_error_m"] for row in rows]
    feature_loss_drop = [row["feature_loss_drop"] for row in rows]

    improved = [row["refined_xy_error_m"] < row["init_xy_error_m"] for row in rows]
    worse = [row["refined_xy_error_m"] > row["init_xy_error_m"] for row in rows]
    accept = [bool(row["safe_gate_accepted"]) for row in rows]
    mean_loss_drop = _mean(feature_loss_drop)
    corr_loss_xy = _corr(feature_loss_drop, xy_error_drop)
    selected_method_counts: Dict[str, int] = {}
    for row in rows:
        selected_method_counts[row["selected_method"]] = selected_method_counts.get(row["selected_method"], 0) + 1
    old_summary_path = output_dir.parent / "sim_sequence_pilot_p11" / "summary_metrics.json"
    old_gate_v2_summary_path = output_dir.parent / "sim_sequence_pilot_p11_gate_v2" / "summary_metrics.json"
    old_summary: Dict[str, Any] = {}
    if old_summary_path.exists() and old_summary_path.resolve() != (output_dir / "summary_metrics.json").resolve():
        old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    old_gate_v2_summary: Dict[str, Any] = {}
    if old_gate_v2_summary_path.exists() and old_gate_v2_summary_path.resolve() != (output_dir / "summary_metrics.json").resolve():
        old_gate_v2_summary = json.loads(old_gate_v2_summary_path.read_text(encoding="utf-8"))

    non_init_candidates = [c for c in gate_candidate_rows if c.get("method") != "init"]
    visual_gate_accept_ratio = float(np.mean([bool(c["visual_gate_pass"]) for c in non_init_candidates])) if non_init_candidates else None
    pose_gate_accept_ratio = float(np.mean([bool(c["pose_gate_pass"]) for c in non_init_candidates])) if non_init_candidates else None
    temporal_gate_accept_ratio = float(np.mean([bool(c["temporal_gate_pass"]) for c in non_init_candidates])) if non_init_candidates else None
    final_accept_ratio = float(np.mean([row["selected_method"] != "init" for row in rows])) if rows else None
    gate_breakdown = {
        "num_candidate_rows": len(gate_candidate_rows),
        "init_pose_policy": args.init_pose_policy,
        "allow_temporal_candidate_selection": bool(args.allow_temporal_candidate_selection),
        "selected_method_counts": selected_method_counts,
        "visual_gate_accept_ratio": visual_gate_accept_ratio,
        "pose_gate_accept_ratio": pose_gate_accept_ratio,
        "temporal_gate_accept_ratio": temporal_gate_accept_ratio,
        "final_accept_ratio": final_accept_ratio,
        "visual_reject_reasons": {},
        "pose_reject_reasons": {},
        "temporal_reject_reasons": {},
    }
    for cand in non_init_candidates:
        if not cand["visual_gate_pass"]:
            gate_breakdown["visual_reject_reasons"][cand["visual_gate_reason"]] = gate_breakdown["visual_reject_reasons"].get(cand["visual_gate_reason"], 0) + 1
        if not cand["pose_gate_pass"]:
            gate_breakdown["pose_reject_reasons"][cand["pose_gate_reason"]] = gate_breakdown["pose_reject_reasons"].get(cand["pose_gate_reason"], 0) + 1
        if not cand["temporal_gate_pass"]:
            gate_breakdown["temporal_reject_reasons"][cand["temporal_gate_reason"]] = gate_breakdown["temporal_reject_reasons"].get(cand["temporal_gate_reason"], 0) + 1

    refined_better = _rmse(refined_xy) is not None and _rmse(init_xy) is not None and _rmse(refined_xy) < _rmse(init_xy)
    selected_better = _rmse(selected_xy) is not None and _rmse(init_xy) is not None and _rmse(selected_xy) <= _rmse(init_xy)
    if refined_better and selected_better:
        interpretation = "PiLoT refinement improves synthetic same-domain localization; safe gate preserves or improves the result."
        recommended_next_step = "Proceed to temporal smoother and sliding-window sequence constraints."
    elif selected_better:
        interpretation = "Raw refinement is unstable on some frames, but safe gate prevents regressions against init."
        recommended_next_step = "Tune safe gate and then proceed to temporal smoothing with selected poses."
    else:
        interpretation = "PiLoT refinement does not reliably improve noisy init on this synthetic sequence."
        recommended_next_step = "Inspect run logs, feature loss/GT error correlation, and candidate generation before temporal optimization."

    summary = {
        "init_pose_policy": args.init_pose_policy,
        "allow_temporal_candidate_selection": bool(args.allow_temporal_candidate_selection),
        "num_frames": len(rows),
        "init_xy_rmse": _rmse(init_xy),
        "refined_xy_rmse": _rmse(refined_xy),
        "raw_refined_xy_rmse": _rmse(refined_xy),
        "selected_xy_rmse": _rmse(selected_xy),
        "old_selected_xy_rmse": old_summary.get("selected_xy_rmse"),
        "old_gate_v2_selected_xy_rmse": old_gate_v2_summary.get("selected_xy_rmse"),
        "delta_xy_rmse_vs_old_gate_v2": (
            _rmse(selected_xy) - float(old_gate_v2_summary["selected_xy_rmse"])
            if old_gate_v2_summary.get("selected_xy_rmse") is not None and _rmse(selected_xy) is not None
            else None
        ),
        "selected_v2_xy_rmse": _rmse(selected_xy),
        "init_yaw_rmse": _rmse(init_yaw),
        "refined_yaw_rmse": _rmse(refined_yaw),
        "raw_refined_yaw_rmse": _rmse(refined_yaw),
        "corrected_refined_yaw_rmse": _rmse(corrected_yaw),
        "selected_yaw_rmse": _rmse(selected_yaw),
        "old_selected_yaw_rmse": old_summary.get("selected_yaw_rmse"),
        "old_gate_v2_selected_yaw_rmse": old_gate_v2_summary.get("selected_yaw_rmse"),
        "delta_yaw_rmse_vs_old_gate_v2": (
            _rmse(selected_yaw) - float(old_gate_v2_summary["selected_yaw_rmse"])
            if old_gate_v2_summary.get("selected_yaw_rmse") is not None and _rmse(selected_yaw) is not None
            else None
        ),
        "selected_v2_yaw_rmse": _rmse(selected_yaw),
        "improved_frame_ratio": float(np.mean(improved)) if improved else None,
        "worse_frame_ratio": float(np.mean(worse)) if worse else None,
        "safe_gate_accept_ratio": float(np.mean(accept)) if accept else None,
        "visual_gate_accept_ratio": visual_gate_accept_ratio,
        "pose_gate_accept_ratio": pose_gate_accept_ratio,
        "temporal_gate_accept_ratio": temporal_gate_accept_ratio,
        "final_accept_ratio": final_accept_ratio,
        "selected_method_counts": selected_method_counts,
        "mean_init_chamfer": _mean([row["init_chamfer"] for row in rows]),
        "mean_refined_chamfer": _mean([row["refined_chamfer"] for row in rows]),
        "mean_init_overlap": _mean([row["init_overlap"] for row in rows]),
        "mean_refined_overlap": _mean([row["refined_overlap"] for row in rows]),
        "mean_feature_loss_drop": mean_loss_drop,
        "correlation_feature_loss_drop_vs_xy_error_drop": corr_loss_xy,
        "interpretation": interpretation,
        "recommended_next_step": recommended_next_step,
    }
    sanity = {
        "num_frames": len(sanity_rows),
        "mean_gt_chamfer": _mean([row["gt_chamfer"] for row in sanity_rows]),
        "mean_init_chamfer": _mean([row["init_chamfer"] for row in sanity_rows]),
        "mean_gt_overlap": _mean([row["gt_overlap"] for row in sanity_rows]),
        "mean_init_overlap": _mean([row["init_overlap"] for row in sanity_rows]),
    }
    sanity["passed"] = bool(
        sanity["mean_gt_chamfer"] is not None
        and sanity["mean_init_chamfer"] is not None
        and sanity["mean_gt_overlap"] is not None
        and sanity["mean_init_overlap"] is not None
        and sanity["mean_gt_chamfer"] < sanity["mean_init_chamfer"]
        and sanity["mean_gt_overlap"] > sanity["mean_init_overlap"]
    )
    _write_json(output_dir / "summary_metrics.json", summary)
    _write_json(output_dir / "sanity_check.json", sanity)
    if gate_v2_enabled:
        _write_json(output_dir / "gate_decision_breakdown.json", gate_breakdown)

    _plot_trajectory(rows, output_dir / "trajectory_gt_init_refined.png")
    if gate_v2_enabled:
        _plot_trajectory(rows, output_dir / "trajectory_gt_init_refined_selected.png")
    _plot_error_curve(rows, output_dir / "xy_error_curve.png", "xy_error_m", "XY error (m)")
    _plot_error_curve(rows, output_dir / "yaw_error_curve.png", "yaw_error_deg", "Yaw error (deg)")
    _plot_visual_curve(rows, output_dir / "visual_metric_curve.png")
    report_path = output_dir.parent / f"{output_dir.name}_check.md" if gate_v2_enabled else _resolve(DEFAULT_DOC)
    _write_report(report_path, sequence_meta, summary, sanity)

    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
