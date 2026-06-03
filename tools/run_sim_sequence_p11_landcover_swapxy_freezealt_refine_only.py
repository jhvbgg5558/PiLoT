#!/usr/bin/env python3
"""Run P11 landcover sequence with refine-only swap-XY/freeze-alt propagation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.pixlib.geometry import Camera
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import (
    compute_enu_delta_m,
    get_domdsm_transformers,
    make_downward_euler_from_yaw,
    make_safe_domdsm_pose_from_refined,
    normalize_angle_deg,
    refined_yaw_to_downward_yaw,
)
from pixloc.utils.get_depth import generate_render_camera, pad_to_multiple
from pixloc.utils.transform import WGS84_to_ECEF
from tools.run_sim_sequence_pilot_p11 import (
    IMAGE_EXTS,
    _back_project_for_pose,
    _dense_feature_image_loss,
    _load_pose_file,
    _make_debug_frame,
    _mean,
    _pose_error,
    _pose_from_ret,
    _pose_xy_alt_yaw,
    _prepare_render_config,
    _render_metric,
    _rmse,
    _safe_jsonable,
    _scale_camera,
)


DEFAULT_SEQUENCE_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover"
DEFAULT_OUTPUT_DIR = (
    "docs/experiments/dom_dsm_prepare/"
    "sim_sequence_p11_landcover_swapxy_freezealt_refine_only"
)
DEFAULT_METHOD = "swap_xy_freeze_alt_refine_only"
CONSTRAINED_METHOD = "swap_xy_freeze_alt_constrained_yaw"
YAW_GRID_OFFSETS_DEG = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
YAW_TRUST_REGION_DEG = 3.0


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "frame_index",
        "frame_id",
        "init_source",
        "selected_method",
        "yaw_policy",
        "yaw_pred",
        "yaw_obs_corrected",
        "yaw_obs_delta_deg",
        "accepted_yaw_delta_deg",
        "yaw_obs_trusted",
        "selected_yaw_source",
        "refine_success",
        "elapsed_ms",
        "gt_east",
        "gt_north",
        "gt_alt",
        "gt_yaw",
        "init_east",
        "init_north",
        "init_alt",
        "init_yaw",
        "raw_refined_east",
        "raw_refined_north",
        "raw_refined_alt",
        "raw_refined_yaw",
        "selected_east",
        "selected_north",
        "selected_alt",
        "selected_yaw",
        "raw_delta_east_m",
        "raw_delta_north_m",
        "raw_delta_alt_m",
        "applied_delta_east_m",
        "applied_delta_north_m",
        "applied_delta_alt_m",
        "init_xy_error_m",
        "raw_refined_xy_error_m",
        "selected_xy_error_m",
        "init_alt_error_m",
        "raw_refined_alt_error_m",
        "selected_alt_error_m",
        "init_yaw_error_deg",
        "raw_refined_yaw_error_deg",
        "selected_yaw_error_deg",
        "init_chamfer",
        "raw_refined_chamfer",
        "selected_chamfer",
        "init_overlap",
        "raw_refined_overlap",
        "selected_overlap",
        "feature_loss_initial",
        "feature_loss_refined",
        "feature_loss_drop",
        "failure_reason",
    ]
    extras = sorted({k for row in rows for k in row.keys()} - set(preferred))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + extras, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"rmse": None, "mean": None, "median": None, "max": None, "final": None}
    return {
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "final": float(values[-1]) if values else None,
    }


def _angle_diff_signed_deg(a: float, b: float) -> float:
    return float(normalize_angle_deg(float(a) - float(b)))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(float(lo), min(float(hi), float(value))))


def _plot_trajectory(rows: List[Dict[str, Any]], path: Path) -> None:
    plt.figure(figsize=(8, 8))
    for prefix, label in [
        ("gt", "GT"),
        ("init", "Init"),
        ("selected", "Refine-only selected"),
    ]:
        plt.plot(
            [row[f"{prefix}_east"] for row in rows],
            [row[f"{prefix}_north"] for row in rows],
            marker=".",
            linewidth=1,
            markersize=3,
            label=label,
        )
    plt.axis("equal")
    plt.xlabel("Raster X / Easting (m)")
    plt.ylabel("Raster Y / Northing (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_error_curve(rows: List[Dict[str, Any]], path: Path, key: str, ylabel: str) -> None:
    x = np.arange(len(rows))
    plt.figure(figsize=(10, 4))
    plt.plot(x, [row[f"init_{key}"] for row in rows], label="Init")
    plt.plot(x, [row[f"selected_{key}"] for row in rows], label="Refine-only selected")
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _make_adapted_pose(
    frame_name: str,
    init_pose: Dict[str, Any],
    raw_refined_pose: Dict[str, Any],
    to_raster: Any,
    from_raster: Any,
) -> Dict[str, Any]:
    adapted = make_safe_domdsm_pose_from_refined(
        init_pose["trans"],
        init_pose["euler"],
        raw_refined_pose["trans"],
        raw_refined_pose["euler"],
        "swap_xy_freeze_alt_corrected_downward_yaw",
        to_raster=to_raster,
        from_raster=from_raster,
    )
    return {
        "name": frame_name,
        "trans": adapted["translation_lon_lat_alt"],
        "euler": adapted["euler_pitch_roll_yaw"],
        "adapter_metadata": {
            "raw_delta_east_north_alt_m": adapted["raw_delta_east_north_alt_m"],
            "applied_delta_east_north_alt_m": adapted[
                "applied_delta_east_north_alt_m"
            ],
            "corrected_downward_yaw": adapted["corrected_downward_yaw"],
        },
    }


def _make_adapted_translation_pose(
    frame_name: str,
    init_pose: Dict[str, Any],
    raw_refined_pose: Dict[str, Any],
    yaw: float,
    to_raster: Any,
    from_raster: Any,
) -> Dict[str, Any]:
    adapted = make_safe_domdsm_pose_from_refined(
        init_pose["trans"],
        init_pose["euler"],
        raw_refined_pose["trans"],
        raw_refined_pose["euler"],
        "swap_xy_freeze_alt_corrected_downward_yaw",
        to_raster=to_raster,
        from_raster=from_raster,
    )
    return {
        "name": frame_name,
        "trans": adapted["translation_lon_lat_alt"],
        "euler": make_downward_euler_from_yaw(yaw),
        "adapter_metadata": {
            "raw_delta_east_north_alt_m": adapted["raw_delta_east_north_alt_m"],
            "applied_delta_east_north_alt_m": adapted[
                "applied_delta_east_north_alt_m"
            ],
            "corrected_downward_yaw": adapted["corrected_downward_yaw"],
        },
    }


def _unique_yaw_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for cand in candidates:
        key = round(float(cand["yaw"]), 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _select_constrained_yaw(
    frame_index: int,
    frame_name: str,
    query_rgb: np.ndarray,
    init_pose: Dict[str, Any],
    raw_refined_pose: Dict[str, Any],
    renderer: DOMDSMRenderer,
    localizer: RenderLocalizer,
    init_visual: Dict[str, float],
    init_feature_loss: Optional[float],
    render_camera_width: int,
    yaw_pred: float,
    to_raster: Any,
    from_raster: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    yaw_obs = refined_yaw_to_downward_yaw(raw_refined_pose["euler"][2])
    yaw_obs_delta = _angle_diff_signed_deg(yaw_obs, yaw_pred)
    yaw_obs_trusted = abs(yaw_obs_delta) <= YAW_TRUST_REGION_DEG
    clamped_obs = normalize_angle_deg(
        float(yaw_pred)
        + _clamp(yaw_obs_delta, -YAW_TRUST_REGION_DEG, YAW_TRUST_REGION_DEG)
    )
    cand_specs: List[Dict[str, Any]] = [
        {"source": "prediction", "yaw": normalize_angle_deg(yaw_pred)}
    ]
    for offset in YAW_GRID_OFFSETS_DEG:
        cand_specs.append(
            {
                "source": "grid",
                "grid_offset_deg": float(offset),
                "yaw": normalize_angle_deg(float(yaw_pred) + float(offset)),
            }
        )
    cand_specs.append({"source": "clamped_raw_observation", "yaw": clamped_obs})
    if yaw_obs_trusted:
        cand_specs.append({"source": "trusted_raw_observation", "yaw": yaw_obs})
    cand_specs = _unique_yaw_candidates(cand_specs)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    chamfer_den = max(float(init_visual["chamfer"]), 1e-6)
    feature_den = (
        max(float(init_feature_loss), 1e-9)
        if init_feature_loss is not None and np.isfinite(init_feature_loss)
        else None
    )
    for cand in cand_specs:
        yaw = float(cand["yaw"])
        pose = _make_adapted_translation_pose(
            frame_name, init_pose, raw_refined_pose, yaw, to_raster, from_raster
        )
        try:
            color, _depth, _meta, visual = _render_metric(renderer, query_rgb, pose)
            feature_loss = _dense_feature_image_loss(
                localizer, query_rgb, color, render_camera_width
            )
            chamfer_norm = float(visual["chamfer"]) / chamfer_den
            feature_norm = (
                float(feature_loss) / feature_den
                if feature_loss is not None and feature_den is not None
                else 0.0
            )
            yaw_delta_abs = abs(_angle_diff_signed_deg(yaw, yaw_pred))
            score = (
                chamfer_norm
                + (1.0 - float(visual["overlap"]))
                + 0.25 * feature_norm
                + 0.03 * yaw_delta_abs
            )
            failed = False
            failure_reason = None
        except Exception as exc:
            pose = _make_adapted_translation_pose(
                frame_name, init_pose, raw_refined_pose, yaw_pred, to_raster, from_raster
            )
            visual = {"chamfer": float("inf"), "overlap": 0.0}
            feature_loss = None
            chamfer_norm = float("inf")
            feature_norm = None
            yaw_delta_abs = abs(_angle_diff_signed_deg(yaw, yaw_pred))
            score = float("inf")
            failed = True
            failure_reason = repr(exc)
        row = {
            "frame_index": frame_index,
            "frame_id": frame_name,
            "candidate_source": cand["source"],
            "grid_offset_deg": cand.get("grid_offset_deg"),
            "yaw_pred": float(yaw_pred),
            "yaw_obs_corrected": yaw_obs,
            "yaw_obs_delta_deg": yaw_obs_delta,
            "yaw_obs_trusted": bool(yaw_obs_trusted),
            "candidate_yaw": yaw,
            "candidate_yaw_delta_deg": _angle_diff_signed_deg(yaw, yaw_pred),
            "edge_chamfer": visual["chamfer"],
            "edge_overlap": visual["overlap"],
            "feature_loss": feature_loss,
            "chamfer_norm": chamfer_norm,
            "feature_norm": feature_norm,
            "score": score,
            "render_failed": failed,
            "failure_reason": failure_reason,
            "selected": False,
            "pose": pose,
        }
        rows.append(row)
        if not failed and (best is None or float(row["score"]) < float(best["score"])):
            best = row

    if best is None:
        fallback_pose = _make_adapted_translation_pose(
            frame_name, init_pose, raw_refined_pose, yaw_pred, to_raster, from_raster
        )
        best = {
            "candidate_source": "fallback_prediction",
            "candidate_yaw": normalize_angle_deg(yaw_pred),
            "candidate_yaw_delta_deg": 0.0,
            "pose": fallback_pose,
            "score": None,
        }
    for row in rows:
        row["selected"] = (
            row["candidate_source"] == best["candidate_source"]
            and abs(float(row["candidate_yaw"]) - float(best["candidate_yaw"])) < 1e-6
        )
    debug = {
        "yaw_pred": float(yaw_pred),
        "yaw_obs_corrected": yaw_obs,
        "yaw_obs_delta_deg": yaw_obs_delta,
        "yaw_obs_trusted": bool(yaw_obs_trusted),
        "accepted_yaw_delta_deg": _angle_diff_signed_deg(
            float(best["candidate_yaw"]), yaw_pred
        ),
        "selected_yaw_source": best["candidate_source"],
        "selected_yaw_score": best.get("score"),
    }
    export_rows = []
    for row in rows:
        export = {k: v for k, v in row.items() if k != "pose"}
        export_rows.append(export)
    return best["pose"], export_rows, debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--render-backend",
        default=None,
        help=(
            "Override render_config.dom_dsm.render_backend. If omitted, use the "
            "backend recorded for frame 0 in sequence summary when available."
        ),
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument(
        "--first-pose-source",
        choices=["gt", "init_noisy"],
        default="gt",
        help="Use only this source for frame 0; later frames always propagate from previous accepted pose.",
    )
    parser.add_argument(
        "--yaw-policy",
        choices=["raw_corrected", "constrained_search"],
        default="raw_corrected",
    )
    parser.add_argument("--save-debug-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_dir = _resolve(args.sequence_dir)
    sequence_meta = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    config_arg = args.config or sequence_meta.get("config")
    if not config_arg:
        raise ValueError("Config must be provided or present in sequence metadata")
    config_path = _resolve(config_arg)
    width = int(args.width or sequence_meta.get("camera", {}).get("width") or 512)
    height = args.height
    if height is None and sequence_meta.get("camera", {}).get("height") is not None:
        height = int(sequence_meta["camera"]["height"])

    output_dir = _resolve(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "sample_debug_frames"
    if args.save_debug_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame_records = json.loads(
        (sequence_dir / "summary.json").read_text(encoding="utf-8")
    ).get("frame_records", [])
    generated_backend = None
    if frame_records:
        generated_backend = (
            frame_records[0].get("renderer_metadata", {}).get("backend_used")
        )
    render_backend = args.render_backend or generated_backend
    if render_backend:
        config.setdefault("render_config", {}).setdefault("dom_dsm", {})[
            "render_backend"
        ] = render_backend
    cam_cfg, render_camera_np = _scale_camera(config, width, height)
    render_config = _prepare_render_config(config, render_camera_np, output_dir)
    query_camera = Camera.from_colmap(cam_cfg).to("cuda")
    render_camera = generate_render_camera(render_camera_np).float().to("cuda")
    renderer = DOMDSMRenderer(render_config)
    localizer = RenderLocalizer(config["default_confs"]["from_render_test"])

    gt_poses = _load_pose_file(sequence_dir / "poses" / "gt_poses.txt")
    init_noisy_poses = (
        _load_pose_file(sequence_dir / "poses" / "init_noisy_poses.txt")
        if args.first_pose_source == "init_noisy"
        else {}
    )
    image_paths = sorted(
        p for p in (sequence_dir / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if args.max_frames and args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError(f"No sequence images found in {sequence_dir / 'images'}")

    to_raster, from_raster, raster_crs = get_domdsm_transformers(config)
    first_name = image_paths[0].name
    if first_name not in gt_poses:
        raise KeyError(f"Missing GT first pose for {first_name}")
    if args.first_pose_source == "init_noisy":
        if first_name not in init_noisy_poses:
            raise KeyError(f"Missing init noisy first pose for {first_name}")
        first_pose = deepcopy(init_noisy_poses[first_name])
        first_pose_source = "init_noisy_poses.txt:first_frame"
    else:
        first_pose = deepcopy(gt_poses[first_name])
        first_pose_source = "gt_poses.txt:first_frame"
    first_pose["name"] = first_name

    origin_np = np.asarray(WGS84_to_ECEF(first_pose["trans"]), dtype=np.float64)
    origin = torch.as_tensor(origin_np, device="cuda", dtype=torch.float32)
    refine_conf = deepcopy(config["default_confs"]["refine"])
    refine_conf["origin"] = origin_np
    mul = float(refine_conf.get("mul", 1.0))
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    yaw_candidate_rows: List[Dict[str, Any]] = []
    run_logs: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    current_init_pose = first_pose
    current_init_source = "init_noisy_first_pose" if args.first_pose_source == "init_noisy" else "gt_first_pose"

    for idx, image_path in enumerate(image_paths):
        name = image_path.name
        if name not in gt_poses:
            raise KeyError(f"Missing GT pose for frame {name}")
        gt_pose = gt_poses[name]
        init_pose = deepcopy(current_init_pose)
        init_pose["name"] = name
        query_rgb = read_image(image_path, scale=None)

        init_color, init_depth, init_meta, init_visual = _render_metric(
            renderer, query_rgb, init_pose
        )
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

        t0 = time.perf_counter()
        failure_reason = None
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
        refine_success = bool(ret.get("success"))
        if not refine_success:
            failure_reason = str(ret.get("error") or "run_query returned success=False")
            failures.append(
                {"frame_index": idx, "frame_id": name, "failure_reason": failure_reason}
            )

        raw_refined_pose = _pose_from_ret(ret, init_pose, name)
        raw_color, _raw_depth, raw_meta, raw_visual = _render_metric(
            renderer, query_rgb, raw_refined_pose
        )
        feature_loss_initial = _dense_feature_image_loss(
            localizer, query_rgb, init_color, int(render_camera_np[0])
        )
        feature_loss_refined = _dense_feature_image_loss(
            localizer, query_rgb, raw_color, int(render_camera_np[0])
        )
        feature_loss_drop = (
            feature_loss_initial - feature_loss_refined
            if feature_loss_initial is not None and feature_loss_refined is not None
            else None
        )
        yaw_pred = float(init_pose["euler"][2])
        yaw_debug = {
            "yaw_pred": yaw_pred,
            "yaw_obs_corrected": refined_yaw_to_downward_yaw(raw_refined_pose["euler"][2]),
            "yaw_obs_delta_deg": _angle_diff_signed_deg(
                refined_yaw_to_downward_yaw(raw_refined_pose["euler"][2]), yaw_pred
            ),
            "accepted_yaw_delta_deg": None,
            "yaw_obs_trusted": None,
            "selected_yaw_source": None,
            "selected_yaw_score": None,
        }
        if refine_success:
            if args.yaw_policy == "constrained_search":
                selected_pose, yaw_rows, yaw_debug = _select_constrained_yaw(
                    idx,
                    name,
                    query_rgb,
                    init_pose,
                    raw_refined_pose,
                    renderer,
                    localizer,
                    init_visual,
                    feature_loss_initial,
                    int(render_camera_np[0]),
                    yaw_pred,
                    to_raster,
                    from_raster,
                )
                yaw_candidate_rows.extend(yaw_rows)
                selected_method = CONSTRAINED_METHOD
            else:
                selected_pose = _make_adapted_pose(
                    name, init_pose, raw_refined_pose, to_raster, from_raster
                )
                yaw_debug["accepted_yaw_delta_deg"] = _angle_diff_signed_deg(
                    selected_pose["euler"][2], yaw_pred
                )
                yaw_debug["yaw_obs_trusted"] = True
                yaw_debug["selected_yaw_source"] = "raw_corrected"
                selected_method = DEFAULT_METHOD
        else:
            selected_pose = deepcopy(init_pose)
            selected_pose["adapter_metadata"] = {
                "raw_delta_east_north_alt_m": [0.0, 0.0, 0.0],
                "applied_delta_east_north_alt_m": [0.0, 0.0, 0.0],
                "corrected_downward_yaw": float(init_pose["euler"][2]),
            }
            yaw_debug["accepted_yaw_delta_deg"] = 0.0
            yaw_debug["yaw_obs_trusted"] = False
            yaw_debug["selected_yaw_source"] = "failed_keep_init"
            selected_method = "failed_keep_init"

        selected_color, _selected_depth, selected_meta, selected_visual = _render_metric(
            renderer, query_rgb, selected_pose
        )

        gt_x, gt_y, gt_alt, gt_yaw = _pose_xy_alt_yaw(gt_pose, to_raster)
        init_x, init_y, init_alt, init_yaw = _pose_xy_alt_yaw(init_pose, to_raster)
        raw_x, raw_y, raw_alt, raw_yaw = _pose_xy_alt_yaw(raw_refined_pose, to_raster)
        sel_x, sel_y, sel_alt, sel_yaw = _pose_xy_alt_yaw(selected_pose, to_raster)
        init_err = _pose_error(gt_pose, init_pose, to_raster)
        raw_err = _pose_error(gt_pose, raw_refined_pose, to_raster)
        selected_err = _pose_error(gt_pose, selected_pose, to_raster)
        adapter_meta = selected_pose.get("adapter_metadata", {})
        raw_delta = adapter_meta.get(
            "raw_delta_east_north_alt_m",
            compute_enu_delta_m(init_pose["trans"], raw_refined_pose["trans"], to_raster),
        )
        applied_delta = adapter_meta.get("applied_delta_east_north_alt_m", [0.0, 0.0, 0.0])

        row: Dict[str, Any] = {
            "frame_index": idx,
            "frame_id": name,
            "init_source": current_init_source,
            "selected_method": selected_method,
            "yaw_policy": args.yaw_policy,
            "yaw_pred": yaw_debug["yaw_pred"],
            "yaw_obs_corrected": yaw_debug["yaw_obs_corrected"],
            "yaw_obs_delta_deg": yaw_debug["yaw_obs_delta_deg"],
            "accepted_yaw_delta_deg": yaw_debug["accepted_yaw_delta_deg"],
            "yaw_obs_trusted": yaw_debug["yaw_obs_trusted"],
            "selected_yaw_source": yaw_debug["selected_yaw_source"],
            "selected_yaw_score": yaw_debug["selected_yaw_score"],
            "refine_success": refine_success,
            "elapsed_ms": elapsed_ms,
            "gt_east": gt_x,
            "gt_north": gt_y,
            "gt_alt": gt_alt,
            "gt_yaw": gt_yaw,
            "init_east": init_x,
            "init_north": init_y,
            "init_alt": init_alt,
            "init_yaw": init_yaw,
            "raw_refined_east": raw_x,
            "raw_refined_north": raw_y,
            "raw_refined_alt": raw_alt,
            "raw_refined_yaw": raw_yaw,
            "selected_east": sel_x,
            "selected_north": sel_y,
            "selected_alt": sel_alt,
            "selected_yaw": sel_yaw,
            "raw_delta_east_m": raw_delta[0],
            "raw_delta_north_m": raw_delta[1],
            "raw_delta_alt_m": raw_delta[2],
            "applied_delta_east_m": applied_delta[0],
            "applied_delta_north_m": applied_delta[1],
            "applied_delta_alt_m": applied_delta[2],
            "init_xy_error_m": init_err["xy_error_m"],
            "raw_refined_xy_error_m": raw_err["xy_error_m"],
            "selected_xy_error_m": selected_err["xy_error_m"],
            "init_alt_error_m": init_err["alt_error_m"],
            "raw_refined_alt_error_m": raw_err["alt_error_m"],
            "selected_alt_error_m": selected_err["alt_error_m"],
            "init_yaw_error_deg": init_err["yaw_error_deg"],
            "raw_refined_yaw_error_deg": raw_err["yaw_error_deg"],
            "selected_yaw_error_deg": selected_err["yaw_error_deg"],
            "init_chamfer": init_visual["chamfer"],
            "raw_refined_chamfer": raw_visual["chamfer"],
            "selected_chamfer": selected_visual["chamfer"],
            "init_overlap": init_visual["overlap"],
            "raw_refined_overlap": raw_visual["overlap"],
            "selected_overlap": selected_visual["overlap"],
            "feature_loss_initial": feature_loss_initial,
            "feature_loss_refined": feature_loss_refined,
            "feature_loss_drop": feature_loss_drop,
            "failure_reason": failure_reason,
            "renderer_backend_init": init_meta.get("backend_used"),
            "renderer_backend_raw_refined": raw_meta.get("backend_used"),
            "renderer_backend_selected": selected_meta.get("backend_used"),
        }
        rows.append(row)
        run_logs.append(
            {
                "frame_index": idx,
                "frame_id": name,
                "init_source": current_init_source,
                "refine_success": refine_success,
                "failure_reason": failure_reason,
                "elapsed_ms": elapsed_ms,
                "ret_keys": sorted(ret.keys()),
                "overall_loss": _safe_jsonable(ret.get("overall_loss")),
                "fail_list": _safe_jsonable(ret.get("fail_list")),
                "diff_R": _safe_jsonable(ret.get("diff_R")),
                "diff_t": _safe_jsonable(ret.get("diff_t")),
            }
        )

        if args.save_debug_frames and (idx < 10 or idx == len(image_paths) - 1):
            _make_debug_frame(
                query_rgb,
                init_color,
                raw_color,
                selected_color,
                debug_dir / f"{idx:06d}_{selected_method}.jpg",
            )
        print(
            f"[{idx + 1}/{len(image_paths)}] {name} "
            f"init_xy={init_err['xy_error_m']:.3f} "
            f"selected_xy={selected_err['xy_error_m']:.3f} "
            f"method={selected_method} success={refine_success}"
        )
        current_init_pose = deepcopy(selected_pose)
        current_init_source = f"previous_frame_refine_only:{name}"

    _write_csv(output_dir / "per_frame_metrics.csv", rows)
    if yaw_candidate_rows:
        _write_csv(output_dir / "yaw_candidate_metrics.csv", yaw_candidate_rows)
    _write_json(output_dir / "run_logs.json", {"frames": run_logs})

    selected_xy = [float(row["selected_xy_error_m"]) for row in rows]
    selected_alt = [float(row["selected_alt_error_m"]) for row in rows]
    selected_yaw = [float(row["selected_yaw_error_deg"]) for row in rows]
    init_xy = [float(row["init_xy_error_m"]) for row in rows]
    init_alt = [float(row["init_alt_error_m"]) for row in rows]
    init_yaw = [float(row["init_yaw_error_deg"]) for row in rows]
    raw_xy = [float(row["raw_refined_xy_error_m"]) for row in rows]
    selected_methods: Dict[str, int] = {}
    for row in rows:
        selected_methods[row["selected_method"]] = (
            selected_methods.get(row["selected_method"], 0) + 1
        )
    yaw_obs_rejected_count = sum(
        1
        for row in rows
        if row.get("yaw_obs_trusted") is False and bool(row.get("refine_success"))
    )
    accepted_yaw_steps = [
        abs(float(row["accepted_yaw_delta_deg"]))
        for row in rows
        if row.get("accepted_yaw_delta_deg") is not None
        and np.isfinite(float(row["accepted_yaw_delta_deg"]))
    ]

    summary = {
        "experiment": "P11 landcover refine-only swapXY freeze-alt propagation",
        "sequence_dir": str(sequence_dir),
        "config": str(config_path),
        "output_dir": str(output_dir),
        "raster_crs": raster_crs,
        "num_frames": len(rows),
        "first_pose_source": first_pose_source,
        "uses_init_noisy_poses": args.first_pose_source == "init_noisy",
        "uses_init_noisy_poses_only_for_first_frame": args.first_pose_source == "init_noisy",
        "uses_odom_edges": False,
        "uses_safe_gate": False,
        "requested_render_backend": render_backend,
        "query_generated_backend": generated_backend,
        "yaw_policy": args.yaw_policy,
        "yaw_trust_region_deg": YAW_TRUST_REGION_DEG,
        "yaw_grid_offsets_deg": YAW_GRID_OFFSETS_DEG,
        "propagation_policy": "next init equals previous accepted adapted refine pose",
        "adapter_mode": "swap_xy_freeze_alt_corrected_downward_yaw",
        "refine_success_count": sum(bool(row["refine_success"]) for row in rows),
        "failure_count": len(failures),
        "failures": failures,
        "selected_method_counts": selected_methods,
        "yaw_obs_rejected_count": yaw_obs_rejected_count,
        "mean_accepted_yaw_step_deg": _mean(accepted_yaw_steps),
        "max_accepted_yaw_step_deg": max(accepted_yaw_steps)
        if accepted_yaw_steps
        else None,
        "num_yaw_candidate_rows": len(yaw_candidate_rows),
        "init_xy_error_m": _stats(init_xy),
        "raw_refined_xy_error_m": _stats(raw_xy),
        "selected_xy_error_m": _stats(selected_xy),
        "init_alt_error_m": _stats(init_alt),
        "selected_alt_error_m": _stats(selected_alt),
        "init_yaw_error_deg": _stats(init_yaw),
        "selected_yaw_error_deg": _stats(selected_yaw),
        "selected_minus_init_xy_rmse": (
            _rmse(selected_xy) - _rmse(init_xy)
            if _rmse(selected_xy) is not None and _rmse(init_xy) is not None
            else None
        ),
        "selected_minus_init_alt_rmse": (
            _rmse(selected_alt) - _rmse(init_alt)
            if _rmse(selected_alt) is not None and _rmse(init_alt) is not None
            else None
        ),
        "selected_minus_init_yaw_rmse": (
            _rmse(selected_yaw) - _rmse(init_yaw)
            if _rmse(selected_yaw) is not None and _rmse(init_yaw) is not None
            else None
        ),
        "drift_indicators": {
            "final_xy_error_m": selected_xy[-1] if selected_xy else None,
            "max_xy_error_m": max(selected_xy) if selected_xy else None,
            "final_alt_error_m": selected_alt[-1] if selected_alt else None,
            "max_alt_error_m": max(selected_alt) if selected_alt else None,
            "final_yaw_error_deg": selected_yaw[-1] if selected_yaw else None,
            "max_yaw_error_deg": max(selected_yaw) if selected_yaw else None,
        },
        "mean_feature_loss_drop": _mean([row["feature_loss_drop"] for row in rows]),
    }
    _write_json(output_dir / "summary_metrics.json", summary)

    _plot_trajectory(rows, output_dir / "trajectory_gt_refine_only.png")
    _plot_error_curve(rows, output_dir / "xy_error_curve.png", "xy_error_m", "XY error (m)")
    _plot_error_curve(rows, output_dir / "alt_error_curve.png", "alt_error_m", "Altitude error (m)")
    _plot_error_curve(rows, output_dir / "yaw_error_curve.png", "yaw_error_deg", "Yaw error (deg)")
    _plot_error_curve(rows, output_dir / "accepted_yaw_curve.png", "yaw", "Yaw (deg)")
    plt.figure(figsize=(10, 4))
    plt.plot(
        np.arange(len(rows)),
        [abs(float(row["accepted_yaw_delta_deg"])) for row in rows],
        label="Accepted yaw step",
    )
    plt.xlabel("Frame")
    plt.ylabel("Yaw step (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "yaw_step_curve.png", dpi=150)
    plt.close()

    print(json.dumps(_safe_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
