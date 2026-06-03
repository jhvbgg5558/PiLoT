#!/usr/bin/env python3
"""Run three focused DOM/DSM raw-refine diagnostics for exif_test/0000."""

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

import direct_abs_cost_cuda  # noqa: E402
from pixloc.localization.localizer import RenderLocalizer  # noqa: E402
from pixloc.pixlib.datasets.view import read_image  # noqa: E402
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer  # noqa: E402
from pixloc.utils.dom_dsm.pose_adapter import apply_enu_offset  # noqa: E402
from pixloc.utils.get_depth import pad_to_multiple, zero_pad  # noqa: E402
from src.utils.pose_utils import load_initial_pose  # noqa: E402
from tools.compare_cuda_torch_feature_loss_fixed_poses import _prepare_refiner_features  # noqa: E402
from tools.diagnose_rebuilt_cuda_optimizer_trajectory import (  # noqa: E402
    _run_trajectory,
)
from tools.diagnose_yawfix_refinement_update import (  # noqa: E402
    BASE_EULER,
    _checkerboard,
    _edge_overlay,
    _get_raster_transformers,
    _make_overlay,
    _offset_between,
    _read_query_rgb,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import (  # noqa: E402
    _back_project,
    _depth_stats,
    _resize_query_for_refine,
    _setup_camera,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_SOURCE_EXPERIMENT = "docs/experiments/dom_dsm_prepare/domdsm_refine_single_0000_exif_test"
DEFAULT_P13_SUMMARY = "docs/experiments/dom_dsm_prepare/visual_objective_optimizer_p13/0000/summary_metrics.json"
DEFAULT_P12_SUMMARY = "docs/experiments/dom_dsm_prepare/batch_local_visual_safe_candidate_search_p12/0000/summary_metrics.json"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/raw_refine_diagnostics_exif_test_0000"
DEFAULT_WIDTH = 512
LINE_SCALES = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _norm(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _axis_sign_matches(a: float, b: float) -> bool:
    if abs(float(a)) < 1e-9 or abs(float(b)) < 1e-9:
        return abs(float(a)) < 1e-9 and abs(float(b)) < 1e-9
    return bool(np.sign(float(a)) == np.sign(float(b)))


def _scale_name(scale: float) -> str:
    sign = "plus" if scale >= 0 else "minus"
    value = str(abs(float(scale))).replace(".", "p")
    return f"scale_{sign}_{value}"


def _read_source_metrics(source_dir: Path) -> Dict[str, Dict[str, Any]]:
    names = [
        "initial",
        "raw_refined_full",
        "raw_refined_translation_initial_rotation",
        "corrected_downward_yaw",
        "safe_selected",
    ]
    out = {}
    for name in names:
        path = source_dir / name / "metrics.json"
        if path.exists():
            out[name] = _load_json(path)
    missing = [name for name in ["initial", "raw_refined_full"] if name not in out]
    if missing:
        raise FileNotFoundError(f"Missing source metrics in {source_dir}: {missing}")
    return out


def _setup_render_context(args: argparse.Namespace) -> Dict[str, Any]:
    config = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    default_confs = config["default_confs"]
    default_confs["cam_query"]["max_size"] = int(args.width)
    _loaded_euler, trans, origin_np = load_initial_pose(args.pose_file)
    trans = [float(x) for x in trans]
    euler = [float(x) for x in BASE_EULER]
    config["render_config"]["init_rot"] = euler
    config["render_config"]["init_trans"] = trans
    default_confs["refine"]["origin"] = origin_np
    query_resize_ratio, raw_query_camera, render_camera_gs, query_camera, render_camera = _setup_camera(config)
    width, height = int(render_camera_gs[0]), int(render_camera_gs[1])
    renderer = DOMDSMRenderer(config["render_config"])
    query_rgb = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
    to_raster, from_raster, raster_crs = _get_raster_transformers(config)
    return {
        "config": config,
        "default_confs": default_confs,
        "initial_trans": trans,
        "initial_euler": euler,
        "origin_np": origin_np,
        "query_resize_ratio": query_resize_ratio,
        "raw_query_camera": raw_query_camera,
        "render_camera_gs": render_camera_gs,
        "query_camera": query_camera,
        "render_camera": render_camera,
        "renderer": renderer,
        "query_rgb": query_rgb,
        "to_raster": to_raster,
        "from_raster": from_raster,
        "raster_crs": raster_crs,
        "width": width,
        "height": height,
    }


def _render_candidate(
    out_dir: Path,
    name: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: Sequence[float],
    euler: Sequence[float],
    checker_tile: int,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    cand_dir = out_dir / name
    cand_dir.mkdir(parents=True, exist_ok=True)
    render_rgb, depth = renderer.render([float(x) for x in trans], [float(x) for x in euler])
    overlay = _make_overlay(query_rgb, render_rgb)
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    checkerboard = _checkerboard(query_rgb, render_rgb, checker_tile)
    _write_rgb(cand_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(cand_dir / "overlay.png", overlay)
    _write_rgb(cand_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(cand_dir / "checkerboard.png", checkerboard)
    metrics = {
        "candidate": name,
        "translation_lon_lat_alt": [float(x) for x in trans],
        "euler_pitch_roll_yaw": [float(x) for x in euler],
        **_depth_stats(depth),
        **edge_metrics,
        **extra,
    }
    _write_json(cand_dir / "metrics.json", metrics)
    return metrics


def _run_experiment_2(args: argparse.Namespace, ctx: Dict[str, Any], source_metrics: Dict[str, Dict[str, Any]], root: Path) -> Dict[str, Any]:
    out_dir = root / "experiment_2_raw_delta_line_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    initial = source_metrics["initial"]
    raw = source_metrics["raw_refined_full"]
    raw_delta = [
        float(raw["offset_east_m"]),
        float(raw["offset_north_m"]),
        float(raw["offset_alt_m"]),
    ]
    rows = []
    initial_trans = [float(x) for x in initial["translation_lon_lat_alt"]]
    initial_euler = [float(x) for x in initial["euler_pitch_roll_yaw"]]
    for scale in LINE_SCALES:
        offsets = [float(scale) * x for x in raw_delta]
        trans = apply_enu_offset(
            initial_trans,
            offsets[0],
            offsets[1],
            offsets[2],
            ctx["to_raster"],
            ctx["from_raster"],
        )
        name = _scale_name(scale)
        metrics = _render_candidate(
            out_dir,
            name,
            ctx["renderer"],
            ctx["query_rgb"],
            trans,
            initial_euler,
            args.checker_tile,
            {
                "scale": float(scale),
                "translation_delta_east_m": offsets[0],
                "translation_delta_north_m": offsets[1],
                "translation_delta_alt_m": offsets[2],
                "euler_delta_pitch_deg": 0.0,
                "euler_delta_roll_deg": 0.0,
                "euler_delta_yaw_deg": 0.0,
                "source": "raw_delta_line_search_translation_only",
            },
        )
        rows.append(
            {
                "scale": float(scale),
                "translation_delta_east_m": offsets[0],
                "translation_delta_north_m": offsets[1],
                "translation_delta_alt_m": offsets[2],
                "euler_delta_pitch_deg": 0.0,
                "euler_delta_roll_deg": 0.0,
                "euler_delta_yaw_deg": 0.0,
                "chamfer": metrics["edge_chamfer"],
                "overlap": metrics["edge_overlap_ratio"],
                "valid_depth_ratio": metrics["valid_depth_ratio"],
            }
        )
    fields = [
        "scale",
        "translation_delta_east_m",
        "translation_delta_north_m",
        "translation_delta_alt_m",
        "euler_delta_pitch_deg",
        "euler_delta_roll_deg",
        "euler_delta_yaw_deg",
        "chamfer",
        "overlap",
        "valid_depth_ratio",
    ]
    _write_csv(out_dir / "line_search.csv", rows, fields)
    best_by_chamfer = min(rows, key=lambda item: float(item["chamfer"]))
    best_by_overlap = max(rows, key=lambda item: float(item["overlap"]))
    summary = {
        "experiment": "raw_delta_line_search",
        "raw_delta_east_north_alt_m": raw_delta,
        "scales": LINE_SCALES,
        "rows": rows,
        "best_by_chamfer": best_by_chamfer,
        "best_by_overlap": best_by_overlap,
        "scale_zero_matches_recent_initial": {
            "source_initial_chamfer": initial.get("edge_chamfer"),
            "source_initial_overlap": initial.get("edge_overlap_ratio"),
            "scale_zero_chamfer": next(r["chamfer"] for r in rows if r["scale"] == 0.0),
            "scale_zero_overlap": next(r["overlap"] for r in rows if r["scale"] == 0.0),
        },
    }
    _write_json(out_dir / "line_search.json", summary)
    _write_json(out_dir / "summary.json", summary)
    _make_contact_sheet(
        [out_dir / _scale_name(scale) / "overlay.png" for scale in LINE_SCALES],
        [f"scale {scale:+.2f}" for scale in LINE_SCALES],
        out_dir / "contact_sheet.png",
    )
    return summary


def _candidate_row(candidate: str, source: str, data: Dict[str, Any], raw_vec: Sequence[float]) -> Dict[str, Any]:
    vec3 = [
        float(data.get("offset_east_m", 0.0)),
        float(data.get("offset_north_m", 0.0)),
        float(data.get("offset_alt_m", 0.0)),
    ]
    vec2 = vec3[:2]
    return {
        "candidate": candidate,
        "source": source,
        "east_m": vec3[0],
        "north_m": vec3[1],
        "alt_m": vec3[2],
        "yaw_offset_deg": float(data.get("yaw_offset_deg", 0.0)),
        "delta_norm_xy_m": _norm(vec2),
        "delta_norm_3d_m": _norm(vec3),
        "cosine_vs_raw_xy": _cosine(vec2, raw_vec[:2]),
        "cosine_vs_raw_3d": _cosine(vec3, raw_vec),
        "axis_sign_east_matches_raw": _axis_sign_matches(vec3[0], raw_vec[0]),
        "axis_sign_north_matches_raw": _axis_sign_matches(vec3[1], raw_vec[1]),
        "axis_sign_alt_matches_raw": _axis_sign_matches(vec3[2], raw_vec[2]),
        "chamfer": data.get("edge_chamfer"),
        "overlap": data.get("edge_overlap_ratio"),
        "feature_loss_if_available": data.get("unweighted_feature_loss")
        or data.get("torch_feature_loss")
        or data.get("weighted_feature_loss_combined"),
    }


def _interpret_direction(cosine_xy: float) -> str:
    if cosine_xy > 0.5:
        return "direction_broadly_consistent"
    if cosine_xy < -0.5:
        return "direction_broadly_opposite"
    return "direction_inconsistent_or_axis_biased"


def _run_experiment_3(args: argparse.Namespace, source_metrics: Dict[str, Dict[str, Any]], root: Path) -> Dict[str, Any]:
    out_dir = root / "experiment_3_local_best_vs_raw_delta"
    out_dir.mkdir(parents=True, exist_ok=True)
    p13 = _load_json(REPO_ROOT / args.p13_summary)["selected"]
    p12 = _load_json(REPO_ROOT / args.p12_summary)["selected"]
    raw = source_metrics["raw_refined_full"]
    raw_tir = source_metrics.get("raw_refined_translation_initial_rotation", raw)
    raw_vec = [
        float(raw["offset_east_m"]),
        float(raw["offset_north_m"]),
        float(raw["offset_alt_m"]),
    ]
    rows = [
        _candidate_row("raw_refined_full", "domdsm_refine_single_0000_exif_test", raw, raw_vec),
        _candidate_row("raw_translation_initial_rotation", "domdsm_refine_single_0000_exif_test", raw_tir, raw_vec),
        _candidate_row("p13_selected", args.p13_summary, p13, raw_vec),
        _candidate_row("p12_selected", args.p12_summary, p12, raw_vec),
    ]
    fields = [
        "candidate",
        "source",
        "east_m",
        "north_m",
        "alt_m",
        "yaw_offset_deg",
        "delta_norm_xy_m",
        "delta_norm_3d_m",
        "cosine_vs_raw_xy",
        "cosine_vs_raw_3d",
        "axis_sign_east_matches_raw",
        "axis_sign_north_matches_raw",
        "axis_sign_alt_matches_raw",
        "chamfer",
        "overlap",
        "feature_loss_if_available",
    ]
    _write_csv(out_dir / "delta_comparison.csv", rows, fields)
    local_rows = [row for row in rows if row["candidate"] in {"p13_selected", "p12_selected"}]
    best = min(local_rows, key=lambda row: float(row["chamfer"]))
    axis_mismatches = [
        axis
        for axis, key in [
            ("east", "axis_sign_east_matches_raw"),
            ("north", "axis_sign_north_matches_raw"),
            ("alt", "axis_sign_alt_matches_raw"),
        ]
        if not bool(best[key])
    ]
    raw_step_too_large = bool(float(rows[0]["delta_norm_xy_m"]) > 2.0 * max(float(best["delta_norm_xy_m"]), 1e-9))
    summary = {
        "experiment": "local_best_delta_vs_raw_delta",
        "raw_delta_east_north_alt_m": raw_vec,
        "rows": rows,
        "best_local_by_chamfer": best,
        "direction_interpretation_vs_raw_xy": _interpret_direction(float(best["cosine_vs_raw_xy"])),
        "axis_sign_mismatch": axis_mismatches,
        "raw_step_too_large": raw_step_too_large,
    }
    _write_json(out_dir / "delta_comparison.json", {"rows": rows})
    _write_json(out_dir / "summary.json", summary)
    _draw_delta_vectors(rows, out_dir / "delta_vectors.png")
    _draw_axis_comparison(rows, out_dir / "axis_comparison.png")
    return summary


def _run_experiment_1(args: argparse.Namespace, ctx: Dict[str, Any], root: Path) -> Dict[str, Any]:
    out_dir = root / "experiment_1_cuda_visual_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = ctx["config"]
    default_confs = config["default_confs"]
    refine_conf = default_confs["refine"]
    conf = copy.deepcopy(default_confs["from_render_test"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Experiment 1 requires CUDA")

    cam_cfg = default_confs["cam_query"]
    query_image = read_image(
        args.query_image,
        scale=ctx["query_resize_ratio"],
        distortion=cam_cfg["distortion"],
        query_camera=ctx["raw_query_camera"],
    )
    color, depth = ctx["renderer"].render(ctx["initial_trans"], ctx["initial_euler"])
    color_for_refine = pad_to_multiple(color, 16) if default_confs.get("padding", False) else color
    origin = torch.tensor(ctx["origin_np"], device=device)
    query_camera = ctx["query_camera"].to(device)
    render_camera = ctx["render_camera"].to(device)
    p3d, t_render, t_grid_init, dd = _back_project(
        depth,
        ctx["initial_euler"],
        ctx["initial_trans"],
        ctx["initial_euler"],
        ctx["initial_trans"],
        ctx["render_camera_gs"],
        render_camera,
        origin,
        refine_conf["mul"],
        device,
        is_init=True,
    )
    localizer = RenderLocalizer(conf)
    refiner = localizer.refiner
    q_w, _ = query_camera.size
    query_image_for_refine = _resize_query_for_refine(query_image, ctx["render_camera_gs"])
    query_feature_image = zero_pad(int(q_w.item()), query_image_for_refine)
    render_feature_image = zero_pad(int(q_w.item()), color_for_refine)
    with torch.no_grad():
        features_ref_raw, scales_ref = refiner.dense_feature_extraction(render_feature_image)
        features_q_raw, scales_q = refiner.dense_feature_extraction(query_feature_image)
    features_q, features_ref, weights_q, weights_ref = _prepare_refiner_features(
        refiner,
        features_q_raw,
        features_ref_raw,
    )
    if weights_q is None or weights_ref is None:
        raise RuntimeError("Experiment 1 requires uncertainty weights for CUDA optimizer path")
    opt = refiner.optimizer
    if isinstance(opt, (list, tuple)):
        opt_getter = lambda level: opt[refiner.conf.layer_indices[level]] if refiner.conf.layer_indices else opt[level]
    else:
        opt_getter = lambda _level: opt

    trajectory, _final_pose, meta = _run_trajectory(
        "trajectory",
        t_grid_init,
        opt_getter,
        features_q,
        features_ref,
        weights_q,
        weights_ref,
        features_q_raw,
        features_ref_raw,
        scales_q,
        scales_ref,
        p3d,
        t_render,
        query_camera,
        render_camera,
        dd,
        refine_conf["mul"],
        origin,
        ctx["initial_trans"],
        ctx["to_raster"],
        ctx["renderer"],
        ctx["query_rgb"],
        out_dir,
        not args.skip_iteration_images,
    )
    rows = []
    for item in trajectory:
        pose = item["best_pose"]
        offsets = [
            float(pose["east_m"]),
            float(pose["north_m"]),
            float(pose["alt_offset_m"]),
        ]
        euler = np.asarray(pose["euler_pitch_roll_yaw"], dtype=np.float64)
        initial_euler = np.asarray(ctx["initial_euler"], dtype=np.float64)
        euler_delta = (euler - initial_euler).tolist()
        metrics = item["best_metrics"]
        rows.append(
            {
                "iteration": int(item["global_iteration"]),
                "level": item["level"],
                "local_iteration": item["local_iteration"],
                "cuda_loss": metrics["cuda_w_loss_mean"],
                "torch_feature_loss": metrics["torch_feature_loss"],
                "translation_delta_east_m": offsets[0],
                "translation_delta_north_m": offsets[1],
                "translation_delta_alt_m": offsets[2],
                "euler_delta_pitch_deg": euler_delta[0],
                "euler_delta_roll_deg": euler_delta[1],
                "euler_delta_yaw_deg": euler_delta[2],
                "chamfer": metrics["visual_chamfer"],
                "overlap": metrics["visual_overlap"],
            }
        )
    fields = [
        "iteration",
        "level",
        "local_iteration",
        "cuda_loss",
        "torch_feature_loss",
        "translation_delta_east_m",
        "translation_delta_north_m",
        "translation_delta_alt_m",
        "euler_delta_pitch_deg",
        "euler_delta_roll_deg",
        "euler_delta_yaw_deg",
        "chamfer",
        "overlap",
    ]
    _write_csv(out_dir / "trajectory.csv", rows, fields)
    _write_json(out_dir / "trajectory.json", {"trajectory": trajectory, "rows": rows})
    start = rows[0]
    end = rows[-1]
    summary = {
        "experiment": "cuda_internal_loss_vs_visual_metric",
        "num_records": len(rows),
        "cuda_loss_start": start["cuda_loss"],
        "cuda_loss_end": end["cuda_loss"],
        "torch_feature_loss_start": start["torch_feature_loss"],
        "torch_feature_loss_end": end["torch_feature_loss"],
        "chamfer_start": start["chamfer"],
        "chamfer_end": end["chamfer"],
        "overlap_start": start["overlap"],
        "overlap_end": end["overlap"],
        "cuda_loss_decreased": float(end["cuda_loss"]) < float(start["cuda_loss"]),
        "chamfer_improved": float(end["chamfer"]) < float(start["chamfer"]),
        "overlap_improved": float(end["overlap"]) > float(start["overlap"]),
        "internal_loss_improves_but_visual_worse": (
            float(end["cuda_loss"]) < float(start["cuda_loss"])
            and (
                float(end["chamfer"]) > float(start["chamfer"])
                or float(end["overlap"]) < float(start["overlap"])
            )
        ),
        "topk_events": meta.get("topk_events", []),
        "cuda": {
            "module_path": getattr(direct_abs_cost_cuda, "__file__", None),
            "has_residual": hasattr(direct_abs_cost_cuda, "residual_jacobian_batch_quat_cuda"),
            "has_step": hasattr(direct_abs_cost_cuda, "optimizer_step_cuda"),
        },
    }
    _write_json(out_dir / "summary.json", summary)
    _copy_start_end_images(out_dir)
    return summary


def _copy_start_end_images(out_dir: Path) -> None:
    image_dirs = sorted((out_dir / "trajectory").glob("iter_*"))
    if not image_dirs:
        return
    for label, source_dir in [("start", image_dirs[0]), ("end", image_dirs[-1])]:
        dest = out_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["rendered_rgb.png", "edge_overlay.png"]:
            src = source_dir / name
            if src.exists():
                shutil.copy2(src, dest / name)


def _make_contact_sheet(image_paths: Sequence[Path], labels: Sequence[str], output_path: Path, thumb_w: int = 520) -> None:
    thumbs = []
    for path, label in zip(image_paths, labels):
        bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        thumb = cv2.resize(bgr, (thumb_w, int(h * thumb_w / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(thumb, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = 2
    rows = []
    blank = np.zeros_like(thumbs[0])
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx : idx + cols]
        while len(row) < cols:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    cv2.imwrite(os.fspath(output_path), np.vstack(rows))


def _draw_delta_vectors(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    canvas = np.full((720, 720, 3), 255, dtype=np.uint8)
    center = np.array([360, 360], dtype=np.float64)
    max_abs = max(max(abs(float(r["east_m"])), abs(float(r["north_m"]))) for r in rows)
    scale = 260.0 / max(max_abs, 1.0)
    colors = {
        "raw_refined_full": (40, 40, 230),
        "raw_translation_initial_rotation": (40, 120, 230),
        "p13_selected": (40, 170, 40),
        "p12_selected": (210, 120, 40),
    }
    cv2.line(canvas, (80, 360), (640, 360), (180, 180, 180), 1)
    cv2.line(canvas, (360, 80), (360, 640), (180, 180, 180), 1)
    cv2.putText(canvas, "East +", (600, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(canvas, "North +", (370, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
    for i, row in enumerate(rows):
        end = center + np.array([float(row["east_m"]) * scale, -float(row["north_m"]) * scale])
        color = colors.get(row["candidate"], (0, 0, 0))
        cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(end.astype(int)), color, 3, tipLength=0.12)
        cv2.putText(canvas, row["candidate"], (40, 40 + i * 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.fspath(output_path), canvas)


def _draw_axis_comparison(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    canvas = np.full((720, 960, 3), 255, dtype=np.uint8)
    axes = ["east_m", "north_m", "alt_m"]
    colors = [(80, 80, 220), (80, 170, 80), (220, 140, 60), (160, 80, 160)]
    max_abs = max(abs(float(row[axis])) for row in rows for axis in axes)
    scale = 110.0 / max(max_abs, 1.0)
    for a_idx, axis in enumerate(axes):
        x0 = 180 + a_idx * 260
        cv2.line(canvas, (x0, 120), (x0, 620), (180, 180, 180), 1)
        cv2.putText(canvas, axis, (x0 - 50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 50, 50), 2, cv2.LINE_AA)
        for r_idx, row in enumerate(rows):
            y = 160 + r_idx * 95
            value = float(row[axis])
            length = int(value * scale)
            color = colors[r_idx % len(colors)]
            cv2.line(canvas, (x0, y), (x0 + length, y), color, 16)
            cv2.putText(canvas, f"{value:+.2f}", (x0 + length + 10, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            if a_idx == 0:
                cv2.putText(canvas, row["candidate"], (20, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.imwrite(os.fspath(output_path), canvas)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--source-experiment", default=DEFAULT_SOURCE_EXPERIMENT)
    parser.add_argument("--p13-summary", default=DEFAULT_P13_SUMMARY)
    parser.add_argument("--p12-summary", default=DEFAULT_P12_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument("--skip-experiment-1", action="store_true")
    parser.add_argument("--skip-iteration-images", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    torch.manual_seed(0)
    np.random.seed(0)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run_log: Dict[str, Any] = {
        "config": args.config,
        "query_image": args.query_image,
        "pose_file": args.pose_file,
        "source_experiment": args.source_experiment,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
    try:
        source_metrics = _read_source_metrics(REPO_ROOT / args.source_experiment)
        ctx = _setup_render_context(args)
        summary: Dict[str, Any] = {
            "output_dir": os.fspath(output_dir),
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "config": args.config,
            "width": args.width,
            "render_size": [ctx["width"], ctx["height"]],
            "source_experiment": args.source_experiment,
        }
        run_log["failure_stage"] = "experiment_2"
        summary["experiment_2"] = _run_experiment_2(args, ctx, source_metrics, output_dir)
        run_log["failure_stage"] = "experiment_3"
        summary["experiment_3"] = _run_experiment_3(args, source_metrics, output_dir)
        if not args.skip_experiment_1:
            run_log["failure_stage"] = "experiment_1"
            summary["experiment_1"] = _run_experiment_1(args, ctx, output_dir)
        else:
            summary["experiment_1"] = {"skipped": True}
        summary["total_time_sec"] = time.perf_counter() - started
        _write_json(output_dir / "summary.json", summary)
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
