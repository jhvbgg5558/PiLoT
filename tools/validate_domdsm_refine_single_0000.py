#!/usr/bin/env python3
"""One-shot DOM+DSM 0000 single-frame PiLoT refinement validation."""

import argparse
import copy
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
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CUDA_EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
if CUDA_EXT_DIR.exists() and str(CUDA_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(CUDA_EXT_DIR))

from pixloc.localization.localizer import RenderLocalizer
from pixloc.pixlib.datasets.view import read_image
from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.domdsm_refine import run_domdsm_back_project
from pixloc.utils.dom_dsm.feature_loss_debug import (
    compute_feature_residual_loss,
    extract_pilot_features,
    save_residual_debug_visualization,
)
from pixloc.utils.dom_dsm.pose_adapter import (
    apply_enu_offset,
    compute_enu_delta_m,
    make_safe_domdsm_pose_from_refined,
    make_downward_euler_from_yaw,
    refined_yaw_to_downward_yaw,
)
from pixloc.utils.get_depth import pad_to_multiple
from src.utils.pose_utils import load_initial_pose, load_pose_dict
from tools.diagnose_domdsm_torch_feature_loss import _build_query_pose
from tools.diagnose_yawfix_refinement_update import (
    _checkerboard,
    _edge_overlay,
    _get_raster_transformers,
    _make_overlay,
    _read_query_rgb,
    _safe_jsonable,
    _write_rgb,
)
from tools.run_dom_dsm_single_full import (
    _back_project,
    _depth_stats,
    _format_pose_line,
    _resize_query_for_refine,
    _setup_camera,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_QUERY_IMAGE = "data_caiwangcun/query/images/exif_test_16x9/0000.jpg"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/domdsm_refine_single_0000"
DEFAULT_BASE_EULER = [0.0, 180.0, 29.2]
EPS = 1e-9


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_float_list(values: Any) -> List[float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    return np.asarray(_safe_jsonable(values), dtype=np.float64).reshape(-1).tolist()


def _candidate_passes_gate(candidate: Dict[str, Any], initial: Dict[str, Any]) -> bool:
    return (
        float(candidate["edge_chamfer"]) <= float(initial["edge_chamfer"]) + EPS
        and float(candidate["edge_overlap_ratio"]) + EPS >= float(initial["edge_overlap_ratio"])
    )


def _candidate_objective(candidate: Dict[str, Any]) -> float:
    return float(candidate["edge_chamfer"]) - float(candidate["edge_overlap_ratio"])


def _is_initial_equivalent(candidate: Dict[str, Any]) -> bool:
    return (
        abs(float(candidate.get("offset_east_m", 0.0))) < EPS
        and abs(float(candidate.get("offset_north_m", 0.0))) < EPS
        and abs(float(candidate.get("offset_alt_m", 0.0))) < EPS
        and abs(float(candidate.get("yaw_offset_deg", 0.0))) < EPS
    )


def _render_candidate(
    name: str,
    renderer: DOMDSMRenderer,
    query_rgb: np.ndarray,
    trans: Sequence[float],
    euler: Sequence[float],
    output_dir: Path,
    checker_tile: int,
    extra: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    cand_dir = output_dir / name
    cand_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    render_rgb, depth = renderer.render(list(map(float, trans)), list(map(float, euler)))
    render_time = time.perf_counter() - t0

    _write_rgb(cand_dir / "rendered_rgb.png", render_rgb)
    _write_rgb(cand_dir / "overlay.png", _make_overlay(query_rgb, render_rgb))
    edge_overlay, edge_metrics = _edge_overlay(query_rgb, render_rgb)
    _write_rgb(cand_dir / "edge_overlay.png", edge_overlay)
    _write_rgb(cand_dir / "checkerboard.png", _checkerboard(query_rgb, render_rgb, checker_tile))

    metrics = {
        "candidate": name,
        "translation_lon_lat_alt": [float(x) for x in trans],
        "euler_pitch_roll_yaw": [float(x) for x in euler],
        "render_time_sec": render_time,
        **_depth_stats(depth),
        **edge_metrics,
        **extra,
    }
    return render_rgb, depth, metrics


def _compute_candidate_feature_loss(
    name: str,
    render_rgb: np.ndarray,
    depth: np.ndarray,
    trans: Sequence[float],
    euler: Sequence[float],
    base_trans: Sequence[float],
    localizer: RenderLocalizer,
    feat_query: Dict[str, Any],
    query_camera: Any,
    render_camera: Any,
    render_camera_gs: np.ndarray,
    origin: torch.Tensor,
    refine_mul: float,
    device: str,
    output_dir: Path,
    query_rgb: np.ndarray,
    padding: bool,
    num_points: int,
    sampling_mode: str,
) -> Dict[str, Any]:
    p3d, T_render, _unused, dd, bp_debug = run_domdsm_back_project(
        depth,
        render_rgb,
        list(map(float, euler)),
        list(map(float, trans)),
        list(map(float, euler)),
        list(map(float, trans)),
        render_camera_gs,
        render_camera,
        origin,
        refine_mul,
        device,
        num_samples=num_points,
        sampling_mode=sampling_mode,
        is_init=False,
        seed=0,
    )
    T_query = _build_query_pose(
        trans,
        euler,
        base_trans,
        origin,
        refine_mul,
        dd,
        device,
    )
    render_for_refine = pad_to_multiple(render_rgb, 16) if padding else render_rgb
    feat_render = extract_pilot_features(localizer, render_for_refine, "render")
    loss = compute_feature_residual_loss(
        feat_query["features"],
        feat_render["features"],
        feat_query["scales"],
        feat_render["scales"],
        p3d,
        T_query,
        T_render,
        query_camera,
        render_camera,
    )
    save_residual_debug_visualization(
        output_dir / name,
        query_rgb,
        render_rgb,
        loss["points_query"],
        loss["points_render"],
        loss["residual_per_point"],
        loss["valid_mask"],
    )
    return {
        "torch_feature_loss": loss["loss_total"],
        "loss_by_level": loss["loss_by_level"],
        "num_valid_by_level": loss["num_valid_by_level"],
        "valid_ratio_by_level": loss["valid_ratio_by_level"],
        "feature_sampling_debug": {
            key: value for key, value in bp_debug.items() if key != "weight"
        },
    }


def _interpolate_fixed_alt(
    initial_trans: Sequence[float],
    refined_trans: Sequence[float],
    scale: float,
    to_raster: Any,
    from_raster: Any,
) -> Tuple[List[float], List[float]]:
    ix, iy = to_raster.transform(float(initial_trans[0]), float(initial_trans[1]))
    rx, ry = to_raster.transform(float(refined_trans[0]), float(refined_trans[1]))
    x = ix + float(scale) * (rx - ix)
    y = iy + float(scale) * (ry - iy)
    lon, lat = from_raster.transform(x, y)
    trans = [float(lon), float(lat), float(initial_trans[2])]
    offsets = [float(x - ix), float(y - iy), 0.0]
    return trans, offsets


def _copy_selected_images(source_dir: Path, selected_dir: Path) -> None:
    selected_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "rendered_rgb.png",
        "overlay.png",
        "checkerboard.png",
        "edge_overlay.png",
        "query_residual_overlay.png",
        "render_residual_overlay.png",
        "residual_histogram.png",
        "residual_points.json",
    ]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, selected_dir / name)


def _write_pose_outputs(
    output_dir: Path,
    image_name: str,
    initial: Dict[str, Any],
    raw_refined: Dict[str, Any],
    selected: Dict[str, Any],
) -> None:
    (output_dir / "result_pose_raw_refined.txt").write_text(
        "\n".join(
            [
                "# initial",
                _format_pose_line(
                    image_name,
                    initial["translation_lon_lat_alt"],
                    initial["euler_pitch_roll_yaw"],
                ),
                "# raw_refined_full",
                _format_pose_line(
                    image_name,
                    raw_refined["translation_lon_lat_alt"],
                    raw_refined["euler_pitch_roll_yaw"],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "result_pose_safe_selected.txt").write_text(
        "\n".join(
            [
                "# safe_selected",
                f"method: {selected['candidate']}",
                f"safe_gate_pass: {selected['safe_gate_pass']}",
                _format_pose_line(
                    image_name,
                    selected["translation_lon_lat_alt"],
                    selected["euler_pitch_roll_yaw"],
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_summary_md(output_dir: Path, summary: Dict[str, Any]) -> None:
    candidates = summary["candidates"]
    selected = summary["safe_gate"]["selected_candidate"]
    key_names = [
        "initial",
        "raw_refined_full",
        "corrected_downward_yaw",
        "swap_xy_freeze_alt_corrected_downward_yaw",
        "safe_selected",
    ]
    rows = []
    seen = set()
    for name in key_names:
        if name in seen or name not in candidates:
            continue
        seen.add(name)
        rows.append(candidates[name])

    lines = [
        "# DOM+DSM 0000 Single-Frame Refine Validation",
        "",
        "## Inputs",
        f"- config: `{summary['config']}`",
        f"- query image: `{summary['query_image']}`",
        f"- pose file: `{summary['pose_file']}`",
        f"- output dir: `{summary['output_dir']}`",
        "",
        "## Pose Summary",
        "| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        trans = row["translation_lon_lat_alt"]
        euler = row["euler_pitch_roll_yaw"]
        lines.append(
            f"| {row['candidate']} | {_fmt(trans[0], 10)} | {_fmt(trans[1], 10)} | "
            f"{_fmt(trans[2], 3)} | {_fmt(euler[0], 6)} | {_fmt(euler[1], 6)} | "
            f"{_fmt(euler[2], 6)} | {_fmt(row.get('offset_east_m'), 3)} | "
            f"{_fmt(row.get('offset_north_m'), 3)} | {_fmt(row.get('offset_alt_m'), 3)} |"
        )

    lines += [
        "",
        "## Metrics",
        "| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {_fmt(row.get('torch_feature_loss'), 6)} | "
            f"{_fmt(row.get('edge_chamfer'), 6)} | "
            f"{_fmt(row.get('edge_overlap_ratio'), 6)} | "
            f"{row.get('safe_gate_pass')} | {row.get('selected_by_safe_gate')} |"
        )

    raw = candidates["raw_refined_full"]
    initial = candidates["initial"]
    raw_degrades = (
        float(raw["edge_chamfer"]) > float(initial["edge_chamfer"]) + EPS
        or float(raw["edge_overlap_ratio"]) + EPS < float(initial["edge_overlap_ratio"])
    )
    lines += [
        "",
        "## Raw Refined Delta",
        f"- raw refined translation: `{raw['translation_lon_lat_alt']}`",
        f"- raw refined euler pitch/roll/yaw: `{raw['euler_pitch_roll_yaw']}`",
        f"- raw refined ENU delta m: `{summary['raw_refined_delta_east_north_alt_m']}`",
        "",
        "## Corrected Downward Yaw",
        f"- raw refined yaw: `{summary['yaw_conversion']['raw_refined_yaw']}`",
        f"- corrected downward yaw: `{summary['yaw_conversion']['corrected_downward_yaw']}`",
        "- corrected candidate euler: "
        f"`{candidates['corrected_downward_yaw']['euler_pitch_roll_yaw']}`",
        "",
        "## Swap-XY Adapter",
        "- raw refined ENU delta m: "
        f"`{summary['swap_xy_adapter']['raw_delta_east_north_alt_m']}`",
        "- applied swapped ENU delta m: "
        f"`{summary['swap_xy_adapter']['applied_delta_east_north_alt_m']}`",
        "- candidate euler: "
        f"`{candidates['swap_xy_freeze_alt_corrected_downward_yaw']['euler_pitch_roll_yaw']}`",
        "",
        "## Safe Gate",
        f"- selected candidate: `{selected}`",
        f"- policy: `{summary['safe_gate']['policy']}`",
        f"- raw refined visually degrades initial: `{raw_degrades}`",
        f"- recommendation: `{summary['safe_gate']['recommendation']}`",
    ]
    if raw_degrades:
        lines.append("- raw refined should not be accepted directly under the visual gate.")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-image", default=DEFAULT_QUERY_IMAGE)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-points", type=int, default=500)
    parser.add_argument("--sampling-mode", default="combined")
    parser.add_argument("--checker-tile", type=int, default=32)
    parser.add_argument(
        "--line-scales",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--allow-raw-full-pose",
        action="store_true",
        help="Allow raw_refined_full to enter safe-gate selection. Debug only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "refinement_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log: Dict[str, Any] = {
        "config": args.config,
        "query_image": args.query_image,
        "pose_file": args.pose_file,
        "output_dir": os.fspath(output_dir),
        "failure_stage": None,
        "traceback": None,
    }
    stage = "start"
    start_total = time.perf_counter()

    try:
        stage = "load_config"
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        default_confs = config["default_confs"]
        default_confs["cam_query"]["max_size"] = args.width
        refine_conf = default_confs["refine"]
        conf = copy.deepcopy(default_confs["from_render_test"])

        stage = "setup_camera"
        (
            query_resize_ratio,
            raw_query_camera,
            render_camera_gs,
            query_camera,
            render_camera,
        ) = _setup_camera(config)
        width, height = int(render_camera_gs[0]), int(render_camera_gs[1])

        stage = "load_pose"
        _loaded_euler, initial_trans, origin_np = load_initial_pose(args.pose_file)
        initial_trans = [float(x) for x in initial_trans]
        base_yaw = float(DEFAULT_BASE_EULER[2])
        initial_euler = make_downward_euler_from_yaw(base_yaw)
        config["render_config"]["init_rot"] = initial_euler
        config["render_config"]["init_trans"] = initial_trans
        refine_conf["origin"] = origin_np
        gt_pose_dict = load_pose_dict(args.pose_file, origin=origin_np)
        image_name = Path(args.query_image).name

        stage = "load_query"
        cam_cfg = default_confs["cam_query"]
        query_image = read_image(
            args.query_image,
            scale=query_resize_ratio,
            distortion=cam_cfg["distortion"],
            query_camera=raw_query_camera,
        )
        query_for_visual = _read_query_rgb(REPO_ROOT / args.query_image, width, height)
        query_for_refine = _resize_query_for_refine(query_image, render_camera_gs)
        query_for_feature = (
            pad_to_multiple(query_for_refine, 16)
            if default_confs.get("padding", False)
            else query_for_refine
        )

        stage = "init_renderer"
        renderer = DOMDSMRenderer(config["render_config"])
        to_raster, from_raster, raster_crs = _get_raster_transformers(config)

        stage = "init_localizer"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        origin = torch.tensor(origin_np, device=device)
        query_camera = query_camera.to(device)
        render_camera = render_camera.to(device)
        localizer = RenderLocalizer(conf)
        feat_query = extract_pilot_features(localizer, query_for_feature, "query")

        stage = "render_initial"
        initial_render, initial_depth, initial_metrics = _render_candidate(
            "initial",
            renderer,
            query_for_visual,
            initial_trans,
            initial_euler,
            output_dir,
            args.checker_tile,
            {
                "source": "initial",
                "offset_east_m": 0.0,
                "offset_north_m": 0.0,
                "offset_alt_m": 0.0,
                "yaw_offset_deg": 0.0,
            },
        )

        stage = "back_project_initial_for_refine"
        p3d, T_w2c, T_init, dd = _back_project(
            initial_depth,
            initial_euler,
            initial_trans,
            initial_euler,
            initial_trans,
            render_camera_gs,
            render_camera,
            origin,
            refine_conf["mul"],
            device,
            is_init=True,
        )
        color_for_refine = (
            pad_to_multiple(initial_render, 16)
            if default_confs.get("padding", False)
            else initial_render
        )

        stage = "run_query"
        last_frame_info = {"observations": [], "refine_conf": refine_conf}
        t0 = time.perf_counter()
        ret = localizer.run_query(
            args.query_image,
            query_camera,
            render_camera,
            color_for_refine,
            query_T=T_init,
            render_T=T_w2c,
            Points_3D_ECEF=p3d,
            query_resize_ratio=query_resize_ratio,
            dd=dd,
            gt_pose_dict=gt_pose_dict,
            last_frame_info=last_frame_info,
            image_query=query_for_refine,
        )
        run_query_time = time.perf_counter() - t0
        if not ret.get("success", False):
            raise RuntimeError("run_query returned success=False")

        refined_trans = _as_float_list(ret["translation"])
        refined_euler = _as_float_list(ret["euler_angles"])
        raw_delta = compute_enu_delta_m(initial_trans, refined_trans, to_raster)
        raw_refined_yaw = float(refined_euler[2])
        initial_pose = make_safe_domdsm_pose_from_refined(
            initial_trans,
            initial_euler,
            refined_trans,
            refined_euler,
            "initial",
        )
        raw_debug_pose = make_safe_domdsm_pose_from_refined(
            initial_trans,
            initial_euler,
            refined_trans,
            refined_euler,
            "raw_debug_only",
        )
        corrected_pose = make_safe_domdsm_pose_from_refined(
            initial_trans,
            initial_euler,
            refined_trans,
            refined_euler,
            "corrected_downward_yaw_freeze_alt_pitch_roll",
        )
        swap_xy_pose = make_safe_domdsm_pose_from_refined(
            initial_trans,
            initial_euler,
            refined_trans,
            refined_euler,
            "swap_xy_freeze_alt_corrected_downward_yaw",
            to_raster=to_raster,
            from_raster=from_raster,
        )
        corrected_downward_yaw = float(corrected_pose["corrected_downward_yaw"])
        corrected_trans = corrected_pose["translation_lon_lat_alt"]
        corrected_euler = corrected_pose["euler_pitch_roll_yaw"]
        swap_xy_trans = swap_xy_pose["translation_lon_lat_alt"]
        swap_xy_euler = swap_xy_pose["euler_pitch_roll_yaw"]
        swap_xy_applied_delta = swap_xy_pose["applied_delta_east_north_alt_m"]

        log.update(
            {
                "run_query_success": True,
                "run_query_time_sec": run_query_time,
                "raw_ret": _safe_jsonable(ret),
                "initial_translation_lon_lat_alt": initial_trans,
                "initial_euler_pitch_roll_yaw": initial_euler,
                "refined_translation_lon_lat_alt": refined_trans,
                "refined_euler_pitch_roll_yaw": refined_euler,
                "raw_refined_delta_east_north_alt_m": raw_delta,
                "raster_crs": raster_crs,
                "torch": {
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,
                    "device": device,
                    "gpu_name": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None,
                    "gpu_capability": torch.cuda.get_device_capability(0)
                    if torch.cuda.is_available()
                    else None,
                },
            }
        )
        _write_json(raw_dir / "run_log.json", log)

        stage = "build_candidates"
        candidates = [
            {
                "name": "initial",
                "source": "initial",
                "trans": initial_pose["translation_lon_lat_alt"],
                "euler": initial_pose["euler_pitch_roll_yaw"],
                "offset": [0.0, 0.0, 0.0],
                "yaw_offset": 0.0,
                "debug_only": initial_pose["debug_only"],
                "formal_safe_candidate": initial_pose["formal_safe_candidate"],
                "adapter_mode": initial_pose["mode"],
            },
            {
                "name": "raw_refined_full",
                "source": "raw_refined",
                "trans": raw_debug_pose["translation_lon_lat_alt"],
                "euler": raw_debug_pose["euler_pitch_roll_yaw"],
                "offset": raw_delta,
                "yaw_offset": corrected_downward_yaw - base_yaw,
                "debug_only": True,
                "formal_safe_candidate": bool(args.allow_raw_full_pose),
                "adapter_mode": raw_debug_pose["mode"],
            },
            {
                "name": "raw_refined_translation_initial_rotation",
                "source": "raw_refined",
                "trans": refined_trans,
                "euler": initial_euler,
                "offset": raw_delta,
                "yaw_offset": 0.0,
                "debug_only": True,
                "formal_safe_candidate": False,
                "adapter_mode": "raw_refined_translation_initial_rotation_debug",
            },
            {
                "name": "corrected_downward_yaw",
                "source": "corrected_refined",
                "trans": corrected_trans,
                "euler": corrected_euler,
                "offset": [raw_delta[0], raw_delta[1], 0.0],
                "yaw_offset": corrected_downward_yaw - base_yaw,
                "debug_only": corrected_pose["debug_only"],
                "formal_safe_candidate": corrected_pose["formal_safe_candidate"],
                "adapter_mode": corrected_pose["mode"],
            },
            {
                "name": "swap_xy_freeze_alt_corrected_downward_yaw",
                "source": "swap_xy_adapter",
                "trans": swap_xy_trans,
                "euler": swap_xy_euler,
                "offset": swap_xy_applied_delta,
                "yaw_offset": corrected_downward_yaw - base_yaw,
                "debug_only": swap_xy_pose["debug_only"],
                "formal_safe_candidate": swap_xy_pose["formal_safe_candidate"],
                "adapter_mode": swap_xy_pose["mode"],
                "adapter_metadata": {
                    "raw_delta_east_north_alt_m": swap_xy_pose[
                        "raw_delta_east_north_alt_m"
                    ],
                    "applied_delta_east_north_alt_m": swap_xy_applied_delta,
                    "corrected_downward_yaw": swap_xy_pose["corrected_downward_yaw"],
                },
            },
        ]
        for scale in args.line_scales:
            trans, offset = _interpolate_fixed_alt(
                initial_trans,
                refined_trans,
                float(scale),
                to_raster,
                from_raster,
            )
            scale_name = f"line_search_scale_{float(scale):.2f}".replace(".", "p")
            candidates.append(
                {
                    "name": scale_name,
                    "source": "line_search_fixed_initial_alt",
                    "trans": trans,
                    "euler": initial_euler,
                    "offset": offset,
                    "yaw_offset": 0.0,
                    "scale": float(scale),
                    "debug_only": False,
                    "formal_safe_candidate": True,
                    "adapter_mode": "line_search_fixed_initial_alt",
                }
            )

        stage = "score_candidates"
        metrics_by_name: Dict[str, Dict[str, Any]] = {}
        for cand in candidates:
            render_rgb, depth, metrics = _render_candidate(
                cand["name"],
                renderer,
                query_for_visual,
                cand["trans"],
                cand["euler"],
                output_dir,
                args.checker_tile,
                {
                    "source": cand["source"],
                    "offset_east_m": cand["offset"][0],
                    "offset_north_m": cand["offset"][1],
                    "offset_alt_m": cand["offset"][2],
                    "yaw_offset_deg": cand["yaw_offset"],
                    "line_search_scale": cand.get("scale"),
                    "debug_only": cand["debug_only"],
                    "formal_safe_candidate": cand["formal_safe_candidate"],
                    "adapter_mode": cand["adapter_mode"],
                    **cand.get("adapter_metadata", {}),
                },
            )
            feature_metrics = _compute_candidate_feature_loss(
                cand["name"],
                render_rgb,
                depth,
                cand["trans"],
                cand["euler"],
                initial_trans,
                localizer,
                feat_query,
                query_camera,
                render_camera,
                render_camera_gs,
                origin,
                refine_conf["mul"],
                device,
                output_dir,
                query_for_visual,
                default_confs.get("padding", False),
                args.num_points,
                args.sampling_mode,
            )
            metrics.update(feature_metrics)
            metrics_by_name[cand["name"]] = metrics

        initial = metrics_by_name["initial"]
        for metrics in metrics_by_name.values():
            metrics["safe_gate_pass"] = _candidate_passes_gate(metrics, initial)
            metrics["safe_gate_objective"] = _candidate_objective(metrics)
            metrics["selected_by_safe_gate"] = False

        passing = [
            m
            for name, m in metrics_by_name.items()
            if (
                name != "initial"
                and m["safe_gate_pass"]
                and m["formal_safe_candidate"]
                and not _is_initial_equivalent(m)
            )
        ]
        if passing:
            selected = sorted(
                passing,
                key=lambda m: (
                    float(m["safe_gate_objective"]),
                    float(m["edge_chamfer"]),
                    -float(m["edge_overlap_ratio"]),
                ),
            )[0]
        else:
            selected = initial
        selected["selected_by_safe_gate"] = True
        selected_name = selected["candidate"]

        for name, metrics in metrics_by_name.items():
            _write_json(output_dir / name / "metrics.json", metrics)

        selected_dir = output_dir / "safe_selected"
        _copy_selected_images(output_dir / selected_name, selected_dir)
        safe_selected_metrics = dict(selected)
        safe_selected_metrics["candidate"] = "safe_selected"
        safe_selected_metrics["selected_source_candidate"] = selected_name
        safe_selected_metrics["selected_by_safe_gate"] = True
        safe_selected_metrics["formal_safe_candidate"] = True
        metrics_by_name["safe_selected"] = safe_selected_metrics
        _write_json(selected_dir / "metrics.json", safe_selected_metrics)

        raw_refined = metrics_by_name["raw_refined_full"]
        _write_pose_outputs(output_dir, image_name, initial, raw_refined, selected)

        raw_degrades = (
            float(raw_refined["edge_chamfer"]) > float(initial["edge_chamfer"]) + EPS
            or float(raw_refined["edge_overlap_ratio"]) + EPS
            < float(initial["edge_overlap_ratio"])
        )
        summary = {
            "config": args.config,
            "query_image": args.query_image,
            "pose_file": args.pose_file,
            "output_dir": os.fspath(output_dir),
            "image": image_name,
            "raster_crs": raster_crs,
            "renderer_width": args.width,
            "num_points": args.num_points,
            "sampling_mode": args.sampling_mode,
            "initial_translation_lon_lat_alt": initial_trans,
            "initial_euler_pitch_roll_yaw": initial_euler,
            "raw_refined_translation_lon_lat_alt": refined_trans,
            "raw_refined_euler_pitch_roll_yaw": refined_euler,
            "raw_refined_delta_east_north_alt_m": raw_delta,
            "yaw_conversion": {
                "raw_refined_yaw": raw_refined_yaw,
                "corrected_downward_yaw": corrected_downward_yaw,
                "formula": "normalize(raw_refined_yaw + 180)",
            },
            "swap_xy_adapter": {
                "mode": swap_xy_pose["mode"],
                "raw_delta_east_north_alt_m": swap_xy_pose[
                    "raw_delta_east_north_alt_m"
                ],
                "applied_delta_east_north_alt_m": swap_xy_applied_delta,
                "corrected_downward_yaw": swap_xy_pose["corrected_downward_yaw"],
                "formal_safe_candidate": swap_xy_pose["formal_safe_candidate"],
                "debug_only": swap_xy_pose["debug_only"],
            },
            "safe_gate": {
                "policy": "chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial",
                "selected_candidate": selected_name,
                "selected_is_initial": selected_name == "initial",
                "raw_refined_visually_degrades_initial": raw_degrades,
                "recommendation": (
                    "accept_selected_candidate"
                    if selected_name != "initial"
                    else "keep_initial_pose"
                ),
            },
            "pose_convention_policy": {
                "raw_refined_full_debug_only": True,
                "allow_raw_full_pose": bool(args.allow_raw_full_pose),
                "formal_candidates": [
                    name
                    for name, metrics in metrics_by_name.items()
                    if metrics.get("formal_safe_candidate")
                    and not metrics.get("debug_only")
                    and name != "safe_selected"
                ],
                "reason": (
                    "raw refined full pose may use a different Euler/camera "
                    "convention and must not be accepted before matrix-level "
                    "pose convention audit."
                ),
            },
            "candidates": metrics_by_name,
            "total_time_sec": time.perf_counter() - start_total,
        }
        _write_json(output_dir / "summary_metrics.json", summary)
        _write_summary_md(output_dir, summary)
        print(
            json.dumps(
                {
                    "selected_candidate": selected_name,
                    "raw_refined_visually_degrades_initial": raw_degrades,
                    "summary": os.fspath(output_dir / "summary_metrics.json"),
                },
                indent=2,
            )
        )
        return 0

    except Exception:
        tb = traceback.format_exc()
        log["failure_stage"] = stage
        log["traceback"] = tb
        log["total_time_sec"] = time.perf_counter() - start_total
        _write_json(raw_dir / "run_log.json", log)
        print(tb, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
