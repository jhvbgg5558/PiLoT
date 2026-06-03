#!/usr/bin/env python3
"""Generate a synthetic DOM/DSM flight sequence for PiLoT experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np
import rasterio
import yaml
from pyproj import Transformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixloc.utils.dom_dsm.dom_dsm_render import DOMDSMRenderer
from pixloc.utils.dom_dsm.pose_adapter import (
    domxy_to_wgs84,
    make_downward_euler_from_yaw,
    normalize_angle_deg,
)


DEFAULT_CONFIG = "configs/caiwangcun_domdsm_16x9.yaml"
DEFAULT_OUTPUT_DIR = "docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11"
TRAJECTORY_CHOICES = ("straight_line", "lawnmower", "s_curve", "circle")
LANDCOVER_MODES = ("valid_dsm", "mixed_road_vegetation")


def _safe_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_jsonable(value.tolist())
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
    path.write_text(
        json.dumps(_safe_jsonable(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> Dict[str, Path]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    dirs = {
        "images": output_dir / "images",
        "depth": output_dir / "depth",
        "masks": output_dir / "masks",
        "poses": output_dir / "poses",
        "previews": output_dir / "previews",
        "debug": output_dir / "debug_render",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _scale_camera(config: Dict[str, Any], width: int, height: Optional[int]) -> Tuple[List[float], Dict[str, Any]]:
    cam = config["default_confs"]["cam_query"]
    src_w = float(cam["width"])
    src_h = float(cam["height"])
    if height is None:
        height = int(round(float(width) * src_h / src_w))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output size: {width}x{height}")

    fx, fy, cx, cy = [float(x) for x in cam["params"]]
    sx = float(width) / src_w
    sy = float(height) / src_h
    render_camera = [
        float(width),
        float(height),
        cx * sx,
        cy * sy,
        fx * sx,
        fy * sy,
    ]
    camera_meta = {
        "source_width": int(src_w),
        "source_height": int(src_h),
        "source_params_fx_fy_cx_cy": [fx, fy, cx, cy],
        "width": int(width),
        "height": int(height),
        "scale_x": sx,
        "scale_y": sy,
        "render_camera_w_h_cx_cy_fx_fy": render_camera,
    }
    return render_camera, camera_meta


def _raster_intersection_bounds(dom: rasterio.io.DatasetReader, dsm: rasterio.io.DatasetReader) -> Tuple[float, float, float, float]:
    left = max(float(dom.bounds.left), float(dsm.bounds.left))
    right = min(float(dom.bounds.right), float(dsm.bounds.right))
    bottom = max(float(dom.bounds.bottom), float(dsm.bounds.bottom))
    top = min(float(dom.bounds.top), float(dsm.bounds.top))
    if left >= right or bottom >= top:
        raise ValueError(f"DOM/DSM bounds do not overlap: DOM={dom.bounds}, DSM={dsm.bounds}")
    return left, bottom, right, top


def _bilinear_sample_raster(
    ds: rasterio.io.DatasetReader,
    band: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    inv = ~ds.transform
    cols, rows = inv * (xs, ys)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    out = np.full(rows.shape, np.nan, dtype=np.float64)

    r0 = np.floor(rows).astype(np.int64)
    c0 = np.floor(cols).astype(np.int64)
    r1 = r0 + 1
    c1 = c0 + 1
    valid = (r0 >= 0) & (r1 < band.shape[0]) & (c0 >= 0) & (c1 < band.shape[1])
    if not np.any(valid):
        return out

    wr = rows[valid] - r0[valid]
    wc = cols[valid] - c0[valid]
    v00 = band[r0[valid], c0[valid]].astype(np.float64)
    v01 = band[r0[valid], c1[valid]].astype(np.float64)
    v10 = band[r1[valid], c0[valid]].astype(np.float64)
    v11 = band[r1[valid], c1[valid]].astype(np.float64)
    vals = (
        (1.0 - wr) * (1.0 - wc) * v00
        + (1.0 - wr) * wc * v01
        + wr * (1.0 - wc) * v10
        + wr * wc * v11
    )
    good = np.isfinite(vals)
    if nodata is not None:
        good &= np.isfinite(v00) & np.isfinite(v01) & np.isfinite(v10) & np.isfinite(v11)
        good &= (v00 != float(nodata)) & (v01 != float(nodata)) & (v10 != float(nodata)) & (v11 != float(nodata))
    valid_ids = np.flatnonzero(valid)
    out[valid_ids[good]] = vals[good]
    return out


def _heading_unit_xy(heading_deg: float) -> np.ndarray:
    heading_rad = math.radians(float(heading_deg))
    return np.asarray([math.sin(heading_rad), math.cos(heading_rad)], dtype=np.float64)


def _trajectory_xy_for_center(
    center_xy: Sequence[float],
    num_frames: int,
    step_m: float,
    heading_deg: float,
) -> np.ndarray:
    unit = _heading_unit_xy(heading_deg)
    offsets = (np.arange(num_frames, dtype=np.float64) - (num_frames - 1) / 2.0) * float(step_m)
    return np.asarray(center_xy, dtype=np.float64)[None, :] + offsets[:, None] * unit[None, :]


def _candidate_centers_for_line(
    bounds: Tuple[float, float, float, float],
    num_frames: int,
    step_m: float,
    heading_deg: float,
    grid_size: int = 9,
) -> List[np.ndarray]:
    left, bottom, right, top = bounds
    half_vec = abs((num_frames - 1) * float(step_m) / 2.0) * np.abs(_heading_unit_xy(heading_deg))
    x_min = left + half_vec[0]
    x_max = right - half_vec[0]
    y_min = bottom + half_vec[1]
    y_max = top - half_vec[1]
    if x_min > x_max or y_min > y_max:
        raise ValueError(
            "Requested straight_line trajectory does not fit in DOM/DSM bounds: "
            f"length={(num_frames - 1) * float(step_m):.3f}m bounds={bounds}"
        )

    grid_size = max(int(grid_size), 3)
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    centers = [np.asarray([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0], dtype=np.float64)]
    for y in ys:
        for x in xs:
            candidate = np.asarray([x, y], dtype=np.float64)
            if not any(np.allclose(candidate, c) for c in centers):
                centers.append(candidate)
    return centers


def _read_rgb_window(
    dom: rasterio.io.DatasetReader,
    x: float,
    y: float,
    patch_m: float,
    out_size: int = 96,
) -> Optional[np.ndarray]:
    half = float(patch_m) / 2.0
    window = rasterio.windows.from_bounds(
        x - half, y - half, x + half, y + half, transform=dom.transform
    )
    try:
        arr = dom.read(
            indexes=list(range(1, min(3, dom.count) + 1)),
            window=window,
            out_shape=(min(3, dom.count), out_size, out_size),
            boundless=False,
        )
    except Exception:
        return None
    if arr.shape[0] < 3 or arr.size == 0:
        return None
    rgb = np.moveaxis(arr[:3], 0, -1).astype(np.float32)
    finite = np.isfinite(rgb)
    if not np.any(finite):
        return None
    hi = float(np.nanpercentile(rgb[finite], 99.0))
    if hi <= 0:
        return None
    if hi > 255.0:
        rgb = rgb * (255.0 / hi)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _read_dsm_window(
    dsm: rasterio.io.DatasetReader,
    x: float,
    y: float,
    patch_m: float,
    nodata: Optional[float],
    out_size: int = 64,
) -> Optional[np.ndarray]:
    half = float(patch_m) / 2.0
    window = rasterio.windows.from_bounds(
        x - half, y - half, x + half, y + half, transform=dsm.transform
    )
    try:
        arr = dsm.read(1, window=window, out_shape=(out_size, out_size), boundless=False).astype(np.float32)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if nodata is not None:
        arr[arr == float(nodata)] = np.nan
    return arr


def _landcover_patch_metrics(
    dom: rasterio.io.DatasetReader,
    dsm: rasterio.io.DatasetReader,
    x: float,
    y: float,
    patch_m: float,
    nodata: Optional[float],
) -> Dict[str, float]:
    rgb = _read_rgb_window(dom, x, y, patch_m)
    z = _read_dsm_window(dsm, x, y, patch_m, nodata)
    if rgb is None or z is None:
        return {
            "vegetation_ratio": 0.0,
            "road_like_ratio": 0.0,
            "edge_density": 0.0,
            "height_variation": float("inf"),
            "building_density": 1.0,
            "landcover_score": float("-inf"),
        }

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    vegetation = (h >= 35) & (h <= 95) & (s >= 35) & (v >= 35)
    road_like = (s <= 45) & (v >= 55) & (v <= 225) & (~vegetation)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160) > 0

    finite_z = z[np.isfinite(z)]
    if finite_z.size:
        p10, p50, p90 = np.percentile(finite_z, [10, 50, 90])
        height_variation = float(p90 - p10)
        building_density = float(np.mean(finite_z > p50 + 4.0))
    else:
        height_variation = float("inf")
        building_density = 1.0

    vegetation_ratio = float(np.mean(vegetation))
    road_like_ratio = float(np.mean(road_like))
    edge_density = float(np.mean(edges))
    finite_height = height_variation if np.isfinite(height_variation) else 100.0
    score = (
        1.8 * vegetation_ratio
        + 1.2 * road_like_ratio
        + 0.4 * edge_density
        - 1.4 * building_density
        - 0.025 * finite_height
    )
    return {
        "vegetation_ratio": vegetation_ratio,
        "road_like_ratio": road_like_ratio,
        "edge_density": edge_density,
        "height_variation": height_variation,
        "building_density": building_density,
        "landcover_score": float(score),
    }


def _mean_landcover_metrics(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = ["vegetation_ratio", "road_like_ratio", "edge_density", "height_variation", "building_density", "landcover_score"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.asarray([float(item[key]) for item in items if np.isfinite(float(item[key]))], dtype=np.float64)
        out[key] = float(vals.mean()) if vals.size else (float("-inf") if key == "landcover_score" else float("nan"))
    return out


def _score_trajectory_landcover(
    dom: rasterio.io.DatasetReader,
    dsm: rasterio.io.DatasetReader,
    xy: np.ndarray,
    patch_m: float,
    nodata: Optional[float],
    sample_stride: int,
) -> Dict[str, Any]:
    stride = max(int(sample_stride), 1)
    sample_ids = list(range(0, len(xy), stride))
    if sample_ids[-1] != len(xy) - 1:
        sample_ids.append(len(xy) - 1)
    samples = [
        _landcover_patch_metrics(dom, dsm, float(xy[i, 0]), float(xy[i, 1]), patch_m, nodata)
        for i in sample_ids
    ]
    mean_metrics = _mean_landcover_metrics(samples)
    return {
        **mean_metrics,
        "sample_frame_indices": sample_ids,
    }


def _frame_landcover_metrics(
    dom: rasterio.io.DatasetReader,
    dsm: rasterio.io.DatasetReader,
    trajectory: List[Dict[str, Any]],
    patch_m: float,
    nodata: Optional[float],
) -> List[Dict[str, float]]:
    return [
        _landcover_patch_metrics(dom, dsm, float(frame["x"]), float(frame["y"]), patch_m, nodata)
        for frame in trajectory
    ]


def _build_straight_line_trajectory(
    dsm: rasterio.io.DatasetReader,
    dsm_array: np.ndarray,
    bounds: Tuple[float, float, float, float],
    nodata: Optional[float],
    num_frames: int,
    step_m: float,
    agl_m: float,
    heading_deg: float,
    dom: Optional[rasterio.io.DatasetReader] = None,
    landcover_mode: str = "valid_dsm",
    landcover_patch_m: float = 80.0,
    landcover_grid_size: int = 15,
    landcover_sample_stride: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    best: Optional[Tuple[int, float, np.ndarray, np.ndarray, Dict[str, Any]]] = None
    candidate_summaries: List[Dict[str, Any]] = []
    for center in _candidate_centers_for_line(bounds, num_frames, step_m, heading_deg, landcover_grid_size):
        xy = _trajectory_xy_for_center(center, num_frames, step_m, heading_deg)
        terrain = _bilinear_sample_raster(dsm, dsm_array, xy[:, 0], xy[:, 1], nodata)
        finite_count = int(np.count_nonzero(np.isfinite(terrain)))
        metrics: Dict[str, Any] = {}
        if finite_count == num_frames and landcover_mode == "mixed_road_vegetation" and dom is not None:
            metrics = _score_trajectory_landcover(
                dom, dsm, xy, landcover_patch_m, nodata, landcover_sample_stride
            )
            score = float(metrics["landcover_score"])
        else:
            score = float(np.nanmedian(terrain)) if finite_count else float("-inf")
        payload = (finite_count, score, xy, terrain, metrics)
        if best is None or payload[:2] > best[:2]:
            best = payload
        candidate_summaries.append(
            {
                "center_xy": [float(center[0]), float(center[1])],
                "finite_count": finite_count,
                "score": score,
                **metrics,
            }
        )
        if finite_count == num_frames and landcover_mode == "valid_dsm":
            best = payload
            break

    if best is None or best[0] != num_frames:
        raise ValueError(
            f"Could not find a straight_line center with valid DSM for all frames "
            f"({0 if best is None else best[0]}/{num_frames} valid)."
        )

    _finite_count, _score, xy, terrain, selected_metrics = best
    to_wgs84 = Transformer.from_crs(dsm.crs, "EPSG:4326", always_xy=True)
    trajectory: List[Dict[str, Any]] = []
    for idx, ((x, y), z_ground) in enumerate(zip(xy, terrain)):
        alt = float(z_ground) + float(agl_m)
        lon, lat, _ = domxy_to_wgs84([float(x), float(y), alt], to_wgs84)
        yaw = normalize_angle_deg(heading_deg)
        trajectory.append(
            {
                "idx": idx,
                "name": f"{idx:06d}.jpg",
                "x": float(x),
                "y": float(y),
                "ground_z": float(z_ground),
                "lon": float(lon),
                "lat": float(lat),
                "alt": float(alt),
                "yaw": float(yaw),
                "euler": make_downward_euler_from_yaw(yaw),
            }
        )
    meta = {
        "center_xy": [float(x) for x in np.mean(xy, axis=0)],
        "start_xy": [float(x) for x in xy[0]],
        "end_xy": [float(x) for x in xy[-1]],
        "length_m": float((num_frames - 1) * float(step_m)),
        "selection_score": float(_score),
        "landcover_mode": landcover_mode,
        "selected_landcover_metrics": selected_metrics,
        "top_landcover_candidates": sorted(
            candidate_summaries,
            key=lambda item: float(item.get("score", float("-inf"))),
            reverse=True,
        )[:10],
    }
    return trajectory, meta


def _pose_line(frame: Dict[str, Any]) -> str:
    euler = frame["euler"]
    return (
        f"{frame['name']} {frame['lon']:.10f} {frame['lat']:.10f} {frame['alt']:.3f} "
        f"{float(euler[1]):.6f} {float(euler[0]):.6f} {float(euler[2]):.6f}"
    )


def _noisy_pose_line(
    frame: Dict[str, Any],
    rng: np.random.Generator,
    from_raster: Transformer,
    init_noise_xy_std: float,
    init_noise_alt_std: float,
    init_noise_yaw_std: float,
) -> Tuple[str, Dict[str, float]]:
    dx = float(rng.normal(0.0, init_noise_xy_std))
    dy = float(rng.normal(0.0, init_noise_xy_std))
    dz = float(rng.normal(0.0, init_noise_alt_std))
    dyaw = float(rng.normal(0.0, init_noise_yaw_std))
    alt = float(frame["alt"]) + dz
    lon, lat, _ = domxy_to_wgs84([float(frame["x"]) + dx, float(frame["y"]) + dy, alt], from_raster)
    yaw = normalize_angle_deg(float(frame["yaw"]) + dyaw)
    euler = make_downward_euler_from_yaw(yaw)
    line = (
        f"{frame['name']} {lon:.10f} {lat:.10f} {alt:.3f} "
        f"{float(euler[1]):.6f} {float(euler[0]):.6f} {float(euler[2]):.6f}"
    )
    return line, {"dx_m": dx, "dy_m": dy, "dz_m": dz, "dyaw_deg": dyaw}


def _odom_edge_lines(
    trajectory: List[Dict[str, Any]],
    rng: np.random.Generator,
    odom_noise_xy_std: float,
    odom_noise_yaw_std: float,
) -> List[str]:
    lines: List[str] = []
    for prev, curr in zip(trajectory[:-1], trajectory[1:]):
        dx = float(curr["x"] - prev["x"] + rng.normal(0.0, odom_noise_xy_std))
        dy = float(curr["y"] - prev["y"] + rng.normal(0.0, odom_noise_xy_std))
        dz = float(curr["alt"] - prev["alt"])
        dyaw = normalize_angle_deg(float(curr["yaw"] - prev["yaw"]) + rng.normal(0.0, odom_noise_yaw_std))
        dist = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        lines.append(
            f"{prev['name']} {curr['name']} {dx:.6f} {dy:.6f} {dz:.6f} {dyaw:.6f} {dist:.6f}"
        )
    return lines


def _noise_stats(noise_rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not noise_rows:
        return out
    for key in ["dx_m", "dy_m", "dz_m", "dyaw_deg"]:
        vals = np.asarray([row[key] for row in noise_rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p95_abs": float(np.percentile(np.abs(vals), 95)),
        }
    xy = np.asarray([[row["dx_m"], row["dy_m"]] for row in noise_rows], dtype=np.float64)
    norm = np.linalg.norm(xy, axis=1)
    out["xy_norm_m"] = {
        "mean": float(norm.mean()),
        "std": float(norm.std()),
        "p95_abs": float(np.percentile(norm, 95)),
    }
    return out


def _save_trajectory_preview(path: Path, trajectory: List[Dict[str, Any]], bounds: Tuple[float, float, float, float]) -> None:
    xs = [frame["x"] for frame in trajectory]
    ys = [frame["y"] for frame in trajectory]
    left, bottom, right, top = bounds
    plt.figure(figsize=(8, 8))
    plt.plot([left, right, right, left, left], [bottom, bottom, top, top, bottom], "k--", linewidth=1, label="DOM/DSM overlap")
    plt.plot(xs, ys, "-o", markersize=2, linewidth=1.5, label="GT trajectory")
    plt.scatter([xs[0]], [ys[0]], c="green", s=40, label="start")
    plt.scatter([xs[-1]], [ys[-1]], c="red", s=40, label="end")
    plt.axis("equal")
    plt.xlabel("Raster X / Easting (m)")
    plt.ylabel("Raster Y / Northing (m)")
    plt.legend(loc="best")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _save_sample_frames(path: Path, image_paths: List[Path]) -> None:
    if not image_paths:
        return
    indices = sorted(set(np.linspace(0, len(image_paths) - 1, min(8, len(image_paths)), dtype=int).tolist()))
    imgs = []
    for idx in indices:
        bgr = cv2.imread(os.fspath(image_paths[idx]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        thumb_w = 240
        thumb_h = int(round(rgb.shape[0] * thumb_w / max(rgb.shape[1], 1)))
        imgs.append((idx, cv2.resize(rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)))
    if not imgs:
        return
    cols = min(4, len(imgs))
    rows = int(math.ceil(len(imgs) / cols))
    plt.figure(figsize=(cols * 3.0, rows * 2.4))
    for i, (idx, rgb) in enumerate(imgs, start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(rgb)
        ax.set_title(f"{idx:06d}")
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trajectory", choices=TRAJECTORY_CHOICES, default="straight_line")
    parser.add_argument("--landcover-mode", choices=LANDCOVER_MODES, default="valid_dsm")
    parser.add_argument("--landcover-patch-m", type=float, default=80.0)
    parser.add_argument("--landcover-grid-size", type=int, default=15)
    parser.add_argument("--landcover-sample-stride", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--step-m", type=float, default=1.0)
    parser.add_argument("--agl-m", type=float, default=80.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--heading-deg", type=float, default=29.2)
    parser.add_argument("--init-noise-xy-std", type=float, default=3.0)
    parser.add_argument("--init-noise-alt-std", type=float, default=1.0)
    parser.add_argument("--init-noise-yaw-std", type=float, default=2.0)
    parser.add_argument("--odom-noise-xy-std", type=float, default=0.0)
    parser.add_argument("--odom-noise-yaw-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.trajectory != "straight_line":
        raise NotImplementedError(f"Only --trajectory straight_line is implemented in P11 v1, got {args.trajectory}")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.step_m <= 0:
        raise ValueError("--step-m must be positive")
    if args.agl_m <= 0:
        raise ValueError("--agl-m must be positive")
    if args.landcover_patch_m <= 0:
        raise ValueError("--landcover-patch-m must be positive")

    config_path = _resolve_repo_path(args.config)
    output_dir = _resolve_repo_path(args.output_dir)
    dirs = _prepare_output_dir(output_dir, args.overwrite)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    render_config = deepcopy(config["render_config"])
    render_camera, camera_meta = _scale_camera(config, args.width, args.height)
    render_config["render_camera"] = render_camera
    render_config.setdefault("dom_dsm", {})
    render_config["dom_dsm"]["debug_dir"] = os.fspath(dirs["debug"])
    render_config["dom_dsm"]["debug_every"] = 0
    if args.min_valid_depth_ratio is not None:
        render_config["dom_dsm"]["min_valid_depth_ratio"] = float(args.min_valid_depth_ratio)

    dom_path = Path(render_config["dom_dsm"]["dom_path"])
    dsm_path = Path(render_config["dom_dsm"]["dsm_path"])
    nodata = render_config["dom_dsm"].get("nodata")
    min_valid_ratio = float(render_config["dom_dsm"].get("min_valid_depth_ratio", 0.05))

    rng = np.random.default_rng(args.seed)
    renderer = DOMDSMRenderer(render_config)

    with rasterio.open(dom_path) as dom, rasterio.open(dsm_path) as dsm:
        bounds = _raster_intersection_bounds(dom, dsm)
        dsm_array = dsm.read(1)
        trajectory, trajectory_meta = _build_straight_line_trajectory(
            dsm,
            dsm_array,
            bounds,
            nodata,
            args.num_frames,
            args.step_m,
            args.agl_m,
            args.heading_deg,
            dom=dom,
            landcover_mode=args.landcover_mode,
            landcover_patch_m=args.landcover_patch_m,
            landcover_grid_size=args.landcover_grid_size,
            landcover_sample_stride=args.landcover_sample_stride,
        )
        frame_landcover = _frame_landcover_metrics(
            dom,
            dsm,
            trajectory,
            args.landcover_patch_m,
            nodata,
        )
        from_raster = Transformer.from_crs(dsm.crs, "EPSG:4326", always_xy=True)
        raster_meta = {
            "crs": str(dsm.crs),
            "dom_path": dom_path.as_posix(),
            "dsm_path": dsm_path.as_posix(),
            "dom_bounds": [float(dom.bounds.left), float(dom.bounds.bottom), float(dom.bounds.right), float(dom.bounds.top)],
            "dsm_bounds": [float(dsm.bounds.left), float(dsm.bounds.bottom), float(dsm.bounds.right), float(dsm.bounds.top)],
            "intersection_bounds": [float(x) for x in bounds],
            "dom_res": [float(x) for x in dom.res],
            "dsm_res": [float(x) for x in dsm.res],
            "nodata": nodata,
        }

    gt_lines: List[str] = []
    noisy_lines: List[str] = []
    noise_rows: List[Dict[str, float]] = []
    frame_records: List[Dict[str, Any]] = []
    image_paths: List[Path] = []
    warnings: List[Dict[str, Any]] = []

    for frame in trajectory:
        idx = int(frame["idx"])
        landcover_metrics = frame_landcover[idx] if idx < len(frame_landcover) else {}
        stem = f"{idx:06d}"
        trans = [float(frame["lon"]), float(frame["lat"]), float(frame["alt"])]
        euler = [float(x) for x in frame["euler"]]
        color, depth = renderer.render(trans, euler)
        metadata = deepcopy(getattr(renderer, "last_render_metadata", {}))
        valid = np.isfinite(depth) & (depth > 0)
        valid_ratio = float(np.count_nonzero(valid) / depth.size)

        image_path = dirs["images"] / f"{stem}.jpg"
        mask_path = dirs["masks"] / f"{stem}.png"
        depth_path = dirs["depth"] / f"{stem}.npy"
        cv2.imwrite(
            os.fspath(image_path),
            cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        cv2.imwrite(os.fspath(mask_path), (valid.astype(np.uint8) * 255))
        if args.save_depth:
            np.save(depth_path, depth.astype(np.float32))
        image_paths.append(image_path)

        gt_lines.append(_pose_line(frame))
        noisy_line, noise = _noisy_pose_line(
            frame,
            rng,
            from_raster,
            args.init_noise_xy_std,
            args.init_noise_alt_std,
            args.init_noise_yaw_std,
        )
        noisy_lines.append(noisy_line)
        noise_rows.append(noise)

        fallback_reason = metadata.get("fallback_reason")
        if fallback_reason:
            warnings.append({"frame": stem, "type": "renderer_fallback", "reason": fallback_reason})
        if valid_ratio < min_valid_ratio:
            warnings.append(
                {
                    "frame": stem,
                    "type": "low_valid_depth_ratio",
                    "valid_depth_ratio": valid_ratio,
                    "threshold": min_valid_ratio,
                }
            )
        frame_records.append(
            {
                "frame": stem,
                "image": image_path.relative_to(output_dir).as_posix(),
                "mask": mask_path.relative_to(output_dir).as_posix(),
                "depth": depth_path.relative_to(output_dir).as_posix() if args.save_depth else None,
                "translation_lon_lat_alt": trans,
                "euler_pitch_roll_yaw": euler,
                "raster_xy_ground_z": [frame["x"], frame["y"], frame["ground_z"]],
                "valid_depth_ratio": valid_ratio,
                "landcover_metrics": landcover_metrics,
                "renderer_metadata": metadata,
            }
        )
        if (idx + 1) % 25 == 0 or idx == args.num_frames - 1:
            print(f"Rendered {idx + 1}/{args.num_frames} frames; valid_ratio={valid_ratio:.4f}")

    odom_lines = _odom_edge_lines(
        trajectory,
        rng,
        args.odom_noise_xy_std,
        args.odom_noise_yaw_std,
    )

    gt_pose_path = dirs["poses"] / "gt_poses.txt"
    noisy_pose_path = dirs["poses"] / "init_noisy_poses.txt"
    odom_path = dirs["poses"] / "odom_edges.txt"
    gt_pose_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    noisy_pose_path.write_text("\n".join(noisy_lines) + "\n", encoding="utf-8")
    odom_path.write_text("\n".join(odom_lines) + ("\n" if odom_lines else ""), encoding="utf-8")

    if args.save_preview:
        _save_trajectory_preview(dirs["previews"] / "trajectory_topdown.png", trajectory, tuple(raster_meta["intersection_bounds"]))
        _save_sample_frames(dirs["previews"] / "sample_frames.png", image_paths)
    else:
        _save_trajectory_preview(dirs["previews"] / "trajectory_topdown.png", trajectory, tuple(raster_meta["intersection_bounds"]))

    valid_ratios = np.asarray([row["valid_depth_ratio"] for row in frame_records], dtype=np.float64)
    output_files = {
        "images_dir": (dirs["images"]).relative_to(output_dir).as_posix(),
        "depth_dir": (dirs["depth"]).relative_to(output_dir).as_posix() if args.save_depth else None,
        "masks_dir": (dirs["masks"]).relative_to(output_dir).as_posix(),
        "gt_poses": gt_pose_path.relative_to(output_dir).as_posix(),
        "init_noisy_poses": noisy_pose_path.relative_to(output_dir).as_posix(),
        "odom_edges": odom_path.relative_to(output_dir).as_posix(),
        "trajectory_topdown": (dirs["previews"] / "trajectory_topdown.png").relative_to(output_dir).as_posix(),
        "sample_frames": (dirs["previews"] / "sample_frames.png").relative_to(output_dir).as_posix() if args.save_preview else None,
    }

    metadata = {
        "task": "P11 - Synthetic flight sequence generation for DOMDSM PiLoT",
        "config": args.config,
        "trajectory_type": args.trajectory,
        "landcover_mode": args.landcover_mode,
        "num_frames": args.num_frames,
        "step_m": args.step_m,
        "agl_m": args.agl_m,
        "heading_deg": args.heading_deg,
        "seed": args.seed,
        "noise": {
            "init_noise_xy_std": args.init_noise_xy_std,
            "init_noise_alt_std": args.init_noise_alt_std,
            "init_noise_yaw_std": args.init_noise_yaw_std,
            "odom_noise_xy_std": args.odom_noise_xy_std,
            "odom_noise_yaw_std": args.odom_noise_yaw_std,
        },
        "camera": camera_meta,
        "raster": raster_meta,
        "trajectory": trajectory_meta,
        "landcover": {
            "mode": args.landcover_mode,
            "patch_m": args.landcover_patch_m,
            "grid_size": args.landcover_grid_size,
            "sample_stride": args.landcover_sample_stride,
        },
        "renderer": {
            "type": render_config.get("type"),
            "backend": render_config.get("dom_dsm", {}).get("render_backend"),
            "gpu_renderer": render_config.get("dom_dsm", {}).get("gpu_renderer"),
            "min_valid_depth_ratio": min_valid_ratio,
        },
    }
    landcover_summary = _mean_landcover_metrics(frame_landcover)
    summary = {
        "num_frames": args.num_frames,
        "valid_frame_count": int(np.count_nonzero(valid_ratios >= min_valid_ratio)),
        "mean_valid_depth_ratio": float(valid_ratios.mean()) if valid_ratios.size else None,
        "min_valid_depth_ratio": float(valid_ratios.min()) if valid_ratios.size else None,
        "warnings": warnings,
        "landcover_summary": landcover_summary,
        "init_noise_stats": _noise_stats(noise_rows),
        "output_dir": output_dir.as_posix(),
        "output_files": output_files,
        "frame_records": frame_records,
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "summary.json", summary)

    print(json.dumps(_safe_jsonable({k: summary[k] for k in ["num_frames", "valid_frame_count", "mean_valid_depth_ratio", "min_valid_depth_ratio"]}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
