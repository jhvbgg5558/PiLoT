#!/usr/bin/env python3
"""Run P9.4 DOM+DSM refine validation over the exif_test image set."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/caiwangcun_domdsm.yaml"
DEFAULT_QUERY_DIR = "data_caiwangcun/query/images/exif_test"
DEFAULT_POSE_FILE = "data_caiwangcun/query/poses/exif_test_yawfix.txt"
DEFAULT_OUTPUT_ROOT = (
    "docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4"
)
VALIDATOR = "tools/validate_domdsm_refine_single_0000.py"
SWAP_NAME = "swap_xy_freeze_alt_corrected_downward_yaw"


def _read_pose_lines(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append((stripped.split()[0], stripped))
    return rows


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric(candidate: Dict[str, Any], key: str) -> float:
    value = candidate.get(key)
    return float(value) if value is not None else float("nan")


def _candidate_metrics(summary: Dict[str, Any], name: str) -> Dict[str, Any]:
    return dict(summary.get("candidates", {}).get(name, {}))


def _collect_row(stem: str, image_name: str, output_dir: Path) -> Dict[str, Any]:
    summary_path = output_dir / "summary_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = summary["candidates"]
    initial = candidates["initial"]
    selected_name = summary["safe_gate"]["selected_candidate"]
    selected = candidates[selected_name]
    raw = _candidate_metrics(summary, "raw_refined_full")
    corrected = _candidate_metrics(summary, "corrected_downward_yaw")
    swap = _candidate_metrics(summary, SWAP_NAME)

    initial_chamfer = _metric(initial, "edge_chamfer")
    initial_overlap = _metric(initial, "edge_overlap_ratio")
    selected_chamfer = _metric(selected, "edge_chamfer")
    selected_overlap = _metric(selected, "edge_overlap_ratio")
    chamfer_delta = selected_chamfer - initial_chamfer
    overlap_delta = selected_overlap - initial_overlap
    strict_improves_both = (
        selected_chamfer <= initial_chamfer + 1e-9
        and selected_overlap + 1e-9 >= initial_overlap
    )

    return {
        "stem": stem,
        "image": image_name,
        "output_dir": str(output_dir),
        "summary_metrics": str(summary_path),
        "initial_edge_chamfer": initial_chamfer,
        "initial_edge_overlap_ratio": initial_overlap,
        "selected_candidate": selected_name,
        "selected_edge_chamfer": selected_chamfer,
        "selected_edge_overlap_ratio": selected_overlap,
        "chamfer_delta": chamfer_delta,
        "overlap_delta": overlap_delta,
        "strict_improves_both": strict_improves_both,
        "selected_is_initial_fallback": selected_name == "initial",
        "selected_is_swap_adapter": selected_name == SWAP_NAME,
        "raw_refined_edge_chamfer": _metric(raw, "edge_chamfer"),
        "raw_refined_edge_overlap_ratio": _metric(raw, "edge_overlap_ratio"),
        "corrected_downward_yaw_edge_chamfer": _metric(corrected, "edge_chamfer"),
        "corrected_downward_yaw_edge_overlap_ratio": _metric(
            corrected, "edge_overlap_ratio"
        ),
        "swap_adapter_edge_chamfer": _metric(swap, "edge_chamfer"),
        "swap_adapter_edge_overlap_ratio": _metric(swap, "edge_overlap_ratio"),
        "safe_gate_pass": bool(selected.get("safe_gate_pass")),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "stem",
        "image",
        "initial_edge_chamfer",
        "initial_edge_overlap_ratio",
        "selected_candidate",
        "selected_edge_chamfer",
        "selected_edge_overlap_ratio",
        "chamfer_delta",
        "overlap_delta",
        "strict_improves_both",
        "selected_is_initial_fallback",
        "selected_is_swap_adapter",
        "raw_refined_edge_chamfer",
        "raw_refined_edge_overlap_ratio",
        "corrected_downward_yaw_edge_chamfer",
        "corrected_downward_yaw_edge_overlap_ratio",
        "swap_adapter_edge_chamfer",
        "swap_adapter_edge_overlap_ratio",
        "safe_gate_pass",
        "summary_metrics",
        "output_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_counts(path: Path, rows: List[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = Counter(row["selected_candidate"] for row in rows)
    labels = list(counts.keys()) or ["none"]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.8), 4))
    ax.bar(labels, values, color="#3b82f6")
    ax.set_ylabel("count")
    ax.set_title("Selected candidate counts")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_delta_bar(path: Path, rows: List[Dict[str, Any]], key: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["stem"] for row in rows]
    values = [float(row[key]) for row in rows]
    colors = ["#16a34a" if v <= 0 else "#dc2626" for v in values]
    if key == "overlap_delta":
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in values]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _make_contact_sheet(path: Path, rows: List[Dict[str, Any]]) -> None:
    tiles: List[np.ndarray] = []
    for row in rows:
        overlay_path = Path(row["output_dir"]) / "safe_selected" / "overlay.png"
        image = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile_w = 320
        scale = tile_w / float(image.shape[1])
        tile_h = max(1, int(round(image.shape[0] * scale)))
        tile = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        label = f"{row['stem']}  {row['selected_candidate']}"
        cv2.rectangle(tile, (0, 0), (tile_w, 28), (255, 255, 255), thickness=-1)
        cv2.putText(
            tile,
            label[:48],
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        blank = np.full((120, 320, 3), 255, dtype=np.uint8)
        cv2.putText(
            blank,
            "No selected overlays",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(path), blank)
        return

    cols = min(4, len(tiles))
    max_h = max(tile.shape[0] for tile in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[0] < max_h:
            pad = np.full((max_h - tile.shape[0], tile.shape[1], 3), 255, dtype=np.uint8)
            tile = np.vstack([tile, pad])
        padded.append(tile)
    rows_img = []
    for start in range(0, len(padded), cols):
        chunk = padded[start : start + cols]
        while len(chunk) < cols:
            chunk.append(np.full_like(padded[0], 255))
        rows_img.append(np.hstack(chunk))
    cv2.imwrite(str(path), np.vstack(rows_img))


def _copy_selected_outputs(row: Dict[str, Any], selected_root: Path) -> None:
    src = Path(row["output_dir"]) / "safe_selected"
    dst = selected_root / row["stem"]
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["overlay.png", "rendered_rgb.png", "edge_overlay.png", "checkerboard.png"]:
        candidate = src / name
        if candidate.exists():
            shutil.copy2(candidate, dst / name)


def _batch_summary(rows: List[Dict[str, Any]], failures: List[Dict[str, Any]]) -> Dict[str, Any]:
    chamfer_deltas = [float(row["chamfer_delta"]) for row in rows]
    overlap_deltas = [float(row["overlap_delta"]) for row in rows]
    regressions = [
        row["stem"]
        for row in rows
        if not bool(row["strict_improves_both"])
    ]
    return {
        "num_images_processed": len(rows),
        "num_images_failed": len(failures),
        "strict_improve_count": sum(bool(row["strict_improves_both"]) for row in rows),
        "strict_improve_rate": (
            sum(bool(row["strict_improves_both"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "fallback_to_initial_count": sum(
            bool(row["selected_is_initial_fallback"]) for row in rows
        ),
        "new_swap_adapter_selected_count": sum(
            bool(row["selected_is_swap_adapter"]) for row in rows
        ),
        "selected_candidate_counts": dict(Counter(row["selected_candidate"] for row in rows)),
        "mean_chamfer_delta": mean(chamfer_deltas) if chamfer_deltas else None,
        "median_chamfer_delta": median(chamfer_deltas) if chamfer_deltas else None,
        "mean_overlap_delta": mean(overlap_deltas) if overlap_deltas else None,
        "median_overlap_delta": median(overlap_deltas) if overlap_deltas else None,
        "regressions": regressions,
        "all_selected_refine_outputs_strictly_better_than_initial": (
            len(rows) > 0 and not regressions and not failures
        ),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--query-dir", default=DEFAULT_QUERY_DIR)
    parser.add_argument("--pose-file", default=DEFAULT_POSE_FILE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = (REPO_ROOT / args.output_root).resolve()
    query_dir = REPO_ROOT / args.query_dir
    pose_file = REPO_ROOT / args.pose_file
    output_root.mkdir(parents=True, exist_ok=True)
    pose_tmp_dir = output_root / "_pose_files"
    log_dir = output_root / "_logs"
    pose_tmp_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    pose_rows = _read_pose_lines(pose_file)

    for image_name, pose_line in pose_rows:
        stem = Path(image_name).stem
        query_image = query_dir / image_name
        image_output = output_root / stem
        per_image_pose = pose_tmp_dir / f"{stem}.txt"
        per_image_pose.write_text(pose_line + "\n", encoding="utf-8")

        cmd = [
            args.python,
            VALIDATOR,
            "--config",
            args.config,
            "--query-image",
            str(query_image.relative_to(REPO_ROOT)),
            "--pose-file",
            str(per_image_pose.relative_to(REPO_ROOT)),
            "--output-dir",
            str(image_output.relative_to(REPO_ROOT)),
            "--width",
            str(args.width),
        ]
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (log_dir / f"{stem}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (log_dir / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            failures.append(
                {
                    "stem": stem,
                    "image": image_name,
                    "returncode": proc.returncode,
                    "stdout_log": str(log_dir / f"{stem}.stdout.txt"),
                    "stderr_log": str(log_dir / f"{stem}.stderr.txt"),
                    "output_dir": str(image_output),
                }
            )
            continue
        try:
            row = _collect_row(stem, image_name, image_output)
            rows.append(row)
            _copy_selected_outputs(row, output_root / "selected_overlays")
        except Exception as exc:
            failures.append(
                {
                    "stem": stem,
                    "image": image_name,
                    "returncode": 0,
                    "error": f"aggregate failed: {exc}",
                    "output_dir": str(image_output),
                }
            )

    _write_csv(output_root / "batch_metrics.csv", rows)
    summary = _batch_summary(rows, failures)
    summary.update(
        {
            "config": args.config,
            "query_dir": args.query_dir,
            "pose_file": args.pose_file,
            "output_root": str(output_root),
            "rows": rows,
        }
    )
    _write_json(output_root / "batch_summary.json", summary)
    _plot_counts(output_root / "selected_candidate_counts.png", rows)
    _plot_delta_bar(
        output_root / "chamfer_delta_bar.png",
        rows,
        "chamfer_delta",
        "Selected minus initial chamfer",
    )
    _plot_delta_bar(
        output_root / "overlap_delta_bar.png",
        rows,
        "overlap_delta",
        "Selected minus initial overlap",
    )
    _make_contact_sheet(output_root / "contact_sheet_selected_overlays.png", rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
