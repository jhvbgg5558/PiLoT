"""Pose and CRS helpers for DOM/DSM refinement experiments."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import rasterio
from pyproj import Transformer


def _as_trans(trans: Sequence[float]) -> Tuple[float, float, float]:
    if len(trans) != 3:
        raise ValueError(f"Expected [lon, lat, alt], got {trans}")
    return float(trans[0]), float(trans[1]), float(trans[2])


def _domdsm_raster_path(config: Dict[str, Any]) -> Path:
    """Return the configured DOM/DSM raster path used for CRS lookup."""
    render_config = config.get("render_config", {})
    domdsm = render_config.get("dom_dsm", {})
    path = (
        domdsm.get("dom_path")
        or domdsm.get("dsm_path")
        or render_config.get("dom_path")
        or render_config.get("dsm_path")
        or render_config.get("ortho_path")
    )
    if not path:
        raise KeyError("No DOM/DSM raster path found in render_config")
    return Path(path)


def get_domdsm_transformers(config: Dict[str, Any]) -> Tuple[Transformer, Transformer, str]:
    """Build WGS84 <-> DOM/DSM CRS transformers from the configured raster.

    Args:
        config: PiLoT YAML config dictionary containing render_config.dom_dsm.

    Returns:
        (to_raster, from_raster, raster_crs_string), where to_raster converts
        EPSG:4326 lon/lat into DOM/DSM projected x/y meters and from_raster
        converts back to EPSG:4326.
    """
    raster_path = _domdsm_raster_path(config)
    with rasterio.open(raster_path) as ds:
        raster_crs = ds.crs
    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")
    to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    from_raster = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    return to_raster, from_raster, str(raster_crs)


def wgs84_to_domxy(trans: Sequence[float], to_raster: Transformer) -> List[float]:
    """Convert [lon, lat, alt] to DOM/DSM projected [x, y, alt] meters."""
    lon, lat, alt = _as_trans(trans)
    x, y = to_raster.transform(lon, lat)
    return [float(x), float(y), alt]


def domxy_to_wgs84(xya: Sequence[float], from_raster: Transformer) -> List[float]:
    """Convert DOM/DSM projected [x, y, alt] meters to [lon, lat, alt]."""
    if len(xya) != 3:
        raise ValueError(f"Expected [x, y, alt], got {xya}")
    lon, lat = from_raster.transform(float(xya[0]), float(xya[1]))
    return [float(lon), float(lat), float(xya[2])]


def apply_enu_offset(
    trans: Sequence[float],
    east_m: float,
    north_m: float,
    alt_m: float,
    to_raster: Transformer,
    from_raster: Transformer,
) -> List[float]:
    """Apply meter offsets in the DOM/DSM projected CRS and return WGS84 pose translation."""
    x, y, alt = wgs84_to_domxy(trans, to_raster)
    return domxy_to_wgs84([x + float(east_m), y + float(north_m), alt + float(alt_m)], from_raster)


def compute_enu_delta_m(
    base_trans: Sequence[float],
    target_trans: Sequence[float],
    to_raster: Transformer,
) -> List[float]:
    """Compute target-base translation delta in DOM/DSM east/north/alt meters."""
    bx, by, balt = wgs84_to_domxy(base_trans, to_raster)
    tx, ty, talt = wgs84_to_domxy(target_trans, to_raster)
    return [float(tx - bx), float(ty - by), float(talt - balt)]


def normalize_domdsm_euler(euler: Sequence[float]) -> List[float]:
    """Return a DOMDSMRenderer euler triplet in [pitch, roll, yaw] order.

    This is intentionally a narrow, testable conversion entrypoint. The current
    yawfix convention uses [0.0, 180.0, yaw]. Equivalent refined downward-form
    handling should be added explicitly instead of inferred here.
    """
    if len(euler) != 3:
        raise ValueError(f"Expected [pitch, roll, yaw], got {euler}")
    return [float(euler[0]), float(euler[1]), normalize_angle_deg(float(euler[2]))]


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle in degrees into the [-180, 180) interval."""
    return float(((float(angle) + 180.0) % 360.0) - 180.0)


def normalize_yaw(yaw: float) -> float:
    """Normalize a yaw angle in degrees into the DOM/DSM canonical range."""
    return normalize_angle_deg(yaw)


def refined_yaw_to_downward_yaw(refined_yaw: float) -> float:
    """Convert PiLoT raw refined yaw into DOMDSMRenderer downward yaw.

    Current DOM/DSM experiments use a downward-looking renderer convention
    represented as ``[pitch, roll, yaw] = [0, 180, yaw]``. PiLoT raw refined
    Euler output can describe an equivalent camera orientation on a different
    Euler branch, where yaw differs by 180 degrees while pitch/roll sit near the
    opposite branch. This adapter applies the working assumption
    ``downward_yaw = normalize(refined_yaw + 180)``.

    This is not proof that the raw full pose is safe. The assumption must be
    checked by ``tools/audit_pilot_domdsm_pose_convention.py`` with matrix-level
    relative rotations before using corrected poses beyond diagnostic gating.
    """
    return normalize_angle_deg(float(refined_yaw) + 180.0)


def make_downward_euler_from_yaw(yaw: float) -> List[float]:
    """Create the DOMDSMRenderer downward-looking Euler triplet.

    Returns ``[pitch, roll, yaw] = [0, 180, normalize(yaw)]``, matching the
    DOM+DSM downward-camera convention used by the current renderer. The yaw
    value may come from a pose file or from ``refined_yaw_to_downward_yaw()``;
    when it comes from raw PiLoT refinement, the 180-degree equivalence is an
    adapter hypothesis that should be validated by the P9.2 matrix audit.
    """
    return [0.0, 180.0, normalize_angle_deg(yaw)]


def make_domdsm_downward_euler(yaw: float) -> List[float]:
    """Create the DOMDSMRenderer downward-looking euler [pitch, roll, yaw]."""
    return make_downward_euler_from_yaw(yaw)


def make_safe_domdsm_pose_from_refined(
    initial_trans: Sequence[float],
    initial_euler: Sequence[float],
    refined_trans: Sequence[float],
    refined_euler: Sequence[float],
    mode: str,
    to_raster: Transformer = None,
    from_raster: Transformer = None,
) -> Dict[str, Any]:
    """Build a DOM/DSM renderer pose from PiLoT refined output.

    This is the single adapter entrypoint for DOM/DSM validation scripts. It
    deliberately separates debug-only raw PiLoT output from candidate poses that
    can be considered by a visual safe gate.

    Supported modes:
        ``initial``:
            Return the initial translation and Euler unchanged.
        ``raw_debug_only``:
            Return raw refined translation and Euler unchanged, with
            ``debug_only=True`` and ``formal_safe_candidate=False``. This mode
            is for visualization and convention audits only; callers must not
            write it as a safe/final output unless an explicit debug override is
            supplied.
        ``corrected_downward_yaw_freeze_alt_pitch_roll``:
            Use refined lon/lat, initial altitude, canonical downward
            pitch/roll ``[0, 180]``, and
            ``yaw = normalize(refined_yaw + 180)``.
        ``swap_xy_freeze_alt_corrected_downward_yaw``:
            Compute raw refined ENU delta from initial to refined in the
            DOM/DSM projected CRS, swap east/north as
            ``[east=raw_north, north=raw_east]``, freeze altitude to the
            initial pose, and use corrected downward yaw.

    The corrected-yaw mode encodes the current hypothesis that PiLoT raw refined
    yaw and DOMDSMRenderer downward yaw are related by a 180-degree Euler branch
    equivalence. This hypothesis is not a replacement for calibration; it must
    be checked by ``tools/audit_pilot_domdsm_pose_convention.py`` at the matrix
    level before being treated as a convention fix.
    """
    init_t = [float(x) for x in initial_trans]
    init_e = [float(x) for x in initial_euler]
    ref_t = [float(x) for x in refined_trans]
    ref_e = [float(x) for x in refined_euler]

    if mode == "initial":
        return {
            "mode": mode,
            "translation_lon_lat_alt": init_t,
            "euler_pitch_roll_yaw": init_e,
            "debug_only": False,
            "formal_safe_candidate": True,
            "reason": "Initial DOM/DSM pose.",
        }

    if mode == "raw_debug_only":
        return {
            "mode": mode,
            "translation_lon_lat_alt": ref_t,
            "euler_pitch_roll_yaw": ref_e,
            "debug_only": True,
            "formal_safe_candidate": False,
            "reason": (
                "Raw PiLoT refined full pose may use a different Euler/camera "
                "convention and is retained only for diagnostics."
            ),
        }

    if mode == "corrected_downward_yaw_freeze_alt_pitch_roll":
        downward_yaw = refined_yaw_to_downward_yaw(ref_e[2])
        return {
            "mode": mode,
            "translation_lon_lat_alt": [ref_t[0], ref_t[1], init_t[2]],
            "euler_pitch_roll_yaw": make_downward_euler_from_yaw(downward_yaw),
            "debug_only": False,
            "formal_safe_candidate": True,
            "corrected_downward_yaw": downward_yaw,
            "reason": (
                "Use refined lon/lat with initial altitude and DOM/DSM "
                "downward-camera pitch/roll while adapting raw refined yaw by "
                "the current 180-degree branch hypothesis."
            ),
        }

    if mode == "swap_xy_freeze_alt_corrected_downward_yaw":
        if to_raster is None or from_raster is None:
            raise ValueError(
                "swap_xy_freeze_alt_corrected_downward_yaw requires "
                "to_raster and from_raster transformers"
            )
        raw_delta = compute_enu_delta_m(init_t, ref_t, to_raster)
        applied_delta = [raw_delta[1], raw_delta[0], 0.0]
        adapted_trans = apply_enu_offset(
            init_t,
            applied_delta[0],
            applied_delta[1],
            applied_delta[2],
            to_raster,
            from_raster,
        )
        downward_yaw = refined_yaw_to_downward_yaw(ref_e[2])
        return {
            "mode": mode,
            "translation_lon_lat_alt": adapted_trans,
            "euler_pitch_roll_yaw": make_downward_euler_from_yaw(downward_yaw),
            "debug_only": False,
            "formal_safe_candidate": True,
            "raw_delta_east_north_alt_m": raw_delta,
            "applied_delta_east_north_alt_m": applied_delta,
            "corrected_downward_yaw": downward_yaw,
            "reason": (
                "Apply the P9.3 experimental swap-XY ENU translation adapter "
                "with frozen altitude and corrected DOM/DSM downward yaw."
            ),
        }

    raise ValueError(f"Unknown DOM/DSM refined-pose adapter mode: {mode}")
