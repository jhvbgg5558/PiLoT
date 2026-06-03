# P9.2 PiLoT / DOMDSM Pose Convention Audit

## Purpose
The DOM+DSM single-frame run shows raw refined full pose changes near 180 degrees in Euler components and a large altitude jump. This report audits the matrix-level convention chain before any adapter fix is treated as final.

## Known Results
| Candidate | East Delta | North Delta | Alt Delta | Roll Delta | Pitch Delta | Yaw Delta | Chamfer | Overlap | Feature loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 7.157 | 0.551 | 0.226 |
| raw_refined_full | 1.079 | -6.198 | -9.528 | -180.136 | -179.788 | -178.752 | 11.052 | 0.304 | 0.319 |
| corrected_downward_yaw | 1.079 | -6.198 | 0.000 | 0.000 | 0.000 | 1.248 | 9.735 | 0.401 | 0.272 |

## Convention Audit
- PiLoT/pose-file Euler order: `[pitch, roll, yaw], scipy/transform xyz`.
- ECEF matrix helper: `pixloc.utils.transform.euler_angles_to_matrix_ECEF -> ECEF camera-to-world`.
- DOMDSM renderer Euler helper: `scipy.spatial.transform.Rotation.from_euler('xyz', euler) -> raster/world camera-to-world`.
- Pose dictionary flips camera Y/Z before forming PixLoc `T_w2c`: `src.utils.pose_utils.load_pose_dict flips c2w camera Y/Z before building PixLoc T_w2c`.

### Camera Axes
| Candidate | DOM local z/viewing direction | Downward local? | ECEF z/viewing direction |
|---|---:|---:|---:|
| initial | `[ 1.06902123e-16  5.97455770e-17 -1.00000000e+00]` | True | `[ 0.35684504 -0.78531717 -0.50590371]` |
| raw_refined_full | `[-1.68764170e-04 -4.40096296e-03 -9.99990301e-01]` | True | `[ 0.35607453 -0.7832131  -0.50969419]` |
| corrected_downward_yaw | `[ 1.05575514e-16  6.20597179e-17 -1.00000000e+00]` | True | `[ 0.35684541 -0.78531754 -0.50590287]` |

### Matrix Relative Rotation Angles
| Pair | Angle deg |
|---|---:|
| initial_to_raw_refined_full | 1.273491 |
| initial_to_corrected_downward_yaw | 1.247999 |
| raw_refined_full_to_corrected_downward_yaw | 0.252343 |

### Round Trip
| Candidate | Max abs R diff | Relative angle deg | Input Euler | Roundtrip Euler |
|---|---:|---:|---|---|
| initial | 0.000000000 | 0.000000000 | `[0.0, 180.0, 29.19999999999999]` | `[180.0, 1.2722218725854067e-14, -150.8]` |
| raw_refined_full | 0.000000000 | 0.000000000 | `[-179.78751801837214, -0.13611890171073499, -149.55200736424817]` | `[-179.78751801837214, -0.13611890171072225, -149.55200736424817]` |
| corrected_downward_yaw | 0.000000000 | 0.000000000 | `[0.0, 180.0, 30.44799263575183]` | `[180.0, 2.5444437451708134e-14, -149.55200736424817]` |

### Translation Interpretation
- raw_refined_full delta: `{'east_m': 1.0790679772035219, 'north_m': -6.1980915744788945, 'alt_m': -9.527802128985513, 'pitch_deg': -179.78751801837214, 'roll_deg': -180.13611890171074, 'yaw_deg': -178.75200736424816}`
- corrected_downward_yaw delta: `{'east_m': 1.0790679772035219, 'north_m': -6.1980915744788945, 'alt_m': 0.0, 'pitch_deg': 0.0, 'roll_deg': 0.0, 'yaw_deg': 1.2479926357518423}`
- interpretation: Summary translations are WGS84 lon/lat/alt camera centers. ENU deltas are computed through the DOM/DSM raster CRS via pose_adapter.compute_enu_delta_m, matching the renderer's WGS84-to-raster transform path.
- grid/local comparison: grid_best not found in current summary; rerun local search or provide its summary for delta comparison.

## Decision
- raw_refined_full as formal candidate: `False`.
- corrected_downward_yaw status: reasonable adapter candidate for yaw branch testing
- translation delta same coordinate chain: `True`.
- check east/north sign or axis swap next: `True`.
- raw 180-degree change classification: mostly Euler branch/convention equivalent

## Next Step
- situation: `D`
- recommendation: Translation is in the expected WGS84/DOMDSM chain, but no grid/local best is available for direction comparison. Rerun east/north/yaw local search or P11 scorer before accepting raw objective updates.
