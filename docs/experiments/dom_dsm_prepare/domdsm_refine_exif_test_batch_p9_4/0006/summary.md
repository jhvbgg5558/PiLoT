# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0006.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0006.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0006`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4399703611 | 30.3904358889 | 391.488 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4399665963 | 30.3904024927 | 382.449 | 179.396140 | -0.186017 | -152.302974 | -0.446 | -3.694 | -9.039 |
| corrected_downward_yaw | 114.4399665963 | 30.3904024927 | 391.488 | 0.000000 | 180.000000 | 27.697026 | -0.446 | -3.694 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4399320481 | 30.3904311184 | 391.488 | 0.000000 | 180.000000 | 27.697026 | -3.694 | -0.446 | 0.000 |
| safe_selected | 114.4399703611 | 30.3904358889 | 391.488 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.277654 | 4.195574 | 0.603389 | True | True |
| raw_refined_full | 0.306882 | 4.680187 | 0.630705 | False | False |
| corrected_downward_yaw | 0.332770 | 4.757798 | 0.585639 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.291772 | 4.497944 | 0.602697 | False | False |
| safe_selected | 0.277654 | 4.195574 | 0.603389 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43996659625327, 30.39040249270704, 382.4485569810495]`
- raw refined euler pitch/roll/yaw: `[179.39614033058746, -0.1860171639764631, -152.30297405932893]`
- raw refined ENU delta m: `[-0.44560727971838787, -3.6944211246445775, -9.039443018950521]`

## Corrected Downward Yaw
- raw refined yaw: `-152.30297405932893`
- corrected downward yaw: `27.697025940671068`
- corrected candidate euler: `[0.0, 180.0, 27.697025940671068]`

## Swap-XY Adapter
- raw refined ENU delta m: `[-0.44560727971838787, -3.6944211246445775, -9.039443018950521]`
- applied swapped ENU delta m: `[-3.6944211246445775, -0.44560727971838787, 0.0]`
- candidate euler: `[0.0, 180.0, 27.697025940671068]`

## Safe Gate
- selected candidate: `initial`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `keep_initial_pose`
- raw refined should not be accepted directly under the visual gate.
