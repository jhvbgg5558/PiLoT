# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0007.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0007.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0007`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4407033889 | 30.3892964722 | 391.482 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4406210401 | 30.3891883898 | 385.559 | -179.935108 | -1.694263 | -152.243681 | -8.186 | -11.804 | -5.923 |
| corrected_downward_yaw | 114.4406210401 | 30.3891883898 | 391.482 | 0.000000 | 180.000000 | 27.756319 | -8.186 | -11.804 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4405825667 | 30.3892202688 | 391.482 | 0.000000 | 180.000000 | 27.756319 | -11.804 | -8.186 | 0.000 |
| safe_selected | 114.4407033889 | 30.3892964722 | 391.482 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.268402 | 3.697081 | 0.616970 | True | True |
| raw_refined_full | 0.440997 | 5.659720 | 0.506283 | False | False |
| corrected_downward_yaw | 0.396000 | 5.018207 | 0.504373 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.381833 | 4.762780 | 0.543480 | False | False |
| safe_selected | 0.268402 | 3.697081 | 0.616970 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.44062104014363, 30.389188389837624, 385.5589253306389]`
- raw refined euler pitch/roll/yaw: `[-179.93510824698245, -1.6942633743161684, -152.24368140862902]`
- raw refined ENU delta m: `[-8.186008890566882, -11.804015052039176, -5.923074669361142]`

## Corrected Downward Yaw
- raw refined yaw: `-152.24368140862902`
- corrected downward yaw: `27.756318591370984`
- corrected candidate euler: `[0.0, 180.0, 27.756318591370984]`

## Swap-XY Adapter
- raw refined ENU delta m: `[-8.186008890566882, -11.804015052039176, -5.923074669361142]`
- applied swapped ENU delta m: `[-11.804015052039176, -8.186008890566882, 0.0]`
- candidate euler: `[0.0, 180.0, 27.756318591370984]`

## Safe Gate
- selected candidate: `initial`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `keep_initial_pose`
- raw refined should not be accepted directly under the visual gate.
