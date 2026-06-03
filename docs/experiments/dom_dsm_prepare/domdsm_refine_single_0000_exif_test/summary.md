# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0000.jpg`
- pose file: `data_caiwangcun/query/poses/exif_test_yawfix.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_single_0000_exif_test`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4368608889 | 30.3913609722 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4368735697 | 30.3913053166 | 381.934 | -179.787518 | -0.136119 | -149.552007 | 1.079 | -6.198 | -9.528 |
| corrected_downward_yaw | 114.4368735697 | 30.3913053166 | 391.462 | 0.000000 | 180.000000 | 30.447993 | 1.079 | -6.198 | 0.000 |
| safe_selected | 114.4368608889 | 30.3913609722 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.225790 | 7.156855 | 0.551118 | True | True |
| raw_refined_full | 0.319154 | 11.052322 | 0.304365 | False | False |
| corrected_downward_yaw | 0.272148 | 9.734839 | 0.401379 | False | False |
| safe_selected | 0.225790 | 7.156855 | 0.551118 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43687356974533, 30.391305316646882, 381.9341978710145]`
- raw refined euler pitch/roll/yaw: `[-179.78751801837214, -0.13611890171073499, -149.55200736424817]`
- raw refined ENU delta m: `[1.0790679772035219, -6.1980915744788945, -9.527802128985513]`

## Corrected Downward Yaw
- raw refined yaw: `-149.55200736424817`
- corrected downward yaw: `30.44799263575183`
- corrected candidate euler: `[0.0, 180.0, 30.44799263575183]`

## Safe Gate
- selected candidate: `initial`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `keep_initial_pose`
- raw refined should not be accepted directly under the visual gate.
