# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm_16x9.yaml`
- query image: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- pose file: `data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_single_0000`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4368608916 | 30.3913609745 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4369989146 | 30.3912781173 | 392.061 | -179.079290 | 1.040671 | -146.240165 | 13.058 | -9.487 | 0.599 |
| corrected_downward_yaw | 114.4369989146 | 30.3912781173 | 391.462 | 0.000000 | 180.000000 | 33.759835 | 13.058 | -9.487 | 0.000 |
| safe_selected | 114.4368608916 | 30.3913609745 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.363285 | 15.761106 | 0.177052 | True | True |
| raw_refined_full | 0.414160 | 22.615854 | 0.096694 | False | False |
| corrected_downward_yaw | 0.381518 | 18.821278 | 0.137094 | False | False |
| safe_selected | 0.363285 | 15.761106 | 0.177052 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43699891455707, 30.391278117346396, 392.0610331892967]`
- raw refined euler pitch/roll/yaw: `[-179.07928975593106, 1.0406711321986035, -146.2401651294004]`
- raw refined ENU delta m: `[13.057944558910094, -9.486736088059843, 0.5990331892967333]`

## Corrected Downward Yaw
- raw refined yaw: `-146.2401651294004`
- corrected downward yaw: `33.759834870599605`
- corrected candidate euler: `[0.0, 180.0, 33.759834870599605]`

## Safe Gate
- selected candidate: `initial`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `keep_initial_pose`
- raw refined should not be accepted directly under the visual gate.
