# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0005.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0005.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0005`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4389233889 | 30.3920651667 | 391.457 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4389378748 | 30.3920155991 | 382.183 | 179.893257 | 0.120287 | -151.983746 | 1.268 | -5.527 | -9.274 |
| corrected_downward_yaw | 114.4389378748 | 30.3920155991 | 391.457 | 0.000000 | 180.000000 | 28.016254 | 1.268 | -5.527 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4388656148 | 30.3920754696 | 391.457 | 0.000000 | 180.000000 | 28.016254 | -5.527 | 1.268 | 0.000 |
| safe_selected | 114.4389233889 | 30.3920651667 | 391.457 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.222734 | 5.147426 | 0.479260 | True | True |
| raw_refined_full | 0.303109 | 6.910688 | 0.442614 | False | False |
| corrected_downward_yaw | 0.281987 | 6.626711 | 0.454171 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.218284 | 5.795901 | 0.448919 | False | False |
| safe_selected | 0.222734 | 5.147426 | 0.479260 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43893787479823, 30.392015599120757, 382.1830312535167]`
- raw refined euler pitch/roll/yaw: `[179.89325717046364, 0.12028691286367428, -151.98374586674063]`
- raw refined ENU delta m: `[1.2679319720482454, -5.527022658847272, -9.27396874648332]`

## Corrected Downward Yaw
- raw refined yaw: `-151.98374586674063`
- corrected downward yaw: `28.016254133259366`
- corrected candidate euler: `[0.0, 180.0, 28.016254133259366]`

## Swap-XY Adapter
- raw refined ENU delta m: `[1.2679319720482454, -5.527022658847272, -9.27396874648332]`
- applied swapped ENU delta m: `[-5.527022658847272, 1.2679319720482454, 0.0]`
- candidate euler: `[0.0, 180.0, 28.016254133259366]`

## Safe Gate
- selected candidate: `initial`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `keep_initial_pose`
- raw refined should not be accepted directly under the visual gate.
