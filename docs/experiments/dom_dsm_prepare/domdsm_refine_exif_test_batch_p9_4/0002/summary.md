# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0002.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0002.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0002`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4386081667 | 30.3886414722 | 391.443 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4386245284 | 30.3886087204 | 380.650 | 179.774380 | -0.016148 | -150.175350 | 1.490 | -3.667 | -10.793 |
| corrected_downward_yaw | 114.4386245284 | 30.3886087204 | 391.443 | 0.000000 | 180.000000 | 29.824650 | 1.490 | -3.667 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4385696866 | 30.3886541607 | 391.443 | 0.000000 | 180.000000 | 29.824650 | -3.667 | 1.490 | 0.000 |
| safe_selected | 114.4385696866 | 30.3886541607 | 391.443 | 0.000000 | 180.000000 | 29.824650 | -3.667 | 1.490 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.260145 | 4.344333 | 0.578793 | True | False |
| raw_refined_full | 0.339885 | 5.561292 | 0.515882 | False | False |
| corrected_downward_yaw | 0.305382 | 4.802809 | 0.504622 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.243957 | 3.925602 | 0.599243 | True | True |
| safe_selected | 0.243957 | 3.925602 | 0.599243 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43862452837146, 30.38860872041244, 380.6501564802602]`
- raw refined euler pitch/roll/yaw: `[179.77438042784897, -0.016148491895614302, -150.17535005947212]`
- raw refined ENU delta m: `[1.4904611799283884, -3.666750471573323, -10.79284351973979]`

## Corrected Downward Yaw
- raw refined yaw: `-150.17535005947212`
- corrected downward yaw: `29.824649940527877`
- corrected candidate euler: `[0.0, 180.0, 29.824649940527877]`

## Swap-XY Adapter
- raw refined ENU delta m: `[1.4904611799283884, -3.666750471573323, -10.79284351973979]`
- applied swapped ENU delta m: `[-3.666750471573323, 1.4904611799283884, 0.0]`
- candidate euler: `[0.0, 180.0, 29.824649940527877]`

## Safe Gate
- selected candidate: `swap_xy_freeze_alt_corrected_downward_yaw`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `accept_selected_candidate`
- raw refined should not be accepted directly under the visual gate.
