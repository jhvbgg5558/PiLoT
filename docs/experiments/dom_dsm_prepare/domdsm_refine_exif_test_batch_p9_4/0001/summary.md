# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0001.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0001.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0001`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4379083611 | 30.3897293889 | 391.490 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4380328144 | 30.3896554770 | 381.836 | -178.892259 | 1.032010 | -149.935161 | 11.776 | -8.465 | -9.654 |
| corrected_downward_yaw | 114.4380328144 | 30.3896554770 | 391.490 | 0.000000 | 180.000000 | 30.064839 | 11.776 | -8.465 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4378175598 | 30.3898338262 | 391.490 | 0.000000 | 180.000000 | 30.064839 | -8.465 | 11.776 | 0.000 |
| safe_selected | 114.4378175598 | 30.3898338262 | 391.490 | 0.000000 | 180.000000 | 30.064839 | -8.465 | 11.776 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.239829 | 5.136772 | 0.556565 | True | False |
| raw_refined_full | 0.446068 | 9.634475 | 0.415081 | False | False |
| corrected_downward_yaw | 0.351729 | 7.774983 | 0.443620 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.261262 | 5.000467 | 0.569239 | True | True |
| safe_selected | 0.261262 | 5.000467 | 0.569239 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.4380328144089, 30.38965547697051, 381.83636260963976]`
- raw refined euler pitch/roll/yaw: `[-178.89225948529506, 1.03201000837189, -149.93516094064947]`
- raw refined ENU delta m: `[11.77644533966668, -8.465325868222862, -9.653637390360245]`

## Corrected Downward Yaw
- raw refined yaw: `-149.93516094064947`
- corrected downward yaw: `30.064839059350533`
- corrected candidate euler: `[0.0, 180.0, 30.064839059350533]`

## Swap-XY Adapter
- raw refined ENU delta m: `[11.77644533966668, -8.465325868222862, -9.653637390360245]`
- applied swapped ENU delta m: `[-8.465325868222862, 11.77644533966668, 0.0]`
- candidate euler: `[0.0, 180.0, 30.064839059350533]`

## Safe Gate
- selected candidate: `swap_xy_freeze_alt_corrected_downward_yaw`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `accept_selected_candidate`
- raw refined should not be accepted directly under the visual gate.
