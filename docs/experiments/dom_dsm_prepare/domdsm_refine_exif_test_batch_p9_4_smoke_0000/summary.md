# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0000.jpg`
- pose file: `data_caiwangcun/query/poses/exif_test_yawfix.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4_smoke_0000`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4368608889 | 30.3913609722 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4368724915 | 30.3913121396 | 381.703 | -179.905571 | -0.089859 | -149.560306 | 0.993 | -5.439 | -9.759 |
| corrected_downward_yaw | 114.4368724915 | 30.3913121396 | 391.462 | 0.000000 | 180.000000 | 30.439694 | 0.993 | -5.439 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4368040923 | 30.3913688096 | 391.462 | 0.000000 | 180.000000 | 30.439694 | -5.439 | 0.993 | 0.000 |
| safe_selected | 114.4368040923 | 30.3913688096 | 391.462 | 0.000000 | 180.000000 | 30.439694 | -5.439 | 0.993 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.225790 | 7.156855 | 0.551118 | True | False |
| raw_refined_full | 0.324237 | 10.281864 | 0.353972 | False | False |
| corrected_downward_yaw | 0.282648 | 9.065163 | 0.433522 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.216063 | 6.488281 | 0.694591 | True | True |
| safe_selected | 0.216063 | 6.488281 | 0.694591 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43687249147148, 30.391312139645418, 381.7027398282662]`
- raw refined euler pitch/roll/yaw: `[-179.90557088633366, -0.08985873350965295, -149.56030644349056]`
- raw refined ENU delta m: `[0.9925618687702809, -5.439283803105354, -9.759260171733786]`

## Corrected Downward Yaw
- raw refined yaw: `-149.56030644349056`
- corrected downward yaw: `30.43969355650944`
- corrected candidate euler: `[0.0, 180.0, 30.43969355650944]`

## Swap-XY Adapter
- raw refined ENU delta m: `[0.9925618687702809, -5.439283803105354, -9.759260171733786]`
- applied swapped ENU delta m: `[-5.439283803105354, 0.9925618687702809, 0.0]`
- candidate euler: `[0.0, 180.0, 30.43969355650944]`

## Safe Gate
- selected candidate: `swap_xy_freeze_alt_corrected_downward_yaw`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `accept_selected_candidate`
- raw refined should not be accepted directly under the visual gate.
