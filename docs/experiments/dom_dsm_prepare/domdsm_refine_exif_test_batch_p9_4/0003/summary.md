# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0003.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0003.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0003`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4434653333 | 30.3830338889 | 391.433 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4435287436 | 30.3827316429 | 396.551 | -176.175489 | -0.786239 | -150.505033 | 5.338 | -33.647 | 5.118 |
| corrected_downward_yaw | 114.4435287436 | 30.3827316429 | 391.433 | 0.000000 | 180.000000 | 29.494967 | 5.338 | -33.647 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4431142108 | 30.3830751644 | 391.433 | 0.000000 | 180.000000 | 29.494967 | -33.647 | 5.338 | 0.000 |
| safe_selected | 114.4434970385 | 30.3828827659 | 391.433 | 0.000000 | 180.000000 | 29.200000 | 2.669 | -16.824 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.478342 | 39.441666 | 0.034747 | True | False |
| raw_refined_full | 0.522098 | 40.871300 | 0.024206 | False | False |
| corrected_downward_yaw | 0.494213 | 44.098709 | 0.035841 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.478609 | 46.492973 | 0.042832 | False | False |
| safe_selected | 0.478605 | 38.721581 | 0.043706 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.44352874358694, 30.382731642858417, 396.5511150388047]`
- raw refined euler pitch/roll/yaw: `[-176.17548874888158, -0.7862393784991776, -150.5050327791441]`
- raw refined ENU delta m: `[5.33841093594674, -33.64731531450525, 5.118115038804717]`

## Corrected Downward Yaw
- raw refined yaw: `-150.5050327791441`
- corrected downward yaw: `29.494967220855898`
- corrected candidate euler: `[0.0, 180.0, 29.494967220855898]`

## Swap-XY Adapter
- raw refined ENU delta m: `[5.33841093594674, -33.64731531450525, 5.118115038804717]`
- applied swapped ENU delta m: `[-33.64731531450525, 5.33841093594674, 0.0]`
- candidate euler: `[0.0, 180.0, 29.494967220855898]`

## Safe Gate
- selected candidate: `line_search_scale_0p50`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `accept_selected_candidate`
- raw refined should not be accepted directly under the visual gate.
