# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0004.JPG`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0004.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0004`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4427668611 | 30.3841223889 | 391.455 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4432169426 | 30.3841460134 | 391.641 | -178.701522 | 3.304919 | -154.844786 | 43.322 | 1.642 | 0.186 |
| corrected_downward_yaw | 114.4432169426 | 30.3841460134 | 391.455 | 0.000000 | 180.000000 | 25.155214 | 43.322 | 1.642 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4427737592 | 30.3845132693 | 391.455 | 0.000000 | 180.000000 | 25.155214 | 1.642 | 43.322 | 0.000 |
| safe_selected | 114.4427737592 | 30.3845132693 | 391.455 | 0.000000 | 180.000000 | 25.155214 | 1.642 | 43.322 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.601249 | 34.862686 | 0.143002 | True | False |
| raw_refined_full | 0.573761 | 34.042625 | 0.210207 | True | False |
| corrected_downward_yaw | 0.595918 | 32.312111 | 0.167256 | True | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.571514 | 26.829132 | 0.196492 | True | True |
| safe_selected | 0.571514 | 26.829132 | 0.196492 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.44321694255096, 30.384146013434478, 391.64062968827784]`
- raw refined euler pitch/roll/yaw: `[-178.7015215981222, 3.3049188546372603, -154.84478637114591]`
- raw refined ENU delta m: `[43.32150440305122, 1.6420209701173007, 0.18562968827785653]`

## Corrected Downward Yaw
- raw refined yaw: `-154.84478637114591`
- corrected downward yaw: `25.155213628854085`
- corrected candidate euler: `[0.0, 180.0, 25.155213628854085]`

## Swap-XY Adapter
- raw refined ENU delta m: `[43.32150440305122, 1.6420209701173007, 0.18562968827785653]`
- applied swapped ENU delta m: `[1.6420209701173007, 43.32150440305122, 0.0]`
- candidate euler: `[0.0, 180.0, 25.155213628854085]`

## Safe Gate
- selected candidate: `swap_xy_freeze_alt_corrected_downward_yaw`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `False`
- recommendation: `accept_selected_candidate`
