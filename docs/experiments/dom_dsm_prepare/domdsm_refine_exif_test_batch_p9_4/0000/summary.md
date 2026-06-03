# DOM+DSM 0000 Single-Frame Refine Validation

## Inputs
- config: `configs/caiwangcun_domdsm.yaml`
- query image: `data_caiwangcun/query/images/exif_test/0000.jpg`
- pose file: `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/_pose_files/0000.txt`
- output dir: `/mnt/d/aiproject/PiLoT_work/docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4/0000`

## Pose Summary
| Candidate | Lon | Lat | Alt | Pitch | Roll | Yaw | ENU East | ENU North | ENU Alt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 114.4368608889 | 30.3913609722 | 391.462 | 0.000000 | 180.000000 | 29.200000 | 0.000 | 0.000 | 0.000 |
| raw_refined_full | 114.4368836147 | 30.3913180680 | 381.352 | -179.901672 | 0.083273 | -149.542420 | 2.077 | -4.806 | -10.110 |
| corrected_downward_yaw | 114.4368836147 | 30.3913180680 | 391.462 | 0.000000 | 180.000000 | 30.457580 | 2.077 | -4.806 | 0.000 |
| swap_xy_freeze_alt_corrected_downward_yaw | 114.4368104202 | 30.3913787110 | 391.462 | 0.000000 | 180.000000 | 30.457580 | -4.806 | 2.077 | 0.000 |
| safe_selected | 114.4368104202 | 30.3913787110 | 391.462 | 0.000000 | 180.000000 | 30.457580 | -4.806 | 2.077 | 0.000 |

## Metrics
| Candidate | Feature loss | Visual chamfer | Visual overlap | Safe gate | Selected |
|---|---:|---:|---:|---|---|
| initial | 0.225790 | 7.156855 | 0.551118 | True | False |
| raw_refined_full | 0.312267 | 10.900821 | 0.356304 | False | False |
| corrected_downward_yaw | 0.297358 | 9.428010 | 0.404968 | False | False |
| swap_xy_freeze_alt_corrected_downward_yaw | 0.224465 | 7.076263 | 0.701058 | True | True |
| safe_selected | 0.224465 | 7.076263 | 0.701058 | True | True |

## Raw Refined Delta
- raw refined translation: `[114.43688361470532, 30.3913180679821, 381.35199227090925]`
- raw refined euler pitch/roll/yaw: `[-179.90167206362946, 0.08327330320288644, -149.54241981943423]`
- raw refined ENU delta m: `[2.076548933109734, -4.806225331034511, -10.11000772909074]`

## Corrected Downward Yaw
- raw refined yaw: `-149.54241981943423`
- corrected downward yaw: `30.45758018056577`
- corrected candidate euler: `[0.0, 180.0, 30.45758018056577]`

## Swap-XY Adapter
- raw refined ENU delta m: `[2.076548933109734, -4.806225331034511, -10.11000772909074]`
- applied swapped ENU delta m: `[-4.806225331034511, 2.076548933109734, 0.0]`
- candidate euler: `[0.0, 180.0, 30.45758018056577]`

## Safe Gate
- selected candidate: `swap_xy_freeze_alt_corrected_downward_yaw`
- policy: `chamfer<=initial and overlap>=initial, then min(chamfer-overlap), fallback initial`
- raw refined visually degrades initial: `True`
- recommendation: `accept_selected_candidate`
- raw refined should not be accepted directly under the visual gate.
