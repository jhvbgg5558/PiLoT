# DOM/DSM Refine Experiment Progress

This file is the running experiment log for local DOM/DSM refine work. Keep it updated after each experiment so a new agent can quickly recover the current assumptions, scripts, outputs, and conclusions.

## Fixed Environment

- OS: Ubuntu 22.04
- Windows project directory: `D:\aiproject\PiLoT_work`
- Ubuntu/WSL project directory: `/mnt/d/aiproject/PiLoT_work`
- Python used for experiments: `/home/farsee2/pilot22/bin/python`
- Current main config family: `configs/caiwangcun_domdsm.yaml`, `configs/caiwangcun_domdsm_16x9.yaml`

## Network Proxy

- Proxy app: StarlinkCloud
- Mixed HTTP/SOCKS5 port: `198.18.0.1:12450`
- Ubuntu 22.04 user env file: `/home/farsee2/.pilot_proxy_env`
- Auto-loaded by: `/home/farsee2/.bashrc`
- apt proxy file: `/etc/apt/apt.conf.d/95pilot-proxy`
- git global proxy:
  - `http.proxy=http://198.18.0.1:12450`
  - `https.proxy=http://198.18.0.1:12450`
- Verification on 2026-06-02:
  - `curl -I -L https://www.google.com` through the proxy returned `HTTP/2 200`.

## Active Scripts

- `pixloc/utils/dom_dsm/pose_adapter.py`
  - Contains experimental adapter mode `swap_xy_freeze_alt_corrected_downward_yaw`.
- `tools/validate_domdsm_refine_single_0000.py`
  - Single-frame refine validator with formal safe candidates.
- `tools/run_domdsm_refine_exif_test_batch_p9_4.py`
  - Batch runner for `data_caiwangcun/query/images/exif_test`.
- `tools/run_sim_sequence_p11_landcover_swapxy_freezealt_refine_only.py`
  - Sequence runner for refine-only propagation experiments.
  - Supports `--first-pose-source gt|init_noisy`.
  - Supports `--yaw-policy raw_corrected|constrained_search`.

## Current Experimental Rules

- P9.4 translation adapter:
  - Compute raw ENU delta from current initial pose to raw refined pose.
  - Apply swapped horizontal offset: `east = raw_north`, `north = raw_east`.
  - Freeze altitude to the current initial pose altitude.
- P9.4 old yaw behavior:
  - `corrected_downward_yaw = normalize(raw_refined_yaw + 180)`.
- Constrained yaw behavior:
  - Translation still uses `swap east/north + freeze alt`.
  - Yaw is selected from a trust-region search around previous accepted yaw.
  - Raw corrected yaw is treated as an observation, not accepted unconditionally.
- Sequence propagation rule:
  - Only frame 0 uses the selected first-pose source.
  - Every later frame uses the previous accepted selected pose as init.
  - No odom, no per-frame GT/noisy init, and no safe gate unless explicitly stated.
- Renderer backend rule:
  - The render backend must match the backend used to generate the query sequence.
  - Current simulated sequence images were generated with `prototype`; using `gpu_mesh` for validation caused invalid comparisons and large drift.
- Altitude interpretation:
  - `freeze alt` freezes altitude relative to the current propagated init pose.
  - Without odom/baro/GT height updates, altitude can remain near the first-frame height and produce growing alt error on terrain-changing sequences.

## Experiments

### 2026-06-02 P9.4 Exif Test Batch

Output:

- `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4`

Command pattern:

```bash
/home/farsee2/pilot22/bin/python tools/run_domdsm_refine_exif_test_batch_p9_4.py
```

Result:

- Images processed: 8
- Failures: 0
- Strict success count: 8 / 8
- Fallback to initial: 3
- New swap adapter selected: 4
- Selected candidate counts:
  - `initial`: 3
  - `line_search_scale_0p50`: 1
  - `swap_xy_freeze_alt_corrected_downward_yaw`: 4

Conclusion:

- The P9.4 adapter is a valid formal safe candidate for the exif test batch.
- Safe gate still falls back to initial on some frames, which is expected and should be counted explicitly.

### 2026-06-02 Frame 0000 Altitude Ablation

Outputs:

- `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4_alt_ablation_0000`
- `docs/experiments/dom_dsm_prepare/domdsm_refine_exif_test_batch_p9_4_alt_sweep_0000`

Compared modes:

- Swap east/north with freeze alt.
- Swap east/north while keeping raw refined alt.

Key result on `0000.jpg`:

- Freeze alt: chamfer `7.076263`, overlap `0.701058`
- Keep raw refined alt: chamfer `8.875958`, overlap `0.524652`
- Raw refined altitude delta around `-10.11 m` was worse than freezing altitude.

Conclusion:

- The bad refined altitude is not explained as a simple axis sign or swap issue.
- Freezing altitude currently works better for this case.
- Altitude likely needs an independent constraint or height model rather than directly trusting refine output.

### 2026-06-02 Landcover Sequence, Raw Corrected Yaw

Sequence:

- `docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover`

Output:

- `docs/experiments/dom_dsm_prepare/sim_sequence_p11_landcover_swapxy_freezealt_refine_only_prototype`

Command pattern:

```bash
/home/farsee2/pilot22/bin/python tools/run_sim_sequence_p11_landcover_swapxy_freezealt_refine_only.py \
  --sequence-dir docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover \
  --yaw-policy raw_corrected \
  --output-dir docs/experiments/dom_dsm_prepare/sim_sequence_p11_landcover_swapxy_freezealt_refine_only_prototype \
  --save-debug-frames
```

Result:

- Frames: 100
- Refine success: 100
- Failures: 0
- XY RMSE: `350.7788 m`
- Final XY error: `746.4759 m`
- Yaw RMSE: `37.7191 deg`

Observed issue:

- Large yaw jump around frame 13 caused trajectory drift.
- The old logic let `raw_refined_yaw + 180` enter the propagation chain unconditionally.

Conclusion:

- Raw corrected yaw is unsafe for recursive sequence propagation.
- A constrained per-frame yaw selection module is needed.

### 2026-06-02 Landcover Sequence, Constrained Yaw, GT First Pose

Sequence:

- `docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover`

Output:

- `docs/experiments/dom_dsm_prepare/sim_sequence_p11_landcover_constrained_yaw`

Command:

```bash
/home/farsee2/pilot22/bin/python tools/run_sim_sequence_p11_landcover_swapxy_freezealt_refine_only.py \
  --sequence-dir docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11_landcover \
  --yaw-policy constrained_search \
  --output-dir docs/experiments/dom_dsm_prepare/sim_sequence_p11_landcover_constrained_yaw \
  --save-debug-frames
```

Result:

- Frames: 100
- Refine success: 100
- Failures: 0
- XY RMSE: `0.5935955891 m`
- Final XY error: `0.727724113 m`
- Max XY error: `0.981886069 m`
- Yaw RMSE: `0.177009195 deg`
- Max yaw error: `0.314289089 deg`

Conclusion:

- Constrained yaw prevents the frame-13-style yaw jump.
- Recursive refine-only propagation is stable on this landcover simulated sequence when backend is matched to `prototype`.

### 2026-06-02 Sim Flight Sequence P11, Constrained Yaw, GT First Pose

Sequence:

- `docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11`

Output:

- `docs/experiments/dom_dsm_prepare/sim_sequence_p11_constrained_yaw`

Command:

```bash
/home/farsee2/pilot22/bin/python tools/run_sim_sequence_p11_landcover_swapxy_freezealt_refine_only.py \
  --sequence-dir docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11 \
  --yaw-policy constrained_search \
  --output-dir docs/experiments/dom_dsm_prepare/sim_sequence_p11_constrained_yaw \
  --save-debug-frames
```

Result:

- Frames: 100
- Refine success: 100
- Failures: 0
- Backend: `prototype`
- XY RMSE: `0.8327715016626732 m`
- XY mean: `0.8178677839571153 m`
- XY median: `0.8232888004118246 m`
- XY max: `1.139843056746994 m`
- Final XY error: `0.9242701602675435 m`
- Yaw RMSE: `0.17288074956642768 deg`
- Yaw max: `0.3221924484717533 deg`
- Final yaw error: `0.17780755152824668 deg`
- Alt RMSE: `6.585790339814959 m`
- Alt max: `11.25800000000001 m`
- Final alt error: `2.9950000000000045 m`

Conclusion:

- Horizontal and yaw errors are small with GT first pose and constrained yaw.
- Altitude error remains large because freeze-alt does not track changing height.

### 2026-06-02 Sim Flight Sequence P11, Constrained Yaw, Noisy First Pose

Sequence:

- `docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11`

Output:

- `docs/experiments/dom_dsm_prepare/sim_sequence_p11_constrained_yaw_init_noisy`

Command:

```bash
/home/farsee2/pilot22/bin/python tools/run_sim_sequence_p11_landcover_swapxy_freezealt_refine_only.py \
  --sequence-dir docs/experiments/dom_dsm_prepare/sim_flight_sequence_p11 \
  --first-pose-source init_noisy \
  --yaw-policy constrained_search \
  --output-dir docs/experiments/dom_dsm_prepare/sim_sequence_p11_constrained_yaw_init_noisy \
  --save-debug-frames
```

Important constraint:

- `init_noisy_poses.txt` is used only for frame 0.
- Frames 1-99 still use previous accepted selected pose only.
- No odom, no safe gate, no per-frame noisy or GT init.

Result:

- Frames: 100
- Refine success: 100
- Failures: 0
- XY RMSE: `0.581281954281482 m`
- XY mean: `0.5151141653523643 m`
- XY median: `0.4901153758202914 m`
- XY max: `2.421755433667737 m`
- Final XY error: `0.48432287047869993 m`
- Yaw RMSE: `0.16739012657435948 deg`
- Yaw max: `0.5206139999999948 deg`
- Final yaw error: `0.16207441252808508 deg`
- Alt RMSE: `7.577843608837549 m`
- Alt max: `12.482000000000014 m`
- Final alt error: `4.219000000000008 m`

Conclusion:

- With a lightly perturbed first pose, constrained yaw recursive propagation still keeps XY/yaw stable on this sequence.
- Altitude error is worse than GT-first-pose because freeze-alt preserves the initial height bias.

## Open Questions And Next Steps

- Add a dedicated altitude strategy. Directly trusting raw refined altitude is currently worse in tested cases, while freeze-alt cannot follow terrain or flight height changes.
- Test constrained yaw on real exif/query images, not only simulated prototype-rendered sequences.
- Compare against a future method that uses barometer, DEM height prior, or explicit altitude line search.
- Keep backend provenance in every new experiment summary to avoid prototype/gpu_mesh mismatches.
