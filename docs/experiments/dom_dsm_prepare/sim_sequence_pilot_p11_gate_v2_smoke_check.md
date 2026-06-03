# P11.3 Pose-Consistency and Temporal Safe Gate Check

## Questions
1. P11.2 selected yaw RMSE issue: raw refined yaw RMSE is `179.1194588735348`, corrected refined yaw RMSE is `1.2055756725400149`, so this run distinguishes yaw convention from true pose jumps.
2. Corrected refined yaw improvement: `1.2055756725400149` vs raw `179.1194588735348`.
3. Pose gate accept ratio: `0.7857142857142857`.
4. Temporal gate accept ratio: `0.42857142857142855`.
5. selected_v2 XY RMSE: `2.784194517971351`, init `4.308403169742424`, old selected `3.5947919113321647`.
6. selected_v2 yaw RMSE: `0.5963217147027213`, old selected yaw RMSE `73.83417395771424`.
7. Recommendation: Tune safe gate and then proceed to temporal smoothing with selected poses.

## Interpretation
Raw refinement is unstable on some frames, but safe gate prevents regressions against init.

## Renderer Note
- Requested renderer: `gpu_mesh`.
- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.3.
