# P11.3 Pose-Consistency and Temporal Safe Gate Check

## Questions
1. P11.2 selected yaw RMSE issue: raw refined yaw RMSE is `177.95922771185002`, corrected refined yaw RMSE is `2.040775723317524`, so this run distinguishes yaw convention from true pose jumps.
2. Corrected refined yaw improvement: `2.040775723317524` vs raw `177.95922771185002`.
3. Pose gate accept ratio: `0.7995991983967936`.
4. Temporal gate accept ratio: `0.20641282565130262`.
5. selected_v2 XY RMSE: `4.080530016881124`, init `4.080530016881124`, old selected `3.5947919113321647`.
6. selected_v2 yaw RMSE: `1.0206139999999948`, old selected yaw RMSE `73.83417395771424`.
7. Recommendation: Tune safe gate and then proceed to temporal smoothing with selected poses.

## Interpretation
Raw refinement is unstable on some frames, but safe gate prevents regressions against init.

## Renderer Note
- Requested renderer: `gpu_mesh`.
- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.3.
