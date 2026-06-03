# P11.3 Pose-Consistency and Temporal Safe Gate Check

## Questions
1. P11.2 selected yaw RMSE issue: raw refined yaw RMSE is `176.92723820355707`, corrected refined yaw RMSE is `4.102395829972316`, so this run distinguishes yaw convention from true pose jumps.
2. Corrected refined yaw improvement: `4.102395829972316` vs raw `176.92723820355707`.
3. Pose gate accept ratio: `0.7595190380761523`.
4. Temporal gate accept ratio: `0.29458917835671344`.
5. selected_v2 XY RMSE: `2.756603611314328`, init `3.861568477851895`, old selected `3.5947919113321647`.
6. selected_v2 yaw RMSE: `1.0235596297867737`, old selected yaw RMSE `73.83417395771424`.
7. Recommendation: Tune safe gate and then proceed to temporal smoothing with selected poses.

## Interpretation
Raw refinement is unstable on some frames, but safe gate prevents regressions against init.

## Renderer Note
- Requested renderer: `gpu_mesh`.
- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.3.
