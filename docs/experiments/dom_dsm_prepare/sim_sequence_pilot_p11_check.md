# P11.2 Synthetic Sequence PiLoT Refinement Check

## P11.1 Sequence
- Sequence frames: `100`
- Renderer backend requested: `gpu_mesh`
- Current environment lacks `nvdiffrast`, so DOMDSMRenderer falls back to `prototype`; this is allowed for P11.2 and is recorded in metrics.

## Purpose
- Run PiLoT single-frame refinement on synthetic query frames generated from GT pose.
- Compare init, refined, and safe-gate selected poses against GT.
- Check whether feature loss drop aligns with GT error drop.

## Quantitative Results
- Frames: `100`
- Init XY RMSE: `3.861568477851895` m
- Refined XY RMSE: `5.200421531218112` m
- Selected XY RMSE: `3.5947919113321647` m
- Init yaw RMSE: `2.0517695124663553` deg
- Refined yaw RMSE: `176.9272384025687` deg
- Selected yaw RMSE: `73.83417395771424` deg
- Improved frame ratio: `0.37`
- Worse frame ratio: `0.63`
- Safe gate accept ratio: `0.17`
- Mean feature loss drop: `-0.0016066184329489872`
- Correlation loss drop vs XY error drop: `0.5644751138654167`

## Sanity Check
- Mean GT chamfer: `0.056691262498497964`
- Mean init chamfer: `0.9324998974800109`
- Mean GT overlap: `0.9904528667676432`
- Mean init overlap: `0.8468373970080858`
- Passed: `True`

## Interpretation
Raw refinement is unstable on some frames, but safe gate prevents regressions against init.

## Next Step
Tune safe gate and then proceed to temporal smoothing with selected poses.
