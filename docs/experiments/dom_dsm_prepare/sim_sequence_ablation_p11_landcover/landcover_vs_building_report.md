# P11 Landcover vs Building First-Pose Gate Comparison

| Case | Veg | Road-like | Building density | XY RMSE | Yaw RMSE | Safe accept | Worse ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| building_firstpose_gate | None | None | None | 4.080530016881124 | 1.0206139999999948 | 0.0 | 1.0 |
| landcover_firstpose_gate | 0.39974283854166665 | 0.5777419704861111 | 0.01189208984375 | 4.0805290382144985 | 1.0206139999999948 | 0.0 | 1.0 |

## Delta
- selected XY RMSE delta landcover - building: `-9.786666250732878e-07` m
- selected yaw RMSE delta landcover - building: `0.0` deg
- raw refined XY RMSE delta landcover - building: `0.20766885571824378` m
- safe gate accept ratio delta: `0.0`

## Gate Rejection Reasons
- `building_firstpose_gate` visual={} pose={'pose_yaw_jump': 100} temporal={'temporal_xy': 396}
- `landcover_firstpose_gate` visual={'visual_chamfer_worse': 217, 'visual_overlap_worse': 16} pose={'pose_yaw_jump': 100} temporal={'temporal_xy': 396}
