# P11.4 Synthetic Sequence Ablation Check

## Answers
1. Old gate_v2 accuracy: XY RMSE `2.756603611314328` m, yaw RMSE `1.0235596297867737` deg.
2. temporal_pred_only accuracy: XY RMSE `4.080530031610323` m, yaw RMSE `1.0206139999999948` deg.
3. firstpose_odom init accuracy: XY RMSE `4.080530016881124` m, yaw RMSE `1.0206139999999948` deg.
4. firstpose_odom safe selected accuracy: XY RMSE `4.080530016881124` m, yaw RMSE `1.0206139999999948` deg.
5. firstpose safe vs old gate_v2 XY delta `1.3239264055667954` m; lower is better.
6. gate_v2_no_temporal vs init: `5.2003617136491105` m vs `3.861568477851895` m.
7. corrected_refined_only vs raw_refined_only: XY `5.200420081595403` vs `5.200420081595403`, yaw `4.102395829972316` vs `176.92723820355707`.

## Recommendation
Use first-pose+odom safe gate as the P11 sequence baseline; it improves or preserves the propagated prior without trusting feature refinement blindly.
