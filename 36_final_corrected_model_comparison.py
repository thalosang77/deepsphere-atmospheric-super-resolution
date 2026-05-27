# ============================================================
# Final corrected comparison table
# Uses conservation-corrected + clipped DeepSphere as final model
# ============================================================

import pandas as pd

final_rows = []

def add_final_row(name, metrics):
    final_rows.append({
        "Model": name,
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
        "MAPE": metrics["MAPE"],
        "Corr": metrics["Corr"],
        "PSNR": metrics["PSNR"],
        "R2": metrics["R2"],
    })

add_final_row("Bilinear", baseline_metrics_q)
add_final_row("CNN_residual_local", cnn_metrics_q)
add_final_row("DeepSphere_original", original_metrics_q)
add_final_row("DeepSphere_conservative", conservative_metrics_q)
add_final_row("DeepSphere_conservative_clipped", conservative_clipped_metrics_q)

final_corrected_df = pd.DataFrame(final_rows).sort_values(
    "RMSE",
    ascending=True
).reset_index(drop=True)

display(final_corrected_df)

final_corrected_path = (
    ARTIFACT_DIR /
    "final_corrected_model_comparison_q_regular3h.csv"
)

final_corrected_df.to_csv(
    final_corrected_path,
    index=False
)

print("Saved:", final_corrected_path)
print("CELL 34 PASSED")
