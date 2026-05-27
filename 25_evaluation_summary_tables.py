# ============================================================
# Paper-quality summary table
# ============================================================

summary_rows = []

def add_metric(name, model_val, base_val, lower_is_better=True):
    if lower_is_better:
        improvement = 100 * (base_val - model_val) / base_val
        winner = "DeepSphere" if model_val < base_val else "Bilinear"
    else:
        improvement = 100 * (model_val - base_val) / abs(base_val)
        winner = "DeepSphere" if model_val > base_val else "Bilinear"

    summary_rows.append({
        "Metric": name,
        "DeepSphere": model_val,
        "Bilinear": base_val,
        "Improvement_%": improvement,
        "Winner": winner
    })

add_metric("RMSE_normalized", direct_metrics_q["RMSE"], baseline_metrics_q["RMSE"], True)
add_metric("MAE_normalized", direct_metrics_q["MAE"], baseline_metrics_q["MAE"], True)
add_metric("Corr", direct_metrics_q["Corr"], baseline_metrics_q["Corr"], False)
add_metric("PSNR", direct_metrics_q["PSNR"], baseline_metrics_q["PSNR"], False)
add_metric("R2", direct_metrics_q["R2"], baseline_metrics_q["R2"], False)
add_metric("Gradient_RMSE", grad_rmse_model.item(), grad_rmse_bil.item(), True)
add_metric("RMSE_physical_q", model_rmse_phys, bil_rmse_phys, True)
add_metric("MAE_physical_q", model_mae_phys, bil_mae_phys, True)

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

summary_path = ARTIFACT_DIR / "q_regular3h_deepsphere_vs_bilinear_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("Saved summary table to:", summary_path)
print("CELL 24 PASSED")
