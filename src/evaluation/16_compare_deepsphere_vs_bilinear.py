# ============================================================
# Compare DeepSphere bilinear-residual vs Bilinear baseline
# ============================================================

print("=" * 70)
print("DIRECT MR -> HR COMPARISON")
print("=" * 70)

comparison = {
    "RMSE":  (direct_metrics_q["RMSE"],  baseline_metrics_q["RMSE"]),
    "MAE":   (direct_metrics_q["MAE"],   baseline_metrics_q["MAE"]),
    "Corr":  (direct_metrics_q["Corr"],  baseline_metrics_q["Corr"]),
    "PSNR":  (direct_metrics_q["PSNR"],  baseline_metrics_q["PSNR"]),
    "R2":    (direct_metrics_q["R2"],    baseline_metrics_q["R2"]),
}

for metric_name, (model_val, base_val) in comparison.items():
    if metric_name in ["RMSE", "MAE"]:
        improvement = 100 * (base_val - model_val) / base_val
        better = "DeepSphere" if model_val < base_val else "Bilinear"
    else:
        improvement = 100 * (model_val - base_val) / abs(base_val)
        better = "DeepSphere" if model_val > base_val else "Bilinear"

    print(
        f"{metric_name:>6} | "
        f"DeepSphere: {model_val:.6f} | "
        f"Bilinear: {base_val:.6f} | "
        f"Improvement: {improvement:.2f}% | "
        f"Better: {better}"
    )
