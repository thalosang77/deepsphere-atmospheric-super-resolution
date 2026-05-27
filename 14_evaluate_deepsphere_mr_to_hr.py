# ============================================================
# Evaluate trained direct MR -> HR DeepSphere model
# Bilinear-residual version
# ============================================================

direct_metrics_q, direct_preds_q, direct_targets_q = evaluate_model_mr_to_hr(
    model=test_model_q,
    val_loader=val_loader_q_direct,
    val_metric=val_metric_q_direct,
    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,
    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,
    device=device
)

print("\nDIRECT MR -> HR DEEPSPHERE (BILINEAR-RESIDUAL) METRICS")
print("=" * 60)

for key, value in direct_metrics_q.items():
    print(f"{key:>6}: {value:.6f}")

# extra sanity checks (important for paper-quality)
print("\nSanity checks:")
print("Prediction range:", direct_preds_q.min().item(), direct_preds_q.max().item())
print("Target range    :", direct_targets_q.min().item(), direct_targets_q.max().item())

print("\nCELL 13 PASSED")
