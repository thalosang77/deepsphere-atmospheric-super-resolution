# ============================================================
# Denormalized physical-unit evaluation for q
# ============================================================

mean = norm_params_q["mean"].view(1, 1, -1)
std = norm_params_q["std"].view(1, 1, -1)

direct_preds_phys = direct_preds_q * std + mean
direct_targets_phys = direct_targets_q * std + mean
baseline_preds_phys = baseline_preds_q * std + mean
baseline_targets_phys = baseline_targets_q * std + mean

model_rmse_phys = torch.sqrt(torch.mean((direct_preds_phys - direct_targets_phys) ** 2)).item()
bil_rmse_phys = torch.sqrt(torch.mean((baseline_preds_phys - baseline_targets_phys) ** 2)).item()

model_mae_phys = torch.mean(torch.abs(direct_preds_phys - direct_targets_phys)).item()
bil_mae_phys = torch.mean(torch.abs(baseline_preds_phys - baseline_targets_phys)).item()

print("=" * 70)
print("PHYSICAL-UNIT q METRICS")
print("=" * 70)
print(f"RMSE DeepSphere q units : {model_rmse_phys:.8e}")
print(f"RMSE Bilinear q units   : {bil_rmse_phys:.8e}")
print(f"RMSE improvement        : {100 * (bil_rmse_phys - model_rmse_phys) / bil_rmse_phys:.2f}%")
print(f"MAE DeepSphere q units  : {model_mae_phys:.8e}")
print(f"MAE Bilinear q units    : {bil_mae_phys:.8e}")
print(f"MAE improvement         : {100 * (bil_mae_phys - model_mae_phys) / bil_mae_phys:.2f}%")

# Per-level physical RMSE
phys_model_err = direct_preds_phys - direct_targets_phys
phys_bil_err = baseline_preds_phys - baseline_targets_phys

rmse_model_phys_level = torch.sqrt(torch.mean(phys_model_err ** 2, dim=(0, 1))).numpy()
rmse_bil_phys_level = torch.sqrt(torch.mean(phys_bil_err ** 2, dim=(0, 1))).numpy()

plt.figure(figsize=(8, 8))
plt.plot(rmse_bil_phys_level, plev_1d, marker="o", label="Bilinear")
plt.plot(rmse_model_phys_level, plev_1d, marker="o", label="DeepSphere")
plt.gca().invert_yaxis()
plt.xlabel("RMSE in specific humidity units")
plt.ylabel("Pressure level hPa")
plt.title("Physical-Unit RMSE by Pressure Level")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

print("CELL 23 PASSED")
