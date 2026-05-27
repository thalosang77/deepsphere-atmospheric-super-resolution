# ============================================================
# Per-pressure-level evaluation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

pressure_levels = plev_1d  # from Cell 4, shape [37]

model_err = direct_preds_q - direct_targets_q
bil_err = baseline_preds_q - baseline_targets_q

rmse_model_by_level = torch.sqrt(torch.mean(model_err ** 2, dim=(0, 1))).numpy()
rmse_bil_by_level = torch.sqrt(torch.mean(bil_err ** 2, dim=(0, 1))).numpy()

mae_model_by_level = torch.mean(torch.abs(model_err), dim=(0, 1)).numpy()
mae_bil_by_level = torch.mean(torch.abs(bil_err), dim=(0, 1)).numpy()

rmse_improvement_pct = 100 * (rmse_bil_by_level - rmse_model_by_level) / rmse_bil_by_level
mae_improvement_pct = 100 * (mae_bil_by_level - mae_model_by_level) / mae_bil_by_level

plt.figure(figsize=(8, 8))
plt.plot(rmse_bil_by_level, pressure_levels, marker="o", label="Bilinear RMSE")
plt.plot(rmse_model_by_level, pressure_levels, marker="o", label="DeepSphere RMSE")
plt.gca().invert_yaxis()
plt.xlabel("RMSE")
plt.ylabel("Pressure level hPa")
plt.title("RMSE by Pressure Level")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(rmse_improvement_pct, pressure_levels, marker="o")
plt.axvline(0, linestyle="--")
plt.gca().invert_yaxis()
plt.xlabel("RMSE improvement over bilinear (%)")
plt.ylabel("Pressure level hPa")
plt.title("DeepSphere Improvement by Pressure Level")
plt.grid(True, alpha=0.4)
plt.show()

print("CELL 17 PASSED")
print("Mean RMSE improvement %:", rmse_improvement_pct.mean())
print("Best level improvement :", pressure_levels[np.argmax(rmse_improvement_pct)], rmse_improvement_pct.max())
print("Worst level improvement:", pressure_levels[np.argmin(rmse_improvement_pct)], rmse_improvement_pct.min())
