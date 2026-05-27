# ============================================================
# Multi-region transfer bar plot
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

df = region_results_df.copy()

x = np.arange(len(df))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, df["Bilinear_RMSE"], width, label="Bilinear")
plt.bar(x, df["DeepSphere_RMSE"], width, label="DeepSphere")
plt.bar(x + width, df["CNN_RMSE"], width, label="CNN")

plt.xticks(x, df["region"])
plt.ylabel("RMSE")
plt.title("Cross-region Transfer Performance")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(df["region"], df["DeepSphere_improvement_%"], label="DeepSphere vs Bilinear")
plt.bar(df["region"], df["CNN_improvement_%"], alpha=0.7, label="CNN vs Bilinear")
plt.ylabel("RMSE improvement over bilinear (%)")
plt.title("Cross-region Improvement over Bilinear")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("CELL 39 PASSED")
