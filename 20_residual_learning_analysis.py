# ============================================================
# Residual analysis: what does DeepSphere learn beyond bilinear?
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

n_lat_hr = spatial_dict_q["n_lat_hr"]
n_lon_hr = spatial_dict_q["n_lon_hr"]

sample_idx = 0
pressure_idx = 20

true_residual = direct_targets_q - baseline_preds_q
learned_residual = direct_preds_q - baseline_preds_q
residual_error = learned_residual - true_residual

true_res_map = true_residual[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
learned_res_map = learned_residual[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
res_err_map = residual_error[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()

lim = max(
    np.abs(true_res_map).max(),
    np.abs(learned_res_map).max(),
    np.abs(res_err_map).max()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

plots = [
    (true_res_map, "True Residual: GT - Bilinear"),
    (learned_res_map, "Learned Residual: DeepSphere - Bilinear"),
    (res_err_map, "Residual Error"),
]

for ax, (data, title) in zip(axes, plots):
    im = ax.imshow(data, vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"Residual Diagnostics | sample={sample_idx}, pressure_channel={pressure_idx}")
plt.show()

res_corr = torch.corrcoef(torch.stack([
    true_residual.reshape(-1),
    learned_residual.reshape(-1)
]))[0, 1].item()

res_rmse = torch.sqrt(torch.mean((learned_residual - true_residual) ** 2)).item()

print("CELL 19 PASSED")
print("Residual correlation:", res_corr)
print("Residual RMSE       :", res_rmse)
