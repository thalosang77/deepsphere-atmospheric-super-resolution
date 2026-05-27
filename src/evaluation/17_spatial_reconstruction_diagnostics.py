# ============================================================
# Spatial diagnostics: GT vs Bilinear vs DeepSphere
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

sample_idx = 0
pressure_idx = 20   # try 0, 10, 20, 30, 36 later

n_lat_hr = spatial_dict_q["n_lat_hr"]
n_lon_hr = spatial_dict_q["n_lon_hr"]

gt = direct_targets_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
pred = direct_preds_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()
bil = baseline_preds_q[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr).numpy()

err_model = pred - gt
err_bilinear = bil - gt
abs_err_model = np.abs(err_model)
abs_err_bilinear = np.abs(err_bilinear)
improvement_map = abs_err_bilinear - abs_err_model

vmin = min(gt.min(), pred.min(), bil.min())
vmax = max(gt.max(), pred.max(), bil.max())
err_lim = max(np.abs(err_model).max(), np.abs(err_bilinear).max())

fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)

plots = [
    (gt, "Ground Truth HR", vmin, vmax),
    (bil, "Bilinear", vmin, vmax),
    (pred, "DeepSphere", vmin, vmax),
    (err_bilinear, "Bilinear Error", -err_lim, err_lim),
    (err_model, "DeepSphere Error", -err_lim, err_lim),
    (improvement_map, "Error Reduction: Bilinear - DeepSphere", None, None),
    (abs_err_bilinear, "|Bilinear Error|", None, None),
    (abs_err_model, "|DeepSphere Error|", None, None),
    (abs_err_bilinear - abs_err_model, "Positive = DeepSphere Better", None, None),
]

for ax, (data, title, lo, hi) in zip(axes.flat, plots):
    if lo is None:
        im = ax.imshow(data)
    else:
        im = ax.imshow(data, vmin=lo, vmax=hi)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"Spatial Diagnostics | sample={sample_idx}, pressure_channel={pressure_idx}", fontsize=16)
plt.show()

print("CELL 16 PASSED")
print("Mean |error| bilinear  :", abs_err_bilinear.mean())
print("Mean |error| DeepSphere:", abs_err_model.mean())
print("Mean improvement       :", improvement_map.mean())
