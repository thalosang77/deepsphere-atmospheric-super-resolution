# ============================================================
# Scale-dependent RMSE using Fourier filtering
# ============================================================

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fft_band_filter_batch(x, n_lat, n_lon, band):
    """
    x: [B, N, C]
    band:
      'low'  = large scales
      'mid'  = intermediate scales
      'high' = small scales
    """
    B, N, C = x.shape
    grid = x.reshape(B, n_lat, n_lon, C).permute(0, 3, 1, 2).float()  # [B,C,Y,X]

    fft = torch.fft.fft2(grid)
    fy = torch.fft.fftfreq(n_lat, device=grid.device).reshape(-1, 1)
    fx = torch.fft.fftfreq(n_lon, device=grid.device).reshape(1, -1)
    kr = torch.sqrt(fx**2 + fy**2)

    if band == "low":
        mask = kr <= 0.08
    elif band == "mid":
        mask = (kr > 0.08) & (kr <= 0.20)
    elif band == "high":
        mask = kr > 0.20
    else:
        raise ValueError("band must be low, mid, or high")

    filtered = torch.fft.ifft2(fft * mask).real
    filtered = filtered.permute(0, 2, 3, 1).reshape(B, N, C)
    return filtered

scale_rows = []

for band in ["low", "mid", "high"]:
    gt_band = fft_band_filter_batch(direct_targets_q, n_lat_hr, n_lon_hr, band)
    ds_band = fft_band_filter_batch(direct_preds_q, n_lat_hr, n_lon_hr, band)
    bl_band = fft_band_filter_batch(baseline_preds_q, n_lat_hr, n_lon_hr, band)

    rmse_ds = torch.sqrt(torch.mean((ds_band - gt_band) ** 2)).item()
    rmse_bl = torch.sqrt(torch.mean((bl_band - gt_band) ** 2)).item()
    improvement = 100 * (rmse_bl - rmse_ds) / rmse_bl

    scale_rows.append({
        "Scale": band,
        "DeepSphere_RMSE": rmse_ds,
        "Bilinear_RMSE": rmse_bl,
        "Improvement_%": improvement
    })

scale_df = pd.DataFrame(scale_rows)
display(scale_df)

plt.figure(figsize=(8, 5))
plt.plot(scale_df["Scale"], scale_df["Bilinear_RMSE"], marker="o", label="Bilinear")
plt.plot(scale_df["Scale"], scale_df["DeepSphere_RMSE"], marker="o", label="DeepSphere")
plt.ylabel("Band-filtered RMSE")
plt.title("Scale-dependent RMSE")
plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

print("CELL 25 PASSED")
