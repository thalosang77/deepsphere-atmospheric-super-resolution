# ============================================================
# Spectral band-wise error
# Compares model error energy in low / mid / high wavenumber bands
# ============================================================

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("CELL 45 — SPECTRAL BAND-WISE ERROR")
print("=" * 70)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
required = [
    "baseline_preds_q",
    "final_deepsphere_preds_q",
    "final_deepsphere_targets_q",
    "cnn_preds_q",
]

for name in required:
    assert name in globals(), f"Missing {name}. Run previous evaluation cells first."

# ------------------------------------------------------------
# FFT band-pass error
# ------------------------------------------------------------
def fft_bandpass_error(pred_2d, gt_2d, k_low, k_high):
    """
    RMSE of band-passed error field.
    """

    err = pred_2d - gt_2d

    ny, nx = err.shape

    fy = np.fft.fftfreq(ny) * ny
    fx = np.fft.fftfreq(nx) * nx

    FX, FY = np.meshgrid(fx, fy)
    KR = np.sqrt(FX**2 + FY**2)

    mask = (KR >= k_low) & (KR < k_high)

    E = np.fft.fft2(err)

    E_filtered = np.zeros_like(E)
    E_filtered[mask] = E[mask]

    err_filtered = np.fft.ifft2(E_filtered).real

    return float(np.sqrt(np.mean(err_filtered ** 2)))


# ------------------------------------------------------------
# Band-wise model errors
# ------------------------------------------------------------
def spectral_bandwise_errors(
    preds_dict,
    target,
    n_lat,
    n_lon,
    channel_idx=20,
    max_samples=30,
    bands=None
):

    if bands is None:
        bands = {
            "large_low_k": (1, 8),
            "medium_mid_k": (8, 32),
            "small_high_k": (32, 80),
        }

    rows = []

    B = min(target.shape[0], max_samples)

    for model_name, pred in preds_dict.items():

        assert pred.shape == target.shape, (
            f"Shape mismatch for {model_name}: "
            f"{pred.shape} vs {target.shape}"
        )

        for band_name, (k0, k1) in bands.items():

            vals = []

            for i in range(B):

                pred_2d = (
                    pred[i, :, channel_idx]
                    .reshape(n_lat, n_lon)
                    .detach()
                    .cpu()
                    .numpy()
                )

                gt_2d = (
                    target[i, :, channel_idx]
                    .reshape(n_lat, n_lon)
                    .detach()
                    .cpu()
                    .numpy()
                )

                vals.append(
                    fft_bandpass_error(
                        pred_2d,
                        gt_2d,
                        k0,
                        k1
                    )
                )

            rows.append(
                {
                    "Model": model_name,
                    "Band": band_name,
                    "k_min": k0,
                    "k_max": k1,
                    "Band_RMSE": float(np.mean(vals)),
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------
preds_for_band = {
    "Bilinear": baseline_preds_q.float(),
    "DeepSphere": final_deepsphere_preds_q.float(),
    "CNN": cnn_preds_q.float(),
}

bands = {
    "large_low_k": (1, 8),
    "medium_mid_k": (8, 32),
    "small_high_k": (32, 80),
}

band_order = list(bands.keys())

band_df = spectral_bandwise_errors(
    preds_dict=preds_for_band,
    target=final_deepsphere_targets_q.float(),
    n_lat=spatial_dict_q["n_lat_hr"],
    n_lon=spatial_dict_q["n_lon_hr"],
    channel_idx=20,
    max_samples=30,
    bands=bands
)

band_df["Band"] = pd.Categorical(
    band_df["Band"],
    categories=band_order,
    ordered=True
)

band_df = band_df.sort_values(["Band", "Model"]).reset_index(drop=True)

display(band_df)

# ------------------------------------------------------------
# Pivot table
# ------------------------------------------------------------
band_pivot = band_df.pivot(
    index="Band",
    columns="Model",
    values="Band_RMSE"
).loc[band_order]

display(band_pivot)

# ------------------------------------------------------------
# Summary table
# ------------------------------------------------------------
summary_rows = []

eps = 1e-12

for band in band_pivot.index:

    ds = band_pivot.loc[band, "DeepSphere"]
    cnn = band_pivot.loc[band, "CNN"]
    bil = band_pivot.loc[band, "Bilinear"]

    summary_rows.append(
        {
            "Band": band,
            "Bilinear_RMSE": bil,
            "DeepSphere_RMSE": ds,
            "CNN_RMSE": cnn,
            "DS_vs_Bilinear_improvement_%": 100.0 * (bil - ds) / max(bil, eps),
            "DS_vs_CNN_improvement_%": 100.0 * (cnn - ds) / max(cnn, eps),
            "Winner": "DeepSphere" if ds < cnn else "CNN",
        }
    )

band_summary_df = pd.DataFrame(summary_rows)

display(band_summary_df)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

x = np.arange(len(band_pivot.index))
width = 0.25

plt.bar(
    x - width,
    band_pivot["Bilinear"],
    width,
    label="Bilinear"
)

plt.bar(
    x,
    band_pivot["DeepSphere"],
    width,
    label="DeepSphere"
)

plt.bar(
    x + width,
    band_pivot["CNN"],
    width,
    label="CNN"
)

plt.xticks(
    x,
    band_pivot.index,
    rotation=15
)

plt.ylabel("Band-passed error RMSE")
plt.title("Spectral Band-wise Error")

plt.grid(True, axis="y", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
band_path = ARTIFACT_DIR / "spectral_bandwise_error_comparison.csv"
band_summary_path = ARTIFACT_DIR / "spectral_bandwise_error_summary.csv"

band_df.to_csv(band_path, index=False)
band_summary_df.to_csv(band_summary_path, index=False)

print("Saved:", band_path)
print("Saved:", band_summary_path)
print("CELL 45 PASSED")
