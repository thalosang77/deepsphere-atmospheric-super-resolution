# ============================================================
# Spectral slope comparison
# ============================================================

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("CELL 42 — SPECTRAL SLOPE COMPARISON")
print("=" * 70)

# ------------------------------------------------------------
# Radial power spectrum
# ------------------------------------------------------------
def radial_power_spectrum_2d(field_2d):
    """
    Computes isotropic radial power spectrum.
    """

    f = np.fft.fft2(field_2d)
    power = np.abs(f) ** 2

    ny, nx = field_2d.shape

    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx

    KX, KY = np.meshgrid(kx, ky)

    kr = np.sqrt(KX**2 + KY**2)
    kr_int = kr.astype(np.int32)

    max_k = kr_int.max()

    radial_power = np.bincount(
        kr_int.ravel(),
        weights=power.ravel(),
        minlength=max_k + 1
    )

    counts = np.bincount(
        kr_int.ravel(),
        minlength=max_k + 1
    )

    radial_power = radial_power / np.maximum(counts, 1)

    k_bins = np.arange(max_k + 1)

    valid = (
        (counts > 0)
        & (k_bins > 1)
        & np.isfinite(radial_power)
        & (radial_power > 0)
    )

    return k_bins[valid], radial_power[valid]


# ------------------------------------------------------------
# Spectral slope fit
# ------------------------------------------------------------
def spectral_slope(k, p, kmin=8, kmax=60):

    mask = (
        (k >= kmin)
        & (k <= kmax)
        & np.isfinite(p)
        & (p > 0)
    )

    if mask.sum() < 5:
        return np.nan

    logk = np.log(k[mask])
    logp = np.log(p[mask])

    slope, _ = np.polyfit(logk, logp, 1)

    return float(slope)


# ------------------------------------------------------------
# Average spectral slope across samples
# ------------------------------------------------------------
def average_spectral_slope(
    pred,
    target,
    n_lat,
    n_lon,
    channel_idx=20,
    max_samples=20
):

    B = min(pred.shape[0], max_samples)

    slopes_pred = []
    slopes_gt = []

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

        k_pred, p_pred = radial_power_spectrum_2d(pred_2d)
        k_gt, p_gt = radial_power_spectrum_2d(gt_2d)

        slopes_pred.append(spectral_slope(k_pred, p_pred))
        slopes_gt.append(spectral_slope(k_gt, p_gt))

    return (
        float(np.nanmean(slopes_pred)),
        float(np.nanmean(slopes_gt))
    )


# ------------------------------------------------------------
# Compare models
# ------------------------------------------------------------
channel_idx = 20

models_for_slope = [
    ("GroundTruth", final_deepsphere_targets_q, final_deepsphere_targets_q),
    ("Bilinear", baseline_preds_q, baseline_targets_q),
    ("DeepSphere", final_deepsphere_preds_q, final_deepsphere_targets_q),
    ("CNN", cnn_preds_q, cnn_targets_q),
]

slope_rows = []

gt_slope_ref, _ = average_spectral_slope(
    final_deepsphere_targets_q,
    final_deepsphere_targets_q,
    n_lat=spatial_dict_q["n_lat_hr"],
    n_lon=spatial_dict_q["n_lon_hr"],
    channel_idx=channel_idx
)

for name, pred, target in models_for_slope:

    pred_slope, _ = average_spectral_slope(
        pred,
        target,
        n_lat=spatial_dict_q["n_lat_hr"],
        n_lon=spatial_dict_q["n_lon_hr"],
        channel_idx=channel_idx
    )

    slope_rows.append({
        "Model": name,
        "Spectral_slope": pred_slope,
        "GT_slope": gt_slope_ref,
        "Abs_slope_error": abs(pred_slope - gt_slope_ref)
    })

slope_df = (
    pd.DataFrame(slope_rows)
    .sort_values("Abs_slope_error")
    .reset_index(drop=True)
)

display(slope_df)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    slope_df["Model"],
    slope_df["Abs_slope_error"]
)

plt.ylabel("|model spectral slope − GT slope|")
plt.title(f"Spectral slope error | channel={channel_idx}")

plt.grid(True, axis="y", alpha=0.4)

plt.show()

# ------------------------------------------------------------
# Print ranking
# ------------------------------------------------------------
print("\nSpectral slope fidelity ranking")
print("-" * 70)

for _, row in slope_df.iterrows():
    print(
        f"{row['Model']:>12} | "
        f"Slope: {row['Spectral_slope']:.4f} | "
        f"GT: {row['GT_slope']:.4f} | "
        f"Abs error: {row['Abs_slope_error']:.4f}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
slope_path = ARTIFACT_DIR / "spectral_slope_comparison.csv"

slope_df.to_csv(slope_path, index=False)

print("\nSaved:", slope_path)
print("CELL 42 PASSED")
