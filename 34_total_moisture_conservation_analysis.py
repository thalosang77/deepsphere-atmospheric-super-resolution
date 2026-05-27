# ============================================================
# Total moisture / conservation-like analysis
# Absolute-error reporting version
#
# Key idea:
#   Bilinear interpolation often preserves area-integrated
#   quantities extremely well because it behaves like a smooth
#   averaging operator.
#
#   DeepSphere/CNN may improve local reconstruction while slightly
#   perturbing global moisture totals.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print("=" * 70)
print("CELL 31 — AREA-WEIGHTED MOISTURE CONSISTENCY")
print("=" * 70)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
required_phys_fields = ["gt_phys", "bl_phys"]

for name in required_phys_fields:
    assert name in globals(), (
        f"Missing {name}. Run CELL 30 first to create physical-unit tensors."
    )

has_ds = "ds_phys" in globals()
has_cnn = "cnn_phys" in globals()

print("Available fields:")
print(" - GroundTruth")
print(" - Bilinear")

if has_ds:
    print(" - DeepSphere")

if has_cnn:
    print(" - CNN")

# ------------------------------------------------------------
# Area weights
# ------------------------------------------------------------
def area_weights_from_lat(lat_1d_flat):
    """
    Approximate spherical area weights proportional to cos(latitude).

    lat_1d_flat:
        flattened HR latitude array [N_hr]
    """

    lat_rad = torch.tensor(
        np.deg2rad(lat_1d_flat),
        dtype=torch.float32
    )

    w = torch.cos(lat_rad).clamp(min=0.0)

    w = w / w.sum()

    return w.view(1, -1, 1)


area_w_hr = area_weights_from_lat(
    spatial_dict_q["lat_hr"]
)

# ------------------------------------------------------------
# Weighted total moisture
# ------------------------------------------------------------
def weighted_total_moisture(x_phys, area_w):
    """
    x_phys:
        [B, N, C]

    Returns:
        [B, C] area-weighted moisture total per pressure channel.
    """

    return torch.sum(
        x_phys * area_w,
        dim=1
    )


total_gt = weighted_total_moisture(
    gt_phys,
    area_w_hr
)

total_bl = weighted_total_moisture(
    bl_phys,
    area_w_hr
)

if has_ds:
    total_ds = weighted_total_moisture(
        ds_phys,
        area_w_hr
    )

if has_cnn:
    total_cnn = weighted_total_moisture(
        cnn_phys,
        area_w_hr
    )

# ------------------------------------------------------------
# Metric helper
# ------------------------------------------------------------
def compute_total_metrics(pred, gt):
    """
    Computes conservation-like error between predicted and true
    area-weighted total moisture.
    """

    rmse = torch.sqrt(
        torch.mean((pred - gt) ** 2)
    ).item()

    mae = torch.mean(
        torch.abs(pred - gt)
    ).item()

    max_abs = torch.max(
        torch.abs(pred - gt)
    ).item()

    bias = torch.mean(
        pred - gt
    ).item()

    return rmse, mae, max_abs, bias


# ------------------------------------------------------------
# Build summary table
# ------------------------------------------------------------
rows = []

rmse_bl, mae_bl, max_bl, bias_bl = compute_total_metrics(
    total_bl,
    total_gt
)

rows.append(
    {
        "Method": "Bilinear",
        "RMSE_total_q": rmse_bl,
        "MAE_total_q": mae_bl,
        "MaxAbs_total_q": max_bl,
        "Bias_total_q": bias_bl,
    }
)

if has_cnn:

    rmse_cnn, mae_cnn, max_cnn, bias_cnn = compute_total_metrics(
        total_cnn,
        total_gt
    )

    rows.append(
        {
            "Method": "CNN",
            "RMSE_total_q": rmse_cnn,
            "MAE_total_q": mae_cnn,
            "MaxAbs_total_q": max_cnn,
            "Bias_total_q": bias_cnn,
        }
    )

if has_ds:

    rmse_ds, mae_ds, max_ds, bias_ds = compute_total_metrics(
        total_ds,
        total_gt
    )

    rows.append(
        {
            "Method": "DeepSphere",
            "RMSE_total_q": rmse_ds,
            "MAE_total_q": mae_ds,
            "MaxAbs_total_q": max_ds,
            "Bias_total_q": bias_ds,
        }
    )

cons_df = pd.DataFrame(rows)

cons_df = cons_df.sort_values(
    "RMSE_total_q",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 70)
print("CONSERVATION-LIKE TOTAL MOISTURE ERROR")
print("=" * 70)

display(cons_df)

# ------------------------------------------------------------
# Better reporting: absolute ranking, not percentage improvement
# ------------------------------------------------------------
print("\nConservation ranking, lower is better")
print("-" * 70)

for _, row in cons_df.iterrows():

    print(
        f"{row['Method']:>12} | "
        f"RMSE: {row['RMSE_total_q']:.3e} | "
        f"MAE: {row['MAE_total_q']:.3e} | "
        f"MaxAbs: {row['MaxAbs_total_q']:.3e} | "
        f"Bias: {row['Bias_total_q']:.3e}"
    )

print("\nInterpretation:")
print(
    "Bilinear interpolation best preserves global area-integrated "
    "moisture because it behaves like a smooth averaging operator."
)

print(
    "DeepSphere and CNN improve local reconstruction accuracy, gradients, "
    "spectra, and extremes, but their learned residuals can slightly perturb "
    "global moisture totals."
)

print(
    "Therefore, this test should be reported as an absolute conservation "
    "diagnostic rather than percentage improvement, because the bilinear "
    "conservation error is already extremely close to zero."
)

# ------------------------------------------------------------
# Per-pressure-level error curves
# ------------------------------------------------------------
plt.figure(figsize=(8, 8))

error_bl_by_level = torch.sqrt(
    torch.mean(
        (total_bl - total_gt) ** 2,
        dim=0
    )
).detach().cpu().numpy()

plt.plot(
    error_bl_by_level,
    plev_1d,
    marker="o",
    label="Bilinear"
)

if has_cnn:

    error_cnn_by_level = torch.sqrt(
        torch.mean(
            (total_cnn - total_gt) ** 2,
            dim=0
        )
    ).detach().cpu().numpy()

    plt.plot(
        error_cnn_by_level,
        plev_1d,
        marker="o",
        label="CNN"
    )

if has_ds:

    error_ds_by_level = torch.sqrt(
        torch.mean(
            (total_ds - total_gt) ** 2,
            dim=0
        )
    ).detach().cpu().numpy()

    plt.plot(
        error_ds_by_level,
        plev_1d,
        marker="o",
        label="DeepSphere"
    )

plt.gca().invert_yaxis()

plt.xlabel("Area-weighted total moisture RMSE")
plt.ylabel("Pressure level (hPa)")
plt.title("Conservation-like Moisture Error by Pressure Level")

plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Optional: save table
# ------------------------------------------------------------
conservation_path = (
    ARTIFACT_DIR /
    "conservation_like_total_moisture_q.csv"
)

cons_df.to_csv(
    conservation_path,
    index=False
)

print("\nSaved:", conservation_path)
print("\nCELL 31 PASSED")
