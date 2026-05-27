# ============================================================
# Seam artifact quantification
# Compares error near left/right longitude boundaries vs interior
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import torch

def seam_error_analysis(pred, target, n_lat, n_lon, channels, seam_width=8):
    """
    pred/target: [B, N, C]
    seam_width: number of HR pixels near left/right longitude boundary.
    """

    B, N, C = pred.shape

    assert target.shape == pred.shape
    assert N == n_lat * n_lon
    assert C == channels

    pred_g = pred.reshape(B, n_lat, n_lon, channels)
    targ_g = target.reshape(B, n_lat, n_lon, channels)

    abs_err = torch.abs(pred_g - targ_g)

    seam_mask = torch.zeros(
        n_lat,
        n_lon,
        dtype=torch.bool,
        device=pred.device
    )

    seam_mask[:, :seam_width] = True
    seam_mask[:, -seam_width:] = True

    interior_mask = ~seam_mask

    mae_seam = abs_err[:, seam_mask, :].mean().item()
    mae_interior = abs_err[:, interior_mask, :].mean().item()

    rmse_seam = torch.sqrt(
        torch.mean(
            (pred_g[:, seam_mask, :] - targ_g[:, seam_mask, :]) ** 2
        )
    ).item()

    rmse_interior = torch.sqrt(
        torch.mean(
            (pred_g[:, interior_mask, :] - targ_g[:, interior_mask, :]) ** 2
        )
    ).item()

    return {
        "MAE_seam": mae_seam,
        "MAE_interior": mae_interior,
        "MAE_seam_to_interior_ratio": mae_seam / mae_interior,
        "RMSE_seam": rmse_seam,
        "RMSE_interior": rmse_interior,
        "RMSE_seam_to_interior_ratio": rmse_seam / rmse_interior,
    }


# ------------------------------------------------------------
# Models to compare
# ------------------------------------------------------------
seam_models = [
    (
        "Bilinear",
        baseline_preds_q.float(),
        baseline_targets_q.float()
    ),
    (
        "DeepSphere",
        final_deepsphere_preds_q.float(),
        final_deepsphere_targets_q.float()
    ),
]

if "cnn_preds_q" in globals() and "cnn_targets_q" in globals():
    seam_models.append(
        (
            "CNN",
            cnn_preds_q.float(),
            cnn_targets_q.float()
        )
    )

# ------------------------------------------------------------
# Run seam analysis
# ------------------------------------------------------------
seam_rows = []

for name, pred, target in seam_models:

    out = seam_error_analysis(
        pred=pred,
        target=target,
        n_lat=spatial_dict_q["n_lat_hr"],
        n_lon=spatial_dict_q["n_lon_hr"],
        channels=in_channels,
        seam_width=8
    )

    out["Model"] = name
    seam_rows.append(out)

seam_df = pd.DataFrame(seam_rows)

display(seam_df)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.bar(
    seam_df["Model"],
    seam_df["RMSE_seam_to_interior_ratio"]
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1
)

plt.ylabel("Seam / interior RMSE ratio")
plt.title("Boundary / Seam Artifact Ratio")

plt.grid(True, axis="y", alpha=0.4)
plt.show()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
seam_path = ARTIFACT_DIR / "seam_artifact_analysis.csv"

seam_df.to_csv(
    seam_path,
    index=False
)

print("Saved:", seam_path)
print("CELL 41 PASSED")
