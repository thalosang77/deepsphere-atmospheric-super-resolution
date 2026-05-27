# ============================================================
# Longitude-shift consistency test
# Tests robustness under cyclic longitude shift
# ============================================================

import torch
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("CELL 40 — LONGITUDE-SHIFT CONSISTENCY TEST")
print("=" * 70)

# ------------------------------------------------------------
# Pick available DeepSphere model
# ------------------------------------------------------------
if "temporal_model_q" in globals():
    ds_model_for_shift = temporal_model_q
    ds_model_name = "DeepSphere_temporal"
elif "deepsphere_model_q" in globals():
    ds_model_for_shift = deepsphere_model_q
    ds_model_name = "DeepSphere"
elif "test_model_q" in globals():
    ds_model_for_shift = test_model_q
    ds_model_name = "DeepSphere_direct"
else:
    raise NameError(
        "No DeepSphere model found. Expected temporal_model_q, "
        "deepsphere_model_q, or test_model_q."
    )

assert "cnn_model_q" in globals(), "Missing cnn_model_q. Run CELL 29 first."

print("DeepSphere model:", ds_model_name)

# ------------------------------------------------------------
# Helper: cyclic longitude roll
# ------------------------------------------------------------
def roll_vertices_grid(v, n_lat, n_lon, channels, shift_lon):
    """
    v: [B, N, C]
    Cyclically shifts longitude axis.
    """
    B, N, C = v.shape

    assert C == channels
    assert N == n_lat * n_lon

    grid = v.reshape(B, n_lat, n_lon, channels)
    grid_shifted = torch.roll(grid, shifts=shift_lon, dims=2)

    return grid_shifted.reshape(B, N, channels)


# ------------------------------------------------------------
# DeepSphere forward helper
# ------------------------------------------------------------
def forward_deepsphere_shift(model, x, lon_mr, lat_mr, lon_hr, lat_hr):
    """
    Handles models with or without flow-conditioning.
    """
    if getattr(model, "use_flow_blocks", False):
        return model(
            x_low=x,
            lon_low=lon_mr,
            lat_low=lat_mr,
            lon_high=lon_hr,
            lat_high=lat_hr,
            cond_low=x
        )
    else:
        return model(
            x_low=x,
            lon_low=lon_mr,
            lat_low=lat_mr,
            lon_high=lon_hr,
            lat_high=lat_hr
        )


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate_shift_consistency(
    ds_model,
    cnn_model,
    loader,
    shifts,
    spatial_dict,
    lon_mr,
    lat_mr,
    lon_hr,
    lat_hr,
    device,
    max_batches=None
):
    rows = []

    ds_model.eval()
    cnn_model.eval()

    n_lat_mr = spatial_dict["n_lat_mr"]
    n_lon_mr = spatial_dict["n_lon_mr"]
    n_lat_hr = spatial_dict["n_lat_hr"]
    n_lon_hr = spatial_dict["n_lon_hr"]

    C = in_channels

    with torch.no_grad():

        for shift in shifts:

            ds_errs = []
            cnn_errs = []

            for batch_idx, batch in enumerate(loader):

                if max_batches is not None and batch_idx >= max_batches:
                    break

                hr_v, mr_v, _, _ = batch

                mr_v = mr_v.to(device)

                # ----------------------------------------------------
                # Original predictions
                # ----------------------------------------------------
                ds_orig = forward_deepsphere_shift(
                    ds_model,
                    mr_v,
                    lon_mr,
                    lat_mr,
                    lon_hr,
                    lat_hr
                )

                cnn_orig = cnn_model(
                    mr_v
                )

                # ----------------------------------------------------
                # Shift MR input
                # ----------------------------------------------------
                mr_shift = roll_vertices_grid(
                    mr_v,
                    n_lat_mr,
                    n_lon_mr,
                    C,
                    shift
                )

                # ----------------------------------------------------
                # Expected shifted HR output
                # scale factor = 2, so HR shift = 2 * MR shift
                # ----------------------------------------------------
                ds_expected = roll_vertices_grid(
                    ds_orig,
                    n_lat_hr,
                    n_lon_hr,
                    C,
                    shift * 2
                )

                cnn_expected = roll_vertices_grid(
                    cnn_orig,
                    n_lat_hr,
                    n_lon_hr,
                    C,
                    shift * 2
                )

                # ----------------------------------------------------
                # Predictions from shifted input
                # ----------------------------------------------------
                ds_shift_pred = forward_deepsphere_shift(
                    ds_model,
                    mr_shift,
                    lon_mr,
                    lat_mr,
                    lon_hr,
                    lat_hr
                )

                cnn_shift_pred = cnn_model(
                    mr_shift
                )

                # ----------------------------------------------------
                # Consistency errors
                # ----------------------------------------------------
                ds_consistency = torch.sqrt(
                    torch.mean((ds_shift_pred - ds_expected) ** 2)
                )

                cnn_consistency = torch.sqrt(
                    torch.mean((cnn_shift_pred - cnn_expected) ** 2)
                )

                ds_errs.append(ds_consistency.detach().cpu())
                cnn_errs.append(cnn_consistency.detach().cpu())

            rows.append(
                {
                    "shift_mr_pixels": shift,
                    "shift_hr_pixels": shift * 2,
                    "DeepSphere_consistency_RMSE": torch.stack(ds_errs).mean().item(),
                    "CNN_consistency_RMSE": torch.stack(cnn_errs).mean().item(),
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Run test
# ------------------------------------------------------------
shift_df = evaluate_shift_consistency(
    ds_model=ds_model_for_shift,
    cnn_model=cnn_model_q,
    loader=val_loader_q_direct,
    shifts=[1, 2, 4, 8, 16, 24, 32],
    spatial_dict=spatial_dict_q,
    lon_mr=lon_mr_q,
    lat_mr=lat_mr_q,
    lon_hr=lon_hr_q,
    lat_hr=lat_hr_q,
    device=device,

    # set to None for full validation set
    max_batches=None
)

display(shift_df)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    shift_df["shift_mr_pixels"],
    shift_df["DeepSphere_consistency_RMSE"],
    marker="o",
    label=ds_model_name
)

plt.plot(
    shift_df["shift_mr_pixels"],
    shift_df["CNN_consistency_RMSE"],
    marker="o",
    label="CNN"
)

plt.xlabel("Longitude shift at MR grid pixels")
plt.ylabel("Consistency RMSE")
plt.title("Longitude-shift Consistency Test")

plt.grid(True, alpha=0.4)
plt.legend()
plt.show()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
shift_path = ARTIFACT_DIR / "longitude_shift_consistency.csv"

shift_df.to_csv(
    shift_path,
    index=False
)

print("Saved:", shift_path)
print("CELL 40 PASSED")
