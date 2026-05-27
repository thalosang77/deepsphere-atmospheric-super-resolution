# ============================================================
# True spherical rotation consistency test
# 3D Cartesian rotation + nearest-neighbour spherical remap
# ============================================================

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

print("=" * 70)
print("CELL 43 — TRUE SPHERICAL ROTATION CONSISTENCY")
print("=" * 70)

# ------------------------------------------------------------
# Pick available DeepSphere model
# ------------------------------------------------------------
if "temporal_model_q" in globals():
    ds_model_for_rotation = temporal_model_q
    ds_model_name = "DeepSphere_temporal"
elif "deepsphere_model_q" in globals():
    ds_model_for_rotation = deepsphere_model_q
    ds_model_name = "DeepSphere"
elif "test_model_q" in globals():
    ds_model_for_rotation = test_model_q
    ds_model_name = "DeepSphere_direct"
else:
    raise NameError("No DeepSphere model found.")

assert "cnn_model_q" in globals(), "Missing cnn_model_q."

print("DeepSphere model:", ds_model_name)

# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
def lonlat_to_xyz_np(lon, lat):
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)

    return np.stack([x, y, z], axis=-1)


def rotation_matrix(axis="y", angle_deg=20):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)

    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    raise ValueError("axis must be x, y, or z")


def build_rotation_indices(lon, lat, axis="y", angle_deg=20):
    """
    Returns nearest-neighbour remap indices for rotating a field
    on the same spherical grid.

    output_grid_point samples input at R^{-1}(point).
    """
    xyz = lonlat_to_xyz_np(lon, lat)

    R_inv = rotation_matrix(axis=axis, angle_deg=-angle_deg)

    sample_xyz = xyz @ R_inv.T

    tree = cKDTree(xyz)
    _, idx = tree.query(sample_xyz, k=1)

    return idx.astype(np.int64)


def remap_vertices(v, indices):
    idx = torch.as_tensor(indices, dtype=torch.long, device=v.device)
    return v[:, idx, :]


# ------------------------------------------------------------
# Model forward helper
# ------------------------------------------------------------
def forward_deepsphere_rotation(model, x, lon_mr, lat_mr, lon_hr, lat_hr):
    if getattr(model, "use_flow_blocks", False):
        return model(
            x_low=x,
            lon_low=lon_mr,
            lat_low=lat_mr,
            lon_high=lon_hr,
            lat_high=lat_hr,
            cond_low=x
        )

    return model(
        x_low=x,
        lon_low=lon_mr,
        lat_low=lat_mr,
        lon_high=lon_hr,
        lat_high=lat_hr
    )


# ------------------------------------------------------------
# Rotation consistency test
# ------------------------------------------------------------
def true_rotation_consistency_test(
    ds_model,
    cnn_model,
    loader,
    spatial_dict,
    lon_mr,
    lat_mr,
    lon_hr,
    lat_hr,
    rotations,
    device,
    max_batches=3
):
    rows = []

    ds_model.eval()
    cnn_model.eval()

    lon_mr_np = spatial_dict["lon_mr"]
    lat_mr_np = spatial_dict["lat_mr"]

    lon_hr_np = spatial_dict["lon_hr"]
    lat_hr_np = spatial_dict["lat_hr"]

    with torch.no_grad():

        for axis, angle in rotations:

            print(f"Testing rotation: axis={axis}, angle={angle}°")

            idx_mr_rot = build_rotation_indices(
                lon_mr_np,
                lat_mr_np,
                axis=axis,
                angle_deg=angle
            )

            idx_hr_rot = build_rotation_indices(
                lon_hr_np,
                lat_hr_np,
                axis=axis,
                angle_deg=angle
            )

            idx_hr_back = build_rotation_indices(
                lon_hr_np,
                lat_hr_np,
                axis=axis,
                angle_deg=-angle
            )

            ds_errs = []
            cnn_errs = []

            for batch_idx, batch in enumerate(loader):

                if max_batches is not None and batch_idx >= max_batches:
                    break

                hr_v, mr_v, _, _ = batch
                mr_v = mr_v.to(device)

                # Original predictions
                ds_orig = forward_deepsphere_rotation(
                    ds_model,
                    mr_v,
                    lon_mr,
                    lat_mr,
                    lon_hr,
                    lat_hr
                )

                cnn_orig = cnn_model(mr_v)

                # Rotate input field on sphere
                mr_rot = remap_vertices(mr_v, idx_mr_rot)

                # Predict from rotated input
                ds_rot_pred = forward_deepsphere_rotation(
                    ds_model,
                    mr_rot,
                    lon_mr,
                    lat_mr,
                    lon_hr,
                    lat_hr
                )

                cnn_rot_pred = cnn_model(mr_rot)

                # Rotate prediction back
                ds_rot_back = remap_vertices(ds_rot_pred, idx_hr_back)
                cnn_rot_back = remap_vertices(cnn_rot_pred, idx_hr_back)

                # Consistency error
                ds_err = torch.sqrt(
                    torch.mean((ds_rot_back - ds_orig) ** 2)
                )

                cnn_err = torch.sqrt(
                    torch.mean((cnn_rot_back - cnn_orig) ** 2)
                )

                ds_errs.append(ds_err.detach().cpu())
                cnn_errs.append(cnn_err.detach().cpu())

            ds_mean = torch.stack(ds_errs).mean().item()
            cnn_mean = torch.stack(cnn_errs).mean().item()

            rows.append({
                "axis": axis,
                "angle_deg": angle,
                "DeepSphere_rotation_RMSE": ds_mean,
                "CNN_rotation_RMSE": cnn_mean,
                "Winner": "DeepSphere" if ds_mean < cnn_mean else "CNN"
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Run test
# ------------------------------------------------------------
rotations = [
    ("x", 10), ("x", 20), ("x", 30),
    ("y", 10), ("y", 20), ("y", 30),
    ("z", 30),
]

rotation_df = true_rotation_consistency_test(
    ds_model=ds_model_for_rotation,
    cnn_model=cnn_model_q,
    loader=val_loader_q_direct,
    spatial_dict=spatial_dict_q,
    lon_mr=lon_mr_q,
    lat_mr=lat_mr_q,
    lon_hr=lon_hr_q,
    lat_hr=lat_hr_q,
    rotations=rotations,
    device=device,
    max_batches=3
)

display(rotation_df)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))

x = np.arange(len(rotation_df))
width = 0.35

plt.bar(
    x - width / 2,
    rotation_df["DeepSphere_rotation_RMSE"],
    width,
    label=ds_model_name
)

plt.bar(
    x + width / 2,
    rotation_df["CNN_rotation_RMSE"],
    width,
    label="CNN"
)

plt.xticks(
    x,
    rotation_df["axis"] + rotation_df["angle_deg"].astype(str) + "°"
)

plt.ylabel("Rotation consistency RMSE")
plt.title("True Spherical Rotation Consistency")
plt.grid(True, axis="y", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
rotation_path = ARTIFACT_DIR / "true_spherical_rotation_consistency.csv"

rotation_df.to_csv(
    rotation_path,
    index=False
)

print("Saved:", rotation_path)
print("CELL 43 PASSED")
