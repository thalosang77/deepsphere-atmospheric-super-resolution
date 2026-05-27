# ============================================================
# Train/val/test split + dataloaders
# Physics-aware SR version
#
# Keeps:
#   - LR -> MR training
#   - MR -> HR evaluation
#   - direct MR -> HR diagnostic training
#
# Adds:
#   - strict chronological split
#   - no temporal shuffling by default
#   - temporal metadata
#   - future multi-variable structure
#   - physics diagnostics hooks
# ============================================================

from torch.utils.data import DataLoader
import numpy as np
import torch
import json


# ------------------------------------------------------------
# Hyperparameters for data prep
# ------------------------------------------------------------
BATCH_SIZE = 16
TRAIN_FRACTION = 0.8
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# For atmospheric fields, chronological ordering is safer.
SHUFFLE_TRAIN = False

# ------------------------------------------------------------
# Input state dictionary
# Future-proof for q, u, v, t, z, etc.
# ------------------------------------------------------------
state_variables = {
    "q": X_q_grid
}

X_raw_q = state_variables["q"]

# Accept either:
# [T, Y, X, P, C] or already flattened [T, Y, X, C_flat]
if X_raw_q.ndim == 5:
    T, Y, Xdim, P, Cvar = X_raw_q.shape
    n_channels_q = P * Cvar
else:
    T, Y, Xdim, n_channels_q = X_raw_q.shape

assert n_channels_q == 37, (
    f"Expected 37 q pressure-level channels, got {n_channels_q}"
)

total_timesteps = X_raw_q.shape[0]
train_split_idx = int(TRAIN_FRACTION * total_timesteps)

assert 0 < train_split_idx < total_timesteps

# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------
X_train_q = X_raw_q[:train_split_idx]
X_val_q = X_raw_q[train_split_idx:]

# ------------------------------------------------------------
# Temporal metadata
# ------------------------------------------------------------
time_values = None

if "ds_q" in globals() and "time" in ds_q.coords:
    time_values = ds_q["time"].values
    assert len(time_values) == total_timesteps

    time_train = time_values[:train_split_idx]
    time_val = time_values[train_split_idx:]

    months = np.array([
        np.datetime64(t, "M").astype(str).split("-")[1]
        for t in time_values
    ])

    hours = np.array([
        str(np.datetime64(t, "h")).split("T")[1].split(":")[0]
        for t in time_values
    ])

else:
    time_train = None
    time_val = None
    months = None
    hours = None

temporal_metadata = {
    "has_time_values": time_values is not None,
    "total_timesteps": int(total_timesteps),
    "train_timesteps": int(X_train_q.shape[0]),
    "val_timesteps": int(X_val_q.shape[0]),
    "train_fraction": float(TRAIN_FRACTION),
}

if time_values is not None:
    temporal_metadata.update({
        "train_start": str(time_train[0]),
        "train_end": str(time_train[-1]),
        "val_start": str(time_val[0]),
        "val_end": str(time_val[-1]),
        "unique_months": sorted(list(set(months.tolist()))),
        "unique_hours": sorted(list(set(hours.tolist()))),
    })

with open(ARTIFACT_DIR / "temporal_split_metadata_q.json", "w") as f:
    json.dump(temporal_metadata, f, indent=2)

# ------------------------------------------------------------
# Print split summary
# ------------------------------------------------------------
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Total timesteps :", total_timesteps)
print("Train timesteps :", X_train_q.shape[0])
print("Val timesteps   :", X_val_q.shape[0])
print("Channels        :", n_channels_q)
print("Batch size      :", BATCH_SIZE)
print("Pin memory      :", PIN_MEMORY)
print("Shuffle train   :", SHUFFLE_TRAIN)

if time_values is not None:
    print("Train period    :", temporal_metadata["train_start"], "->", temporal_metadata["train_end"])
    print("Val period      :", temporal_metadata["val_start"], "->", temporal_metadata["val_end"])
    print("Months          :", temporal_metadata["unique_months"])
    print("Hours           :", temporal_metadata["unique_hours"])

assert X_train_q.shape[0] > 0
assert X_val_q.shape[0] > 0


# ------------------------------------------------------------
# Dry training dataset for normalization
# Normalization computed from MR training data only.
# ------------------------------------------------------------
train_dataset_q_dry = DownsampleERA5(
    X_train_q,
    spatial_dict_q,
    mode="train",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

train_loader_q_dry = DataLoader(
    train_dataset_q_dry,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

norm_params_q = compute_norm_params(
    train_loader_q_dry,
    n_channels_q,
    save_path=str(ARTIFACT_DIR / "norm_params_q_regular3h.json")
)


# ------------------------------------------------------------
# LR -> MR training loaders
# Original DeepSphere-ResNet training task
# ------------------------------------------------------------
train_dataset_q = DownsampleERA5(
    X_train_q,
    spatial_dict_q,
    mode="train",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

train_dataset_q.norm_params = norm_params_q

train_loader_q = DataLoader(
    train_dataset_q,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TRAIN,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_dataset_q = DownsampleERA5(
    X_val_q,
    spatial_dict_q,
    mode="train",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

val_dataset_q.norm_params = norm_params_q

val_loader_q = DataLoader(
    val_dataset_q,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ------------------------------------------------------------
# MR -> HR held-out evaluation loader
# IMPORTANT:
# Do NOT use X_raw_q here, because that contaminates test metrics.
# This loader uses only validation timestamps.
# ------------------------------------------------------------
test_dataset_q = DownsampleERA5(
    X_val_q,
    spatial_dict_q,
    mode="val",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

test_dataset_q.norm_params = norm_params_q

test_loader_q = DataLoader(
    test_dataset_q,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ------------------------------------------------------------
# Direct MR -> HR diagnostic loaders
# Useful for comparing against bilinear/CNN baselines.
# ------------------------------------------------------------
train_dataset_q_direct = DownsampleERA5(
    X_train_q,
    spatial_dict_q,
    mode="val",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

train_dataset_q_direct.norm_params = norm_params_q

train_loader_q_direct = DataLoader(
    train_dataset_q_direct,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE_TRAIN,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_dataset_q_direct = DownsampleERA5(
    X_val_q,
    spatial_dict_q,
    mode="val",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

val_dataset_q_direct.norm_params = norm_params_q

val_loader_q_direct = DataLoader(
    val_dataset_q_direct,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ------------------------------------------------------------
# Physics diagnostic helpers
# ------------------------------------------------------------
def area_weighted_column_integral(vertices, area_weights=None):
    """
    vertices:
        [N, C] or [B, N, C]

    area_weights:
        [N] or None

    Returns:
        area-weighted sum per channel.
    """
    if vertices.ndim == 2:
        vertices = vertices.unsqueeze(0)

    B, N, C = vertices.shape

    if area_weights is None:
        return vertices.mean(dim=1)

    area_weights = area_weights.to(vertices.device).view(1, N, 1)
    area_weights = area_weights / area_weights.sum()

    return (vertices * area_weights).sum(dim=1)


def variance_retention(low_or_pred, target):
    """
    Simple diagnostic:
    compares spatial variance of prediction/input to target.
    """
    if low_or_pred.ndim == 2:
        low_or_pred = low_or_pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    pred_var = low_or_pred.var(dim=1)
    target_var = target.var(dim=1)

    return pred_var / (target_var + 1e-8)


# ------------------------------------------------------------
# Loader summaries
# ------------------------------------------------------------
print("\nDataloaders ready.")
print("LR->MR train batches       :", len(train_loader_q))
print("LR->MR val batches         :", len(val_loader_q))
print("MR->HR held-out test       :", len(test_loader_q))
print("MR->HR direct train        :", len(train_loader_q_direct))
print("MR->HR direct val          :", len(val_loader_q_direct))


# ------------------------------------------------------------
# One LR -> MR sample sanity check
# ------------------------------------------------------------
sample_train = train_dataset_q[0]

hr_v, mr_v, lr_v, hr_raw, mr_raw, lr_raw = sample_train

print("\n" + "=" * 70)
print("LR -> MR SAMPLE SHAPE CHECK")
print("=" * 70)

print("hr_v   :", hr_v.shape)
print("mr_v   :", mr_v.shape)
print("lr_v   :", lr_v.shape)
print("hr_raw :", hr_raw.shape)
print("mr_raw :", mr_raw.shape)
print("lr_raw :", lr_raw.shape)

assert hr_v.shape == (spatial_dict_q["n_hr"], n_channels_q)
assert mr_v.shape == (spatial_dict_q["n_mr"], n_channels_q)
assert lr_v.shape == (spatial_dict_q["n_lr"], n_channels_q)


# ------------------------------------------------------------
# One direct MR -> HR sample sanity check
# ------------------------------------------------------------
sample_direct = train_dataset_q_direct[0]

hr_v_d, mr_v_d, hr_raw_d, mr_raw_d = sample_direct

print("\n" + "=" * 70)
print("DIRECT MR -> HR SAMPLE SHAPE CHECK")
print("=" * 70)

print("hr_v_direct   :", hr_v_d.shape)
print("mr_v_direct   :", mr_v_d.shape)
print("hr_raw_direct :", hr_raw_d.shape)
print("mr_raw_direct :", mr_raw_d.shape)

assert hr_v_d.shape == (spatial_dict_q["n_hr"], n_channels_q)
assert mr_v_d.shape == (spatial_dict_q["n_mr"], n_channels_q)


# ------------------------------------------------------------
# Numerical sanity checks
# ------------------------------------------------------------
for name, tensor in {
    "hr_v": hr_v,
    "mr_v": mr_v,
    "lr_v": lr_v,
    "hr_v_direct": hr_v_d,
    "mr_v_direct": mr_v_d,
}.items():

    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN/Inf"
    )


# ------------------------------------------------------------
# Basic physical sanity diagnostics
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("BASIC PHYSICAL DIAGNOSTICS")
print("=" * 70)

print("HR q min/max:", hr_v.min().item(), hr_v.max().item())
print("MR q min/max:", mr_v.min().item(), mr_v.max().item())
print("LR q min/max:", lr_v.min().item(), lr_v.max().item())

vr_lr_mr = variance_retention(lr_v, mr_v[: lr_v.shape[0]])

print("Variance diagnostic computed.")

print("\nCELL 8 PASSED")
