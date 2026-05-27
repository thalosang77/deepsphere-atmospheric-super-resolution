# ============================================================
# Temporal generalization split: train Mar/Sep, test Dec
# ============================================================

time_values = ds_q["time"].values
months = pd.to_datetime(time_values).month

train_time_mask = np.isin(months, [3, 6, 9])
test_time_mask = months == 12

X_train_temporal = X_q_grid[train_time_mask]
X_test_temporal = X_q_grid[test_time_mask]

print("=" * 70)
print("TEMPORAL GENERALIZATION SPLIT")
print("=" * 70)
print("Train months: March, June, September")
print("Test month  : December")
print("Train shape :", X_train_temporal.shape)
print("Test shape  :", X_test_temporal.shape)

assert X_train_temporal.shape[0] > 0
assert X_test_temporal.shape[0] > 0

train_temporal_dataset_dry = DownsampleERA5(
    X_train_temporal, spatial_dict_q, mode="train",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)

train_temporal_loader_dry = DataLoader(
    train_temporal_dataset_dry,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

norm_params_temporal = compute_norm_params(
    train_temporal_loader_dry,
    n_channels_q,
    save_path=str(ARTIFACT_DIR / "norm_params_q_temporal_train_mar_jun_sep.json")
)

train_temporal_dataset_direct = DownsampleERA5(
    X_train_temporal, spatial_dict_q, mode="val",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)
train_temporal_dataset_direct.norm_params = norm_params_temporal

test_temporal_dataset_direct = DownsampleERA5(
    X_test_temporal, spatial_dict_q, mode="val",
    clip_specific_humidity=PHYSICAL_CLIP_Q,
    q_min=Q_MIN_PHYSICAL
)
test_temporal_dataset_direct.norm_params = norm_params_temporal

train_temporal_loader_direct = DataLoader(
    train_temporal_dataset_direct,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

test_temporal_loader_direct = DataLoader(
    test_temporal_dataset_direct,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

print("Temporal loaders ready.")
print("Train batches:", len(train_temporal_loader_direct))
print("Test batches :", len(test_temporal_loader_direct))
print("CELL 26 PASSED")
