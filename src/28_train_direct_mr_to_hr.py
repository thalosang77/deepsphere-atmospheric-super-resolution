# ============================================================
# Train temporal generalization model
# Train: Mar/Jun/Sep
# Test : Dec
# ============================================================

import torch
import torch.optim as optim

print("=" * 70)
print("CELL 27 — TEMPORAL GENERALIZATION MODEL")
print("=" * 70)

# ------------------------------------------------------------
# Model
# Must match updated Cell 9 architecture:
#   - bilinear skip + learned residual
#   - local spherical stencil branch
#   - optional flow-aware branch
#   - HR embedding
# ------------------------------------------------------------
temporal_model_q = DeepSphereResNet(
    l_low=L_mr_q,
    l_high=L_hr_q,
    unpool_matrix=unpool_mr_hr_q,

    in_channels=in_channels,
    out_channels=out_channels,

    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    n_resblock=n_resblock,
    poly_order=poly_order,
    dropout_rate=dropout_rate,

    use_residual_skip=True,
    residual_scale_init=0.01,
    skip_type="bilinear",

    n_lat_low=spatial_dict_q["n_lat_mr"],
    n_lon_low=spatial_dict_q["n_lon_mr"],
    n_lat_high=spatial_dict_q["n_lat_hr"],
    n_lon_high=spatial_dict_q["n_lon_hr"],

    use_hr_embedding=use_hr_embedding,

    use_local_stencil=use_local_stencil,

    use_flow_blocks=use_flow_blocks,
    cond_channels=cond_channels,

    edge_index_low=edge_index_mr_q,
    edge_type_low=edge_type_mr_q,

    edge_vectors_low=edge_vectors_mr_q,
    edge_distance_low=edge_distance_mr_q,
    t_kernel_low=t_kernel_mr_q,
).to(device)

# ------------------------------------------------------------
# Optimizer
# Same improved residual-target setup as Cell 12
# ------------------------------------------------------------
learning_rate_temporal = 4e-4
weight_decay_temporal = 5e-7
num_epochs_temporal = 70

optimizer_temporal = optim.AdamW(
    temporal_model_q.parameters(),
    lr=learning_rate_temporal,
    weight_decay=weight_decay_temporal,
    betas=(0.9, 0.99)
)

scheduler_temporal = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_temporal,
    T_max=num_epochs_temporal,
    eta_min=5e-6
)

ckpt_path_temporal = (
    CKPT_DIR /
    "best_temporal_generalization_train_MarJunSep_test_Dec.pt"
)

print("Train months : March, June, September")
print("Test month   : December")
print("LR           :", learning_rate_temporal)
print("Weight decay :", weight_decay_temporal)
print("Epochs       :", num_epochs_temporal)
print("Checkpoint   :", ckpt_path_temporal)

# ------------------------------------------------------------
# Train temporal model
# ------------------------------------------------------------
history_temporal, temporal_state_dict = training_direct_mr_to_hr(
    train_model=temporal_model_q,

    train_loader=train_temporal_loader_direct,
    val_loader=test_temporal_loader_direct,

    optimizer=optimizer_temporal,

    criterion=SRLoss(loss_type="mse"),
    val_metric=SRLoss(loss_type="mse"),

    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,

    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,

    epochs=num_epochs_temporal,

    device=device,

    scheduler=scheduler_temporal,

    ckpt_path=ckpt_path_temporal,

    run_name="temporal_generalization_train_MarJunSep_test_Dec"
)

# ------------------------------------------------------------
# Load best temporal model
# ------------------------------------------------------------
temporal_model_q.load_state_dict(
    temporal_state_dict
)

print("\nLoaded best temporal-generalization model.")

# ------------------------------------------------------------
# Evaluate temporal DeepSphere model on December
# ------------------------------------------------------------
temporal_metrics_q, temporal_preds_q, temporal_targets_q = evaluate_model_mr_to_hr(
    model=temporal_model_q,

    val_loader=test_temporal_loader_direct,

    val_metric=SRLoss(loss_type="mse"),

    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,

    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,

    device=device
)

# ------------------------------------------------------------
# Bilinear baseline on same December split
# ------------------------------------------------------------
temporal_baseline_metrics_q, _, temporal_baseline_preds_q, temporal_baseline_targets_q = evaluate_bilinear_baseline(
    val_loader=test_temporal_loader_direct,

    val_metric=SRLoss(loss_type="mse"),

    n_lat_mr=spatial_dict_q["n_lat_mr"],
    n_lon_mr=spatial_dict_q["n_lon_mr"],

    n_lat_hr=spatial_dict_q["n_lat_hr"],
    n_lon_hr=spatial_dict_q["n_lon_hr"],

    channels=in_channels,

    device=device
)

# ------------------------------------------------------------
# Report comparison
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("TEMPORAL GENERALIZATION: DECEMBER TEST")
print("=" * 70)

for key in ["RMSE", "MAE", "Corr", "PSNR", "R2"]:

    ds = temporal_metrics_q[key]
    bl = temporal_baseline_metrics_q[key]

    if key in ["RMSE", "MAE"]:
        imp = 100.0 * (bl - ds) / bl
        better = "DeepSphere" if ds < bl else "Bilinear"
    else:
        imp = 100.0 * (ds - bl) / abs(bl)
        better = "DeepSphere" if ds > bl else "Bilinear"

    print(
        f"{key:>6} | "
        f"DeepSphere: {ds:.6f} | "
        f"Bilinear: {bl:.6f} | "
        f"Improvement: {imp:.2f}% | "
        f"Better: {better}"
    )

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
print("\nSanity checks:")
print(
    "Temporal prediction range:",
    temporal_preds_q.min().item(),
    temporal_preds_q.max().item()
)

print(
    "Temporal target range    :",
    temporal_targets_q.min().item(),
    temporal_targets_q.max().item()
)

print(
    "Bilinear prediction range:",
    temporal_baseline_preds_q.min().item(),
    temporal_baseline_preds_q.max().item()
)

print("\nCELL 27 PASSED")
