# ============================================================
# Fine-tune best MR -> HR model
# Start from best Cell 12 checkpoint
# ============================================================

import torch
import torch.optim as optim

print("=" * 70)
print("CELL 12B — FINE-TUNING FROM BEST CHECKPOINT")
print("=" * 70)

# ------------------------------------------------------------
# Load best model from Cell 12
# ------------------------------------------------------------
ckpt_path_direct = (
    CKPT_DIR /
    "best_direct_mr_to_hr_q_residual_target_final.pt"
)

checkpoint = torch.load(
    ckpt_path_direct,
    map_location=device
)

test_model_q.load_state_dict(
    checkpoint["model_state_dict"]
)

test_model_q.to(device)

print("Loaded checkpoint:", ckpt_path_direct)

# ------------------------------------------------------------
# Fine-tuning hyperparameters
# ------------------------------------------------------------
learning_rate_ft = 8e-5
weight_decay_ft = 1e-7
num_epochs_ft = 40

criterion_q_ft = SRLoss(loss_type="mse")
val_metric_q_ft = SRLoss(loss_type="mse")

optimizer_q_ft = optim.AdamW(
    test_model_q.parameters(),
    lr=learning_rate_ft,
    weight_decay=weight_decay_ft,
    betas=(0.9, 0.99)
)

scheduler_q_ft = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_q_ft,
    T_max=num_epochs_ft,
    eta_min=1e-6
)

ckpt_path_ft = (
    CKPT_DIR /
    "best_direct_mr_to_hr_q_residual_target_finetuned.pt"
)

# ------------------------------------------------------------
# Fine-tune
# ------------------------------------------------------------
history_q_ft, ft_state_dict_q = training_direct_mr_to_hr(
    train_model=test_model_q,

    train_loader=train_loader_q_direct,
    val_loader=val_loader_q_direct,

    optimizer=optimizer_q_ft,

    criterion=criterion_q_ft,
    val_metric=val_metric_q_ft,

    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,

    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,

    epochs=num_epochs_ft,

    device=device,

    scheduler=scheduler_q_ft,

    ckpt_path=ckpt_path_ft,

    run_name="direct_mr_to_hr_q_residual_target_finetuned"
)

# ------------------------------------------------------------
# Load best fine-tuned model
# ------------------------------------------------------------
test_model_q.load_state_dict(ft_state_dict_q)

best_rmse_ft = min(history_q_ft["val_RMSE"])
best_r2_ft = max(history_q_ft["val_R2"])
best_corr_ft = max(history_q_ft["val_Corr"])

print("\nCELL 12B PASSED")
print("Best fine-tuned checkpoint:", ckpt_path_ft)
print("Best fine-tuned RMSE      :", best_rmse_ft)
print("Best fine-tuned R2        :", best_r2_ft)
print("Best fine-tuned Corr      :", best_corr_ft)

cnn_baseline_rmse = 0.022
print("Gap to CNN baseline       :", best_rmse_ft - cnn_baseline_rmse)
