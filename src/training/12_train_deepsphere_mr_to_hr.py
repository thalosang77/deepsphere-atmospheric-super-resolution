# ============================================================
# Train DIRECT MR -> HR model for q
# FINAL residual-target DeepSphere experiment
#
# Goal:
#   push RMSE toward / below CNN baseline (~0.022)
#
# Key upgrades:
#   - pure MSE residual learning
#   - lower weight decay
#   - longer cosine schedule
#   - gradient clipping
#   - residual-only supervision
# ============================================================

import torch
import torch.optim as optim

# ------------------------------------------------------------
# IMPORTANT PERFORMANCE SETTINGS
# ------------------------------------------------------------
if torch.cuda.is_available():

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Fastest cudnn kernel search
    torch.backends.cudnn.benchmark = True

print("=" * 70)
print("DIRECT MR -> HR RESIDUAL TRAINING")
print("=" * 70)

# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------
learning_rate = 4e-4

# lighter regularization
weight_decay = 5e-7

# residual learning needs longer schedule
num_epochs = 70

print("learning_rate :", learning_rate)
print("weight_decay  :", weight_decay)
print("num_epochs    :", num_epochs)

# ------------------------------------------------------------
# PURE MSE
#
# MSE generally gives:
#   lower RMSE
#   sharper reconstruction
#   better optimization target
#
# compared with MAE mixtures
# ------------------------------------------------------------
criterion_q_direct = SRLoss(loss_type="mse")

# validation metric
val_metric_q_direct = SRLoss(loss_type="mse")

print("loss function : pure MSE")

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------
optimizer_q_direct = optim.AdamW(
    test_model_q.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=(0.9, 0.99)
)

# ------------------------------------------------------------
# Cosine schedule
#
# Much smoother convergence than plateau scheduler
# ------------------------------------------------------------
scheduler_q_direct = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_q_direct,
    T_max=num_epochs,
    eta_min=5e-6
)

# ------------------------------------------------------------
# Checkpoint path
# ------------------------------------------------------------
ckpt_path_direct = (
    CKPT_DIR /
    "best_direct_mr_to_hr_q_residual_target_final.pt"
)

print("Checkpoint path:")
print(ckpt_path_direct)

# ------------------------------------------------------------
# INITIAL RESIDUAL SCALE DIAGNOSTICS
# ------------------------------------------------------------
if hasattr(test_model_q, "residual_scale"):

    print("\nInitial residual scale stats")

    print(
        "mean:",
        test_model_q.residual_scale.mean().item()
    )

    print(
        "min :",
        test_model_q.residual_scale.min().item()
    )

    print(
        "max :",
        test_model_q.residual_scale.max().item()
    )

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
history_q_direct, direct_state_dict_q = training_direct_mr_to_hr(
    train_model=test_model_q,

    train_loader=train_loader_q_direct,
    val_loader=val_loader_q_direct,

    optimizer=optimizer_q_direct,

    criterion=criterion_q_direct,
    val_metric=val_metric_q_direct,

    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,

    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,

    epochs=num_epochs,

    device=device,

    scheduler=scheduler_q_direct,

    ckpt_path=ckpt_path_direct,

    run_name="direct_mr_to_hr_q_residual_target_final"
)

# ------------------------------------------------------------
# LOAD BEST MODEL
# ------------------------------------------------------------
test_model_q.load_state_dict(
    direct_state_dict_q
)

print("\nLoaded best validation checkpoint.")

# ------------------------------------------------------------
# FINAL DIAGNOSTICS
# ------------------------------------------------------------
best_rmse = min(history_q_direct["val_RMSE"])

best_r2 = max(history_q_direct["val_R2"])

best_corr = max(history_q_direct["val_Corr"])

final_loss = history_q_direct["train_loss"][-1]

print("\n" + "=" * 70)
print("FINAL TRAINING RESULTS")
print("=" * 70)

print("Best checkpoint :", ckpt_path_direct)

print("\nMetrics")
print("-------")

print("Final train loss :", final_loss)

print("Best Val RMSE    :", best_rmse)

print("Best Val R2      :", best_r2)

print("Best Val Corr    :", best_corr)

# ------------------------------------------------------------
# Residual scale diagnostics
# ------------------------------------------------------------
if hasattr(test_model_q, "residual_scale"):

    print("\nFinal residual scale stats")

    print(
        "mean:",
        test_model_q.residual_scale.mean().item()
    )

    print(
        "min :",
        test_model_q.residual_scale.min().item()
    )

    print(
        "max :",
        test_model_q.residual_scale.max().item()
    )

# ------------------------------------------------------------
# Learning curve diagnostics
# ------------------------------------------------------------
best_epoch = int(
    torch.tensor(history_q_direct["val_RMSE"]).argmin().item()
) + 1

print("\nBest epoch:", best_epoch)

print(
    "Best learning rate:",
    history_q_direct["lr"][best_epoch - 1]
)

# ------------------------------------------------------------
# CNN baseline comparison
# ------------------------------------------------------------
cnn_baseline_rmse = 0.022

print("\n" + "=" * 70)
print("CNN BASELINE COMPARISON")
print("=" * 70)

print("CNN baseline RMSE :", cnn_baseline_rmse)
print("DeepSphere RMSE   :", best_rmse)

improvement_gap = best_rmse - cnn_baseline_rmse

print("Gap to CNN        :", improvement_gap)

if best_rmse <= cnn_baseline_rmse:
    print("\nSUCCESS: DeepSphere matched/exceeded CNN.")
else:
    print("\nDeepSphere still above CNN baseline.")

print("\nCELL 12 PASSED")
