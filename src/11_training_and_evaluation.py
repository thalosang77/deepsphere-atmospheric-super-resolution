# ============================================================
# Training / evaluation functions
# Residual-target version with checkpointing
#
# Architecture assumption:
#   prediction = bilinear/input skip + learned DeepSphere residual
#
# Training loss:
#   learned residual  vs  true residual
#
# Validation metrics:
#   full prediction  vs  full target
# ============================================================

import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------
# Precision / performance controls
# ------------------------------------------------------------
USE_AMP = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ------------------------------------------------------------
# Checkpoint helper
# ------------------------------------------------------------
def save_checkpoint(model_state, history, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model_state,
            "history": history,
            "metadata": metadata or {},
        },
        path
    )

    print("Saved checkpoint:", path)


# ------------------------------------------------------------
# Residual-target loss helper
# ------------------------------------------------------------
def compute_residual_loss(model, pred, inputs, targets, criterion):
    """
    Full prediction:
        pred = baseline + learned_residual

    Residual-target training:
        learned_residual = pred - baseline
        true_residual    = targets - baseline
    """

    if hasattr(model, "skip_unpool"):

        with torch.no_grad():
            baseline = model.skip_unpool(inputs).detach()

        pred_residual = pred - baseline
        target_residual = targets - baseline

        loss = criterion(pred_residual, target_residual)

    else:
        loss = criterion(pred, targets)

    return loss


# ------------------------------------------------------------
# LR -> MR training
# ------------------------------------------------------------
def training_lr_to_mr(
    train_model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    val_metric,
    t_lon_lr,
    t_lat_lr,
    t_lon_mr,
    t_lat_mr,
    epochs,
    device,
    scheduler=None,
    ckpt_path=None,
    run_name="lr_to_mr"
):
    print("=" * 70)
    print("STARTING TRAINING: LR -> MR")
    print("=" * 70)

    print("Epochs:", epochs)
    print("AMP   :", USE_AMP)
    print(
        "TF32  :",
        torch.backends.cuda.matmul.allow_tf32
        if torch.cuda.is_available()
        else False
    )
    print("Loss  : residual-target")

    history = {
        "train_loss": [],
        "val_RMSE": [],
        "val_MAE": [],
        "val_R2": [],
        "val_Corr": [],
        "lr": [],
    }

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    train_model.to(device)

    best_val_rmse = float("inf")
    best_model_state = None
    best_epoch = None

    for epoch in range(epochs):

        start_time = time.time()

        train_model.train()
        epoch_train_loss = 0.0
        valid_batches = 0

        for batch in train_loader:

            hr_v, mr_v, lr_v, _, _, _ = batch

            inputs = lr_v.to(device, non_blocking=True)
            targets = mr_v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                pred = train_model(
                    x_low=inputs,
                    lon_low=t_lon_lr,
                    lat_low=t_lat_lr,
                    lon_high=t_lon_mr,
                    lat_high=t_lat_mr,
                    cond_low=inputs
                )

                loss = compute_residual_loss(
                    train_model,
                    pred,
                    inputs,
                    targets,
                    criterion
                )

            if not torch.isfinite(loss):
                print("Non-finite loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                train_model.parameters(),
                max_norm=2.0
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(
                "All training batches were skipped due to non-finite loss."
            )

        avg_train_loss = epoch_train_loss / valid_batches

        # ----------------------------------------------------
        # Validation: full prediction vs full target
        # ----------------------------------------------------
        train_model.eval()

        val_preds = []
        val_targets = []

        with torch.no_grad():

            for batch in val_loader:

                hr_v, mr_v, lr_v, _, _, _ = batch

                inputs = lr_v.to(device, non_blocking=True)
                targets = mr_v.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):

                    pred = train_model(
                        x_low=inputs,
                        lon_low=t_lon_lr,
                        lat_low=t_lat_lr,
                        lon_high=t_lon_mr,
                        lat_high=t_lat_mr,
                        cond_low=inputs
                    )

                val_preds.append(pred.float().cpu())
                val_targets.append(targets.float().cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        metrics = val_metric.compute_metrics(
            val_preds,
            val_targets
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_RMSE"].append(metrics["RMSE"])
        history["val_MAE"].append(metrics["MAE"])
        history["val_R2"].append(metrics["R2"])
        history["val_Corr"].append(metrics["Corr"])
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        if metrics["RMSE"] < best_val_rmse:

            best_val_rmse = metrics["RMSE"]
            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                train_model.state_dict()
            )

            print(
                f"New best LR->MR model | "
                f"epoch {best_epoch} | "
                f"RMSE {best_val_rmse:.6f}"
            )

            if ckpt_path is not None:

                save_checkpoint(
                    best_model_state,
                    history,
                    ckpt_path,
                    metadata={
                        "run_name": run_name,
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val_rmse,
                        "task": "LR->MR",
                        "loss_mode": "residual_target",
                        "amp": USE_AMP,
                        "tf32": torch.backends.cuda.matmul.allow_tf32
                        if torch.cuda.is_available()
                        else False,
                        "conditioning": "cond_low=inputs",
                        "local_stencil": getattr(
                            train_model,
                            "use_local_stencil",
                            None
                        ),
                        "flow_blocks": getattr(
                            train_model,
                            "use_flow_blocks",
                            None
                        ),
                    }
                )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"ResLoss {avg_train_loss:.6f} | "
            f"RMSE {metrics['RMSE']:.6f} | "
            f"MAE {metrics['MAE']:.6f} | "
            f"R2 {metrics['R2']:.6f} | "
            f"Corr {metrics['Corr']:.6f} | "
            f"LR {current_lr:.2e} | "
            f"{epoch_time:.2f}s"
        )

    if best_model_state is None:
        best_model_state = copy.deepcopy(
            train_model.state_dict()
        )

    return history, best_model_state


# ------------------------------------------------------------
# Direct MR -> HR training
# ------------------------------------------------------------
def training_direct_mr_to_hr(
    train_model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    val_metric,
    t_lon_mr,
    t_lat_mr,
    t_lon_hr,
    t_lat_hr,
    epochs,
    device,
    scheduler=None,
    ckpt_path=None,
    run_name="direct_mr_to_hr"
):
    print("=" * 70)
    print("STARTING DIRECT TRAINING: MR -> HR")
    print("=" * 70)

    print("Epochs:", epochs)
    print("AMP   :", USE_AMP)
    print(
        "TF32  :",
        torch.backends.cuda.matmul.allow_tf32
        if torch.cuda.is_available()
        else False
    )
    print("Loss  : residual-target")

    history = {
        "train_loss": [],
        "val_RMSE": [],
        "val_MAE": [],
        "val_R2": [],
        "val_Corr": [],
        "lr": [],
    }

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    train_model.to(device)

    best_val_rmse = float("inf")
    best_model_state = None
    best_epoch = None

    for epoch in range(epochs):

        start_time = time.time()

        train_model.train()
        epoch_train_loss = 0.0
        valid_batches = 0

        for batch in train_loader:

            hr_v, mr_v, _, _ = batch

            inputs = mr_v.to(device, non_blocking=True)
            targets = hr_v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                pred = train_model(
                    x_low=inputs,
                    lon_low=t_lon_mr,
                    lat_low=t_lat_mr,
                    lon_high=t_lon_hr,
                    lat_high=t_lat_hr,
                    cond_low=inputs
                )

                loss = compute_residual_loss(
                    train_model,
                    pred,
                    inputs,
                    targets,
                    criterion
                )

            if not torch.isfinite(loss):
                print("Non-finite loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                train_model.parameters(),
                max_norm=2.0
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            valid_batches += 1

        if valid_batches == 0:
            raise RuntimeError(
                "All training batches were skipped due to non-finite loss."
            )

        avg_train_loss = epoch_train_loss / valid_batches

        # ----------------------------------------------------
        # Validation: full prediction vs full target
        # ----------------------------------------------------
        train_model.eval()

        val_preds = []
        val_targets = []

        with torch.no_grad():

            for batch in val_loader:

                hr_v, mr_v, _, _ = batch

                inputs = mr_v.to(device, non_blocking=True)
                targets = hr_v.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):

                    pred = train_model(
                        x_low=inputs,
                        lon_low=t_lon_mr,
                        lat_low=t_lat_mr,
                        lon_high=t_lon_hr,
                        lat_high=t_lat_hr,
                        cond_low=inputs
                    )

                val_preds.append(pred.float().cpu())
                val_targets.append(targets.float().cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_targets = torch.cat(val_targets, dim=0)

        metrics = val_metric.compute_metrics(
            val_preds,
            val_targets
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_RMSE"].append(metrics["RMSE"])
        history["val_MAE"].append(metrics["MAE"])
        history["val_R2"].append(metrics["R2"])
        history["val_Corr"].append(metrics["Corr"])
        history["lr"].append(current_lr)

        if scheduler is not None:
            scheduler.step()

        if metrics["RMSE"] < best_val_rmse:

            best_val_rmse = metrics["RMSE"]
            best_epoch = epoch + 1

            best_model_state = copy.deepcopy(
                train_model.state_dict()
            )

            print(
                f"New best direct MR->HR model | "
                f"epoch {best_epoch} | "
                f"RMSE {best_val_rmse:.6f}"
            )

            if ckpt_path is not None:

                save_checkpoint(
                    best_model_state,
                    history,
                    ckpt_path,
                    metadata={
                        "run_name": run_name,
                        "best_epoch": best_epoch,
                        "best_val_rmse": best_val_rmse,
                        "task": "MR->HR direct",
                        "loss_mode": "residual_target",
                        "amp": USE_AMP,
                        "tf32": torch.backends.cuda.matmul.allow_tf32
                        if torch.cuda.is_available()
                        else False,
                        "conditioning": "cond_low=inputs",
                        "local_stencil": getattr(
                            train_model,
                            "use_local_stencil",
                            None
                        ),
                        "flow_blocks": getattr(
                            train_model,
                            "use_flow_blocks",
                            None
                        ),
                    }
                )

        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"ResLoss {avg_train_loss:.6f} | "
            f"RMSE {metrics['RMSE']:.6f} | "
            f"MAE {metrics['MAE']:.6f} | "
            f"R2 {metrics['R2']:.6f} | "
            f"Corr {metrics['Corr']:.6f} | "
            f"LR {current_lr:.2e} | "
            f"{epoch_time:.2f}s"
        )

    if best_model_state is None:
        best_model_state = copy.deepcopy(
            train_model.state_dict()
        )

    return history, best_model_state


# ------------------------------------------------------------
# Zero-shot MR -> HR evaluation
# ------------------------------------------------------------
def evaluate_zero_shot(
    model,
    train_state_dict,
    val_loader,
    val_metric,
    t_lon_mr,
    t_lat_mr,
    t_lon_hr,
    t_lat_hr,
    device
):
    print("=" * 70)
    print("STARTING ZERO-SHOT EVALUATION: MR -> HR")
    print("=" * 70)

    model.to(device)

    target_param_dict = dict(model.named_parameters())
    transferable_state = {}

    for key, value in train_state_dict.items():

        if key in target_param_dict:

            if target_param_dict[key].shape == value.shape:
                transferable_state[key] = value

    missing, unexpected = model.load_state_dict(
        transferable_state,
        strict=False
    )

    print("Transferred parameter keys:", len(transferable_state))
    print("Missing keys             :", len(missing))
    print("Unexpected keys          :", len(unexpected))

    model.eval()

    all_preds = []
    all_targets = []

    metrics_total = {
        "MAE": [],
        "RMSE": [],
        "MAPE": [],
        "Corr": [],
        "PSNR": [],
        "R2": [],
    }

    with torch.no_grad():

        for batch in val_loader:

            hr_v, mr_v, _, _ = batch

            inputs = mr_v.to(device, non_blocking=True)
            targets = hr_v.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                pred = model(
                    x_low=inputs,
                    lon_low=t_lon_mr,
                    lat_low=t_lat_mr,
                    lon_high=t_lon_hr,
                    lat_high=t_lat_hr,
                    cond_low=inputs
                )

            all_preds.append(pred.float().cpu())
            all_targets.append(targets.float().cpu())

            batch_metrics = val_metric.compute_metrics(
                pred.float(),
                targets.float()
            )

            for key in metrics_total:
                metrics_total[key].append(batch_metrics[key])

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = val_metric.compute_metrics(
        all_preds,
        all_targets
    )

    return metrics, metrics_total, all_preds, all_targets


# ------------------------------------------------------------
# Standard MR -> HR evaluation
# ------------------------------------------------------------
def evaluate_model_mr_to_hr(
    model,
    val_loader,
    val_metric,
    t_lon_mr,
    t_lat_mr,
    t_lon_hr,
    t_lat_hr,
    device
):
    print("=" * 70)
    print("EVALUATING MODEL: MR -> HR")
    print("=" * 70)

    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():

        for batch in val_loader:

            hr_v, mr_v, _, _ = batch

            inputs = mr_v.to(device, non_blocking=True)
            targets = hr_v.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):

                pred = model(
                    x_low=inputs,
                    lon_low=t_lon_mr,
                    lat_low=t_lat_mr,
                    lon_high=t_lon_hr,
                    lat_high=t_lat_hr,
                    cond_low=inputs
                )

            all_preds.append(pred.float().cpu())
            all_targets.append(targets.float().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = val_metric.compute_metrics(
        all_preds,
        all_targets
    )

    return metrics, all_preds, all_targets


print("CELL 11 PASSED")
