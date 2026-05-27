# ============================================================
# Bilinear interpolation baseline for MR -> HR
# Same validation split as direct DeepSphere
# ============================================================

import torch
import torch.nn.functional as F

def evaluate_bilinear_baseline(
    val_loader,
    val_metric,
    n_lat_mr,
    n_lon_mr,
    n_lat_hr,
    n_lon_hr,
    channels,
    device
):
    print("=" * 70)
    print("EVALUATING BILINEAR BASELINE: MR -> HR")
    print("=" * 70)

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
            B = mr_v.shape[0]

            mr_grid = mr_v.reshape(B, n_lat_mr, n_lon_mr, channels)
            mr_grid = mr_grid.permute(0, 3, 1, 2).to(device)

            pred_hr_grid = F.interpolate(
                mr_grid,
                size=(n_lat_hr, n_lon_hr),
                mode="bilinear",
                align_corners=False
            )

            pred_hr_grid = pred_hr_grid.permute(0, 2, 3, 1).cpu()
            pred_hr_v = pred_hr_grid.reshape(B, n_lat_hr * n_lon_hr, channels)

            targets = hr_v

            all_preds.append(pred_hr_v)
            all_targets.append(targets)

            batch_metrics = val_metric.compute_metrics(pred_hr_v, targets)
            for key in metrics_total:
                metrics_total[key].append(batch_metrics[key])

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = val_metric.compute_metrics(all_preds, all_targets)
    return metrics, metrics_total, all_preds, all_targets


baseline_metrics_q, baseline_metrics_total_q, baseline_preds_q, baseline_targets_q = evaluate_bilinear_baseline(
    val_loader=val_loader_q_direct,
    val_metric=val_metric_q_direct,
    n_lat_mr=spatial_dict_q["n_lat_mr"],
    n_lon_mr=spatial_dict_q["n_lon_mr"],
    n_lat_hr=spatial_dict_q["n_lat_hr"],
    n_lon_hr=spatial_dict_q["n_lon_hr"],
    channels=in_channels,
    device=device
)

print("\nBILINEAR BASELINE METRICS")
print("=" * 60)
for key, value in baseline_metrics_q.items():
    print(f"{key:>6}: {value:.6f}")

print("\nSanity checks:")
print("Bilinear prediction range:", baseline_preds_q.min().item(), baseline_preds_q.max().item())
print("Target range             :", baseline_targets_q.min().item(), baseline_targets_q.max().item())

print("\nCELL 14 PASSED")
