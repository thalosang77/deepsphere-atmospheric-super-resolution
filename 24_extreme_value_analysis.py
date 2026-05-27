# ============================================================
# Extreme-value performance, kthvalue-safe version
# ============================================================

import torch

quantiles = [0.90, 0.95, 0.99]

targets_flat = direct_targets_q.reshape(-1).float().cpu()
n_total = targets_flat.numel()

print("=" * 70)
print("EXTREME HUMIDITY PERFORMANCE")
print("=" * 70)
print("Total values:", n_total)

for q in quantiles:
    # kthvalue is safer than torch.quantile for very large tensors
    k = int(q * n_total)
    k = max(1, min(k, n_total))

    threshold = torch.kthvalue(targets_flat, k).values

    mask = direct_targets_q >= threshold

    model_err_ext = direct_preds_q[mask] - direct_targets_q[mask]
    bil_err_ext = baseline_preds_q[mask] - baseline_targets_q[mask]

    model_rmse_ext = torch.sqrt(torch.mean(model_err_ext ** 2))
    bil_rmse_ext = torch.sqrt(torch.mean(bil_err_ext ** 2))

    model_mae_ext = torch.mean(torch.abs(model_err_ext))
    bil_mae_ext = torch.mean(torch.abs(bil_err_ext))

    rmse_improve = 100 * (bil_rmse_ext.item() - model_rmse_ext.item()) / bil_rmse_ext.item()
    mae_improve = 100 * (bil_mae_ext.item() - model_mae_ext.item()) / bil_mae_ext.item()

    print(f"\nTop {(1-q)*100:.0f}% target values")
    print(f"Threshold              : {threshold.item():.6f}")
    print(f"Samples                : {mask.sum().item()}")
    print(f"RMSE DeepSphere        : {model_rmse_ext.item():.6f}")
    print(f"RMSE Bilinear          : {bil_rmse_ext.item():.6f}")
    print(f"RMSE improvement       : {rmse_improve:.2f}%")
    print(f"MAE DeepSphere         : {model_mae_ext.item():.6f}")
    print(f"MAE Bilinear           : {bil_mae_ext.item():.6f}")
    print(f"MAE improvement        : {mae_improve:.2f}%")

print("\nCELL 21 PASSED")
