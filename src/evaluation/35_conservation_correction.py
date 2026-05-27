# ============================================================
# Moisture-conservation correction for DeepSphere
# Area-mean correction using bilinear reference
# ============================================================

import torch
import pandas as pd

print("=" * 70)
print("CELL 33 — CONSERVATION-CORRECTED DEEPSPHERE")
print("=" * 70)

# ------------------------------------------------------------
# Safety checks
# ------------------------------------------------------------
for name in ["ds_phys", "bl_phys", "gt_phys", "area_w_hr", "mean", "std"]:
    assert name in globals(), f"Missing {name}. Run CELLS 30–31 first."

# ------------------------------------------------------------
# Area-mean conservation correction
# ------------------------------------------------------------
def apply_area_mean_conservation(pred_phys, reference_phys, area_w_hr):
    """
    Shifts each sample/channel by a constant so that
    area-weighted total moisture matches the reference.

    pred_phys:
        [B, N_hr, C]

    reference_phys:
        [B, N_hr, C]

    area_w_hr:
        [1, N_hr, 1]
    """

    pred_total = torch.sum(
        pred_phys * area_w_hr,
        dim=1,
        keepdim=True
    )

    ref_total = torch.sum(
        reference_phys * area_w_hr,
        dim=1,
        keepdim=True
    )

    correction = ref_total - pred_total

    corrected = pred_phys + correction

    return corrected


# ------------------------------------------------------------
# Use bilinear field as conservative reference
# ------------------------------------------------------------
ds_phys_corrected = apply_area_mean_conservation(
    pred_phys=ds_phys,
    reference_phys=bl_phys,
    area_w_hr=area_w_hr
)

# Optional physical clipping for q
ds_phys_corrected_clipped = torch.clamp(
    ds_phys_corrected,
    min=0.0
)

# ------------------------------------------------------------
# Convert back to normalized space
# ------------------------------------------------------------
direct_preds_conservative_q = (
    ds_phys_corrected - mean
) / std

direct_preds_conservative_clipped_q = (
    ds_phys_corrected_clipped - mean
) / std

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
metric = SRLoss(loss_type="mse")

conservative_metrics_q = metric.compute_metrics(
    direct_preds_conservative_q,
    direct_targets_q
)

conservative_clipped_metrics_q = metric.compute_metrics(
    direct_preds_conservative_clipped_q,
    direct_targets_q
)

original_metrics_q = metric.compute_metrics(
    direct_preds_q,
    direct_targets_q
)

print("\nNORMALIZED METRICS")
print("=" * 70)

rows = [
    {
        "Model": "DeepSphere_original",
        **original_metrics_q
    },
    {
        "Model": "DeepSphere_conservation_corrected",
        **conservative_metrics_q
    },
    {
        "Model": "DeepSphere_corrected_clipped",
        **conservative_clipped_metrics_q
    },
]

metrics_df = pd.DataFrame(rows)

display(metrics_df)

# ------------------------------------------------------------
# Conservation diagnostics
# ------------------------------------------------------------
def weighted_total_moisture(x_phys, area_w):
    return torch.sum(
        x_phys * area_w,
        dim=1
    )

total_gt = weighted_total_moisture(
    gt_phys,
    area_w_hr
)

total_bl = weighted_total_moisture(
    bl_phys,
    area_w_hr
)

total_ds = weighted_total_moisture(
    ds_phys,
    area_w_hr
)

total_ds_corr = weighted_total_moisture(
    ds_phys_corrected,
    area_w_hr
)

total_ds_corr_clip = weighted_total_moisture(
    ds_phys_corrected_clipped,
    area_w_hr
)

def total_error(pred_total, gt_total):
    rmse = torch.sqrt(
        torch.mean((pred_total - gt_total) ** 2)
    ).item()

    mae = torch.mean(
        torch.abs(pred_total - gt_total)
    ).item()

    return rmse, mae

rmse_bl, mae_bl = total_error(total_bl, total_gt)
rmse_ds, mae_ds = total_error(total_ds, total_gt)
rmse_corr, mae_corr = total_error(total_ds_corr, total_gt)
rmse_corr_clip, mae_corr_clip = total_error(total_ds_corr_clip, total_gt)

cons_rows = [
    {
        "Model": "Bilinear",
        "Total_RMSE": rmse_bl,
        "Total_MAE": mae_bl,
    },
    {
        "Model": "DeepSphere_original",
        "Total_RMSE": rmse_ds,
        "Total_MAE": mae_ds,
    },
    {
        "Model": "DeepSphere_conservation_corrected",
        "Total_RMSE": rmse_corr,
        "Total_MAE": mae_corr,
    },
    {
        "Model": "DeepSphere_corrected_clipped",
        "Total_RMSE": rmse_corr_clip,
        "Total_MAE": mae_corr_clip,
    },
]

cons_corrected_df = pd.DataFrame(cons_rows)

print("\nCONSERVATION-LIKE TOTAL MOISTURE ERROR")
print("=" * 70)

display(cons_corrected_df)

# ------------------------------------------------------------
# Non-negativity after correction
# ------------------------------------------------------------
def negative_fraction(x):
    return 100.0 * (x < 0).float().mean().item()

neg_df = pd.DataFrame([
    {
        "Model": "DeepSphere_original",
        "Negative_%": negative_fraction(ds_phys)
    },
    {
        "Model": "DeepSphere_conservation_corrected",
        "Negative_%": negative_fraction(ds_phys_corrected)
    },
    {
        "Model": "DeepSphere_corrected_clipped",
        "Negative_%": negative_fraction(ds_phys_corrected_clipped)
    },
])

print("\nNON-NEGATIVITY AFTER CORRECTION")
print("=" * 70)

display(neg_df)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
metrics_path = ARTIFACT_DIR / "conservation_corrected_deepsphere_metrics.csv"
cons_path = ARTIFACT_DIR / "conservation_corrected_total_moisture.csv"

metrics_df.to_csv(metrics_path, index=False)
cons_corrected_df.to_csv(cons_path, index=False)

print("Saved:", metrics_path)
print("Saved:", cons_path)

print("\nCELL 33 PASSED")
