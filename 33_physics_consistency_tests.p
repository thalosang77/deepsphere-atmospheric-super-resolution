# ============================================================
# Physics consistency tests
# Non-negativity + vertical smoothness
#
# Compares available models:
#   - DeepSphere temporal model, if available
#   - DeepSphere direct model, if available
#   - CNN baseline, if available
#   - Bilinear baseline
# ============================================================

import pandas as pd
import torch

print("=" * 70)
print("CELL 30 — PHYSICS CONSISTENCY TESTS")
print("=" * 70)

# ------------------------------------------------------------
# Choose DeepSphere model safely
# ------------------------------------------------------------
deepsphere_model = None
deepsphere_name = None

if "temporal_model_q" in globals():
    deepsphere_model = temporal_model_q
    deepsphere_name = "DeepSphere_temporal"

elif "test_model_q" in globals():
    deepsphere_model = test_model_q
    deepsphere_name = "DeepSphere_direct"

else:
    print("WARNING: No DeepSphere model found.")
    print("Expected one of: temporal_model_q or test_model_q")

# ------------------------------------------------------------
# Evaluate DeepSphere if available
# ------------------------------------------------------------
if deepsphere_model is not None:

    direct_metrics_q, direct_preds_q, direct_targets_q = evaluate_model_mr_to_hr(
        model=deepsphere_model,

        val_loader=val_loader_q_direct,

        val_metric=SRLoss(loss_type="mse"),

        t_lon_mr=lon_mr_q,
        t_lat_mr=lat_mr_q,

        t_lon_hr=lon_hr_q,
        t_lat_hr=lat_hr_q,

        device=device
    )

# ------------------------------------------------------------
# Bilinear baseline
# ------------------------------------------------------------
baseline_metrics_q, _, baseline_preds_q, baseline_targets_q = evaluate_bilinear_baseline(
    val_loader=val_loader_q_direct,

    val_metric=SRLoss(loss_type="mse"),

    n_lat_mr=spatial_dict_q["n_lat_mr"],
    n_lon_mr=spatial_dict_q["n_lon_mr"],

    n_lat_hr=spatial_dict_q["n_lat_hr"],
    n_lon_hr=spatial_dict_q["n_lon_hr"],

    channels=in_channels,

    device=device
)

# ------------------------------------------------------------
# CNN predictions if available
# ------------------------------------------------------------
has_cnn = "cnn_model_q" in globals()

if has_cnn:

    cnn_model_q.eval()

    cnn_preds_q = []
    cnn_targets_q = []

    with torch.no_grad():

        for batch in val_loader_q_direct:

            hr_v, mr_v, _, _ = batch

            pred = cnn_model_q(
                mr_v.to(device)
            ).float().cpu()

            cnn_preds_q.append(pred)
            cnn_targets_q.append(hr_v.float())

    cnn_preds_q = torch.cat(cnn_preds_q, dim=0)
    cnn_targets_q = torch.cat(cnn_targets_q, dim=0)

# ------------------------------------------------------------
# Select target tensor
# ------------------------------------------------------------
if deepsphere_model is not None:
    gt_norm = direct_targets_q
else:
    gt_norm = baseline_targets_q

# ------------------------------------------------------------
# Denormalize to physical q units
# ------------------------------------------------------------
mean = norm_params_q["mean"].view(1, 1, -1)
std = norm_params_q["std"].view(1, 1, -1)

gt_phys = gt_norm * std + mean
bl_phys = baseline_preds_q * std + mean

fields = {
    "GroundTruth": gt_phys,
    "Bilinear": bl_phys,
}

if deepsphere_model is not None:
    ds_phys = direct_preds_q * std + mean
    fields[deepsphere_name] = ds_phys

if has_cnn:
    cnn_phys = cnn_preds_q * std + mean
    fields["CNN"] = cnn_phys

# ------------------------------------------------------------
# Non-negativity check
# ------------------------------------------------------------
def negative_fraction(x):
    return (x < 0).float().mean().item()

neg_rows = []

for name, tensor in fields.items():

    neg_rows.append(
        {
            "Field": name,
            "Negative_%": 100.0 * negative_fraction(tensor),
            "Min_q": tensor.min().item(),
            "Max_q": tensor.max().item(),
        }
    )

neg_df = pd.DataFrame(neg_rows)

print("\n" + "=" * 70)
print("NON-NEGATIVITY CHECK")
print("=" * 70)

display(neg_df)

# ------------------------------------------------------------
# Vertical smoothness
# ------------------------------------------------------------
def vertical_smoothness(x):
    """
    x: [B, N, C], C = pressure levels.
    Computes mean absolute vertical second difference.
    """
    second_diff = x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]
    return torch.mean(torch.abs(second_diff)).item()

smooth_rows = []

for name, tensor in fields.items():

    smooth_rows.append(
        {
            "Field": name,
            "Vertical_second_diff_MAE": vertical_smoothness(tensor)
        }
    )

smooth_df = pd.DataFrame(smooth_rows)

gt_smooth = smooth_df.loc[
    smooth_df["Field"] == "GroundTruth",
    "Vertical_second_diff_MAE"
].values[0]

smooth_df["Abs_error_vs_GT"] = (
    smooth_df["Vertical_second_diff_MAE"] - gt_smooth
).abs()

smooth_df["Pct_error_vs_GT"] = (
    100.0 * smooth_df["Abs_error_vs_GT"] / gt_smooth
)

print("\n" + "=" * 70)
print("VERTICAL SMOOTHNESS CHECK")
print("=" * 70)

display(smooth_df)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\nAvailable fields compared:")
for name in fields.keys():
    print(" -", name)

print("\nCELL 30 PASSED")
