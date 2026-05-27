# ============================================================
# FINAL PAPER SUMMARY TABLE
# Correct conservation metric version
# ============================================================

import pandas as pd
import numpy as np

print("=" * 70)
print("CELL 47 — FINAL PAPER SUMMARY TABLE")
print("=" * 70)

# ------------------------------------------------------------
# Load region table
# ------------------------------------------------------------
region_df = pd.read_csv(
    "/content/deepsphere_resnet_clean/artifacts/multi_region_transfer_results.csv"
)

# ------------------------------------------------------------
# Conservation values from Cell 33
# ------------------------------------------------------------
ds_corr_row = cons_corrected_df[
    cons_corrected_df["Model"] == "DeepSphere_conservation_corrected"
]

ds_clip_row = cons_corrected_df[
    cons_corrected_df["Model"] == "DeepSphere_corrected_clipped"
]

corrected_total_rmse = ds_clip_row["Total_RMSE"].values[0]
corrected_total_mae = ds_clip_row["Total_MAE"].values[0]

# ------------------------------------------------------------
# Seam ratios
# ------------------------------------------------------------
cnn_seam = seam_df[
    seam_df["Model"].str.contains("CNN", case=False, na=False)
]

ds_seam = seam_df[
    seam_df["Model"].str.contains("DeepSphere", case=False, na=False)
]

cnn_seam_ratio = cnn_seam["RMSE_seam_to_interior_ratio"].values[0]
ds_seam_ratio = ds_seam["RMSE_seam_to_interior_ratio"].values[0]

# ------------------------------------------------------------
# Rotation averages
# ------------------------------------------------------------
cnn_rotation_avg = rotation_df["CNN_rotation_RMSE"].mean()
ds_rotation_avg = rotation_df["DeepSphere_rotation_RMSE"].mean()

rotation_winner = (
    "DeepSphere lower rotation RMSE"
    if ds_rotation_avg < cnn_rotation_avg
    else "CNN slightly lower rotation RMSE"
)

# ------------------------------------------------------------
# Cross-region averages
# ------------------------------------------------------------
cnn_region_avg = region_df["CNN_RMSE"].mean()
ds_region_avg = region_df["DeepSphere_RMSE"].mean()

# ------------------------------------------------------------
# Final summary table
# ------------------------------------------------------------
final_summary = pd.DataFrame([
    {
        "Experiment": "Direct MR→HR raw accuracy",
        "CNN": cnn_metrics_q["RMSE"],
        "DeepSphere": conservative_clipped_metrics_q["RMSE"],
        "Metric": "RMSE",
        "Winner": "CNN",
        "Interpretation": "CNN gives lowest pointwise error on fixed regular grid.",
    },
    {
        "Experiment": "Zero-shot LR→MR→HR transfer",
        "CNN": np.nan,
        "DeepSphere": zero_metrics["RMSE"],
        "Metric": "RMSE",
        "Winner": "DeepSphere only",
        "Interpretation": "Graph filters transfer across resolutions without direct MR→HR training.",
    },
    {
        "Experiment": "Rotation consistency",
        "CNN": cnn_rotation_avg,
        "DeepSphere": ds_rotation_avg,
        "Metric": "Average rotation RMSE",
        "Winner": rotation_winner,
        "Interpretation": "Both models show comparable robustness under approximate spherical rotations.",
    },
    {
        "Experiment": "Cross-region transfer",
        "CNN": cnn_region_avg,
        "DeepSphere": ds_region_avg,
        "Metric": "Average RMSE",
        "Winner": "CNN",
        "Interpretation": "Both improve over bilinear; CNN remains lower in RMSE.",
    },
    {
        "Experiment": "Physical conservation after correction",
        "CNN": np.nan,
        "DeepSphere": corrected_total_rmse,
        "Metric": "Area-total moisture RMSE",
        "Winner": "DeepSphere",
        "Interpretation": "Conservation correction brings DeepSphere to near-bilinear conservation.",
    },
    {
        "Experiment": "Seam stability",
        "CNN": cnn_seam_ratio,
        "DeepSphere": ds_seam_ratio,
        "Metric": "Seam/interior RMSE ratio, ideal=1",
        "Winner": "DeepSphere",
        "Interpretation": "DeepSphere is closer to ideal seam stability.",
    },
    {
        "Experiment": "Spectral slope realism",
        "CNN": slope_df.loc[slope_df["Model"] == "CNN", "Abs_slope_error"].values[0],
        "DeepSphere": slope_df.loc[slope_df["Model"] == "DeepSphere", "Abs_slope_error"].values[0],
        "Metric": "Absolute spectral slope error",
        "Winner": "DeepSphere",
        "Interpretation": "DeepSphere best preserves the ground-truth spectral scaling.",
    },
])

display(final_summary)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
final_summary_path = ARTIFACT_DIR / "final_paper_summary_table.csv"

final_summary.to_csv(
    final_summary_path,
    index=False
)

print("Saved:", final_summary_path)
print("CELL 47 PASSED")
