# ============================================================
# Zero-shot vs direct comparison table
# ============================================================

print("=" * 70)
print("CELL 46 — ZERO-SHOT VS DIRECT COMPARISON")
print("=" * 70)

zero_shot_rows = []

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------
def add_row(name, metrics, notes):

    zero_shot_rows.append(
        {
            "Setting": name,

            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],

            "MAPE": metrics["MAPE"],
            "Corr": metrics["Corr"],
            "PSNR": metrics["PSNR"],
            "R2": metrics["R2"],

            "Notes": notes,
        }
    )

# ------------------------------------------------------------
# Bilinear
# ------------------------------------------------------------
add_row(
    "Bilinear MR->HR",
    baseline_metrics_q,
    "Interpolation baseline"
)

# ------------------------------------------------------------
# DeepSphere direct
# ------------------------------------------------------------
if "conservative_metrics_clipped_q" in globals():

    add_row(
        "DeepSphere direct MR->HR (conservative clipped)",
        conservative_metrics_clipped_q,
        "Direct MR->HR + conservation correction + nonnegative clipping"
    )

elif "conservative_metrics_q" in globals():

    add_row(
        "DeepSphere direct MR->HR (conservative)",
        conservative_metrics_q,
        "Direct MR->HR + conservation correction"
    )

else:

    add_row(
        "DeepSphere direct MR->HR",
        direct_metrics_q,
        "Direct MR->HR"
    )

# ------------------------------------------------------------
# Zero-shot transfer
# ------------------------------------------------------------
add_row(
    "DeepSphere zero-shot LR->MR → MR->HR",
    zero_metrics,
    "Weights transferred from LR->MR model"
)

# ------------------------------------------------------------
# CNN baseline
# ------------------------------------------------------------
add_row(
    "CNN direct MR->HR",
    cnn_metrics_q,
    "Fixed-grid CNN residual baseline"
)

# ------------------------------------------------------------
# DataFrame
# ------------------------------------------------------------
zero_shot_df = (
    pd.DataFrame(zero_shot_rows)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

display(zero_shot_df)

# ------------------------------------------------------------
# Pretty ranking
# ------------------------------------------------------------
print("\n")
print("=" * 70)
print("RANKING")
print("=" * 70)

for _, row in zero_shot_df.iterrows():

    print(
        f"{row['Setting']:>45} | "
        f"RMSE: {row['RMSE']:.6f} | "
        f"MAE: {row['MAE']:.6f} | "
        f"R2: {row['R2']:.6f}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
zero_shot_path = ARTIFACT_DIR / "zero_shot_vs_direct_comparison.csv"

zero_shot_df.to_csv(
    zero_shot_path,
    index=False
)

print("\nSaved:", zero_shot_path)
print("CELL 46 PASSED")
