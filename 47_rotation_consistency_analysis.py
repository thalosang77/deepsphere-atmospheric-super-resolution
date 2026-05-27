# ============================================================
# CELL 44 — SEA values on fixed Europe graph/grid
#
# Tests distribution shift WITHOUT rebuilding graph geometry.
#
# IMPORTANT:
# This experiment is only valid if SEA and Europe have the same
# HR grid dimensions. If the grids differ, the cell skips safely.
# ============================================================

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

print("=" * 70)
print("CELL 44 — SEA VALUES ON FIXED EUROPE GRAPH / GRID")
print("=" * 70)

SEA_FILE = DATA_RAW_DIR / "era5_q_2025_sea_regular3h.grib"

# ------------------------------------------------------------
# Pick available DeepSphere model
# ------------------------------------------------------------
if "temporal_model_q" in globals():
    ds_fixed_model = temporal_model_q
    ds_model_name = "DeepSphere_temporal"

elif "deepsphere_model_q" in globals():
    ds_fixed_model = deepsphere_model_q
    ds_model_name = "DeepSphere"

elif "test_model_q" in globals():
    ds_fixed_model = test_model_q
    ds_model_name = "DeepSphere_direct"

else:
    raise RuntimeError(
        "No DeepSphere model found. Expected temporal_model_q, "
        "deepsphere_model_q, or test_model_q."
    )

print("Using DeepSphere model:", ds_model_name)

# ------------------------------------------------------------
# Load SEA data
# ------------------------------------------------------------
sea_region = load_q_region_grib(SEA_FILE)
X_sea_q = sea_region["X_grid"]

sea_lat = X_sea_q.shape[1]
sea_lon = X_sea_q.shape[2]

eu_lat = spatial_dict_q["n_lat_hr"]
eu_lon = spatial_dict_q["n_lon_hr"]

print("\nGrid check")
print("-" * 70)
print("SEA HR grid    :", sea_lat, sea_lon)
print("Europe HR grid :", eu_lat, eu_lon)

same_grid = (
    sea_lat == eu_lat
    and sea_lon == eu_lon
)

# ------------------------------------------------------------
# If grid dimensions differ, skip safely
# ------------------------------------------------------------
if not same_grid:

    print("\nWARNING:")
    print("SEA grid shape differs from the Europe graph/grid.")
    print("Fixed-Europe-graph evaluation is not valid without resampling.")
    print("Skipping Cell 44 evaluation.")
    print("\nUse Cell 38 cross-region transfer instead, because it rebuilds")
    print("the correct SEA graph geometry.")

    sea_fixed_graph_df = pd.DataFrame(
        [
            {
                "Experiment": "SEA_on_fixed_Europe_graph",
                "Status": "skipped",
                "Reason": "SEA grid dimensions differ from Europe graph/grid",
                "SEA_n_lat_hr": sea_lat,
                "SEA_n_lon_hr": sea_lon,
                "Europe_n_lat_hr": eu_lat,
                "Europe_n_lon_hr": eu_lon,
            }
        ]
    )

    display(sea_fixed_graph_df)

    sea_fixed_path = ARTIFACT_DIR / "sea_values_fixed_europe_graph_skipped.csv"

    sea_fixed_graph_df.to_csv(
        sea_fixed_path,
        index=False
    )

    print("Saved:", sea_fixed_path)
    print("CELL 44 PASSED — skipped safely")

# ------------------------------------------------------------
# Otherwise run valid fixed-grid distribution-shift test
# ------------------------------------------------------------
else:

    # --------------------------------------------------------
    # Build dataset using Europe graph + Europe normalization
    # --------------------------------------------------------
    sea_on_europe_dataset = DownsampleERA5(
        X_sea_q,
        spatial_dict_q,
        mode="val",
        clip_specific_humidity=PHYSICAL_CLIP_Q,
        q_min=Q_MIN_PHYSICAL
    )

    sea_on_europe_dataset.norm_params = norm_params_q

    sea_on_europe_loader = DataLoader(
        sea_on_europe_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # --------------------------------------------------------
    # DeepSphere on fixed Europe graph
    # --------------------------------------------------------
    sea_ds_metrics, sea_ds_preds, sea_ds_targets = evaluate_model_mr_to_hr(
        model=ds_fixed_model,
        val_loader=sea_on_europe_loader,
        val_metric=SRLoss(loss_type="mse"),

        t_lon_mr=lon_mr_q,
        t_lat_mr=lat_mr_q,

        t_lon_hr=lon_hr_q,
        t_lat_hr=lat_hr_q,

        device=device
    )

    # --------------------------------------------------------
    # Bilinear on fixed Europe grid
    # --------------------------------------------------------
    sea_bil_metrics, _, sea_bil_preds, sea_bil_targets = evaluate_bilinear_baseline(
        val_loader=sea_on_europe_loader,
        val_metric=SRLoss(loss_type="mse"),

        n_lat_mr=spatial_dict_q["n_lat_mr"],
        n_lon_mr=spatial_dict_q["n_lon_mr"],

        n_lat_hr=spatial_dict_q["n_lat_hr"],
        n_lon_hr=spatial_dict_q["n_lon_hr"],

        channels=in_channels,
        device=device
    )

    # --------------------------------------------------------
    # CNN fixed-grid baseline
    # --------------------------------------------------------
    sea_cnn_metrics = None

    if "cnn_model_q" in globals():

        sea_cnn_preds = []
        sea_cnn_targets = []

        cnn_model_q.eval()

        with torch.no_grad():

            for batch in sea_on_europe_loader:

                hr_v, mr_v, _, _ = batch

                pred = cnn_model_q(
                    mr_v.to(device)
                ).float().cpu()

                sea_cnn_preds.append(pred)
                sea_cnn_targets.append(hr_v.float())

        sea_cnn_preds = torch.cat(sea_cnn_preds, dim=0)
        sea_cnn_targets = torch.cat(sea_cnn_targets, dim=0)

        sea_cnn_metrics = SRLoss(loss_type="mse").compute_metrics(
            sea_cnn_preds,
            sea_cnn_targets
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------
    sea_fixed_graph_rows = [
        {
            "Model": "Bilinear_fixed_Europe_grid",
            **sea_bil_metrics
        },
        {
            "Model": f"{ds_model_name}_fixed_Europe_graph",
            **sea_ds_metrics
        },
    ]

    if sea_cnn_metrics is not None:
        sea_fixed_graph_rows.append(
            {
                "Model": "CNN_fixed_Europe_grid",
                **sea_cnn_metrics
            }
        )

    sea_fixed_graph_df = (
        pd.DataFrame(sea_fixed_graph_rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    display(sea_fixed_graph_df)

    print("\n" + "=" * 70)
    print("SEA VALUES ON FIXED EUROPE GRAPH / GRID")
    print("=" * 70)

    for _, row in sea_fixed_graph_df.iterrows():

        print(
            f"{row['Model']:>35} | "
            f"RMSE: {row['RMSE']:.6f} | "
            f"MAE: {row['MAE']:.6f} | "
            f"R2: {row['R2']:.6f}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    sea_fixed_path = ARTIFACT_DIR / "sea_values_fixed_europe_graph.csv"

    sea_fixed_graph_df.to_csv(
        sea_fixed_path,
        index=False
    )

    print("Saved:", sea_fixed_path)
    print("CELL 44 PASSED")
