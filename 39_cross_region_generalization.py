# ============================================================
# Cross-region evaluation function
# Safe graph-to-graph parameter transfer
# Final architecture compatible
# ============================================================

import gc
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader


def load_trainable_parameters_only(target_model, source_state_dict):
    """
    Transfer only trainable parameters.
    Avoids graph-specific buffers:
      L_indices, L_values, L_shape, parent_of_child, child_order, etc.
    """

    target_params = dict(target_model.named_parameters())
    transferable = {}

    for key, value in source_state_dict.items():
        if key in target_params and target_params[key].shape == value.shape:
            transferable[key] = value

    missing, unexpected = target_model.load_state_dict(
        transferable,
        strict=False
    )

    print("Transferred trainable parameter keys:", len(transferable))
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    return target_model


def get_deepsphere_transfer_state():
    """
    Gets trained DeepSphere state dict safely.
    Uses in-memory state if available, otherwise loads checkpoint.
    """

    if "direct_state_dict_q" in globals():
        return direct_state_dict_q

    if "temporal_state_dict" in globals():
        return temporal_state_dict

    candidate_paths = [
        CKPT_DIR / "best_direct_mr_to_hr_q_residual_target_final.pt",
        CKPT_DIR / "best_direct_mr_to_hr_q_residual_target_finetuned.pt",
        CKPT_DIR / "best_temporal_generalization_train_MarJunSep_test_Dec.pt",
    ]

    for path in candidate_paths:
        if path.exists():
            print("Loading transfer checkpoint:", path)
            ckpt = torch.load(path, map_location=device)
            return ckpt["model_state_dict"]

    raise FileNotFoundError(
        "No DeepSphere state dict found. "
        "Expected direct_state_dict_q, temporal_state_dict, or a saved checkpoint."
    )


def evaluate_region_transfer(region_name, grib_file):
    print("\n" + "=" * 80)
    print(f"CROSS-REGION EVALUATION: {region_name}")
    print("=" * 80)

    gc.collect()
    torch.cuda.empty_cache()

    region = load_q_region_grib(grib_file)

    spatial_region = compute_spatial_structures(
        lat_grid=region["lat_grid"],
        lon_grid=region["lon_grid"],
        scale_factor=2,
        use_unit_sphere=True,
        save_dir="/content/deepsphere_resnet_clean/artifacts/static_structures",
        file_prefix=f"{region_name}_",
        laplacian_type="scaled_normalized",
        k_transport=4
    )

    dataset_region = DownsampleERA5(
        region["X_grid"],
        spatial_region,
        mode="val",
        clip_specific_humidity=PHYSICAL_CLIP_Q,
        q_min=Q_MIN_PHYSICAL
    )

    # True transfer test: use training-region normalization
    dataset_region.norm_params = norm_params_q

    loader_region = DataLoader(
        dataset_region,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY
    )

    lon_mr = torch.tensor(spatial_region["lon_mr"], dtype=torch.float32, device=device)
    lat_mr = torch.tensor(spatial_region["lat_mr"], dtype=torch.float32, device=device)
    lon_hr = torch.tensor(spatial_region["lon_hr"], dtype=torch.float32, device=device)
    lat_hr = torch.tensor(spatial_region["lat_hr"], dtype=torch.float32, device=device)

    model_region = DeepSphereResNet(
        l_low=spatial_region["L_mr"],
        l_high=spatial_region["L_hr"],
        unpool_matrix=spatial_region["unpool_mr_hr"],

        in_channels=in_channels,
        out_channels=out_channels,

        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_resblock=n_resblock,
        poly_order=poly_order,
        dropout_rate=dropout_rate,

        use_absolute_embedding=True,
        use_hr_embedding=use_hr_embedding,

        use_residual_skip=True,
        residual_scale_init=0.01,
        skip_type="bilinear",

        n_lat_low=spatial_region["n_lat_mr"],
        n_lon_low=spatial_region["n_lon_mr"],
        n_lat_high=spatial_region["n_lat_hr"],
        n_lon_high=spatial_region["n_lon_hr"],

        use_local_stencil=use_local_stencil,

        # Disable flow for cross-region speed/stability
        use_flow_blocks=False,
        cond_channels=None,

        edge_index_low=spatial_region["edge_index_mr"],
        edge_type_low=spatial_region["edge_type_mr"],
        edge_vectors_low=spatial_region["edge_vectors_mr"],
        edge_distance_low=spatial_region["edge_distance_mr"],
        t_kernel_low=spatial_region["t_kernel_mr"],
    ).to(device)

    source_state = get_deepsphere_transfer_state()

    model_region = load_trainable_parameters_only(
        target_model=model_region,
        source_state_dict=source_state
    )

    ds_metrics, ds_preds, ds_targets = evaluate_model_mr_to_hr(
        model=model_region,
        val_loader=loader_region,
        val_metric=SRLoss(loss_type="mse"),
        t_lon_mr=lon_mr,
        t_lat_mr=lat_mr,
        t_lon_hr=lon_hr,
        t_lat_hr=lat_hr,
        device=device
    )

    bil_metrics, _, bil_preds, bil_targets = evaluate_bilinear_baseline(
        val_loader=loader_region,
        val_metric=SRLoss(loss_type="mse"),
        n_lat_mr=spatial_region["n_lat_mr"],
        n_lon_mr=spatial_region["n_lon_mr"],
        n_lat_hr=spatial_region["n_lat_hr"],
        n_lon_hr=spatial_region["n_lon_hr"],
        channels=in_channels,
        device=device
    )

    # CNN transfer only valid if grid size matches training grid
    cnn_metrics = None

    if "cnn_model_q" in globals():

        same_grid = (
            spatial_region["n_lat_mr"] == spatial_dict_q["n_lat_mr"]
            and spatial_region["n_lon_mr"] == spatial_dict_q["n_lon_mr"]
            and spatial_region["n_lat_hr"] == spatial_dict_q["n_lat_hr"]
            and spatial_region["n_lon_hr"] == spatial_dict_q["n_lon_hr"]
        )

        if same_grid:
            cnn_preds = []
            cnn_targets = []

            cnn_model_q.eval()

            with torch.no_grad():
                for batch in loader_region:
                    hr_v, mr_v, _, _ = batch

                    pred = cnn_model_q(
                        mr_v.to(device)
                    ).float().cpu()

                    cnn_preds.append(pred)
                    cnn_targets.append(hr_v.float())

            cnn_preds = torch.cat(cnn_preds, dim=0)
            cnn_targets = torch.cat(cnn_targets, dim=0)

            cnn_metrics = SRLoss(loss_type="mse").compute_metrics(
                cnn_preds,
                cnn_targets
            )

    print("\nREGION RESULTS:", region_name)
    print("Bilinear RMSE   :", bil_metrics["RMSE"])
    print("DeepSphere RMSE :", ds_metrics["RMSE"])

    if cnn_metrics is not None:
        print("CNN RMSE        :", cnn_metrics["RMSE"])
    else:
        print("CNN RMSE        : skipped")

    result = {
        "region": region_name,

        "Bilinear_RMSE": bil_metrics["RMSE"],
        "Bilinear_MAE": bil_metrics["MAE"],
        "Bilinear_R2": bil_metrics["R2"],

        "DeepSphere_RMSE": ds_metrics["RMSE"],
        "DeepSphere_MAE": ds_metrics["MAE"],
        "DeepSphere_R2": ds_metrics["R2"],

        "DeepSphere_improvement_%": 100.0 * (
            bil_metrics["RMSE"] - ds_metrics["RMSE"]
        ) / bil_metrics["RMSE"],
    }

    if cnn_metrics is not None:
        result["CNN_RMSE"] = cnn_metrics["RMSE"]
        result["CNN_MAE"] = cnn_metrics["MAE"]
        result["CNN_R2"] = cnn_metrics["R2"]
        result["CNN_improvement_%"] = 100.0 * (
            bil_metrics["RMSE"] - cnn_metrics["RMSE"]
        ) / bil_metrics["RMSE"]

    del model_region
    gc.collect()
    torch.cuda.empty_cache()

    return result


print("CELL 37 PASSED")
