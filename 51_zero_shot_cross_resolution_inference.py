# ------------------------------------------------------------
# ZERO-SHOT: transfer LR->MR learned weights into MR->HR model
# ------------------------------------------------------------

zero_model = DeepSphereResNet(
    l_low=L_mr_q,
    l_high=L_hr_q,
    unpool_matrix=unpool_mr_hr_q,

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
    residual_scale_init=0.02,
    skip_type="bilinear",

    n_lat_low=spatial_dict_q["n_lat_mr"],
    n_lon_low=spatial_dict_q["n_lon_mr"],
    n_lat_high=spatial_dict_q["n_lat_hr"],
    n_lon_high=spatial_dict_q["n_lon_hr"],

    use_local_stencil=use_local_stencil,

    use_flow_blocks=False,
    cond_channels=None,

    edge_index_low=spatial_dict_q["edge_index_mr"],
    edge_type_low=spatial_dict_q["edge_type_mr"],
    edge_vectors_low=spatial_dict_q["edge_vectors_mr"],
    edge_distance_low=spatial_dict_q["edge_distance_mr"],
    t_kernel_low=spatial_dict_q["t_kernel_mr"],
).to(device)

# Transfer only trainable compatible weights
zero_model = transfer_weights_only(
    zero_model,
    zero_state_dict
)

zero_metrics, zero_preds_q, zero_targets_q = evaluate_model_mr_to_hr(
    model=zero_model,
    val_loader=val_loader_q_direct,
    val_metric=SRLoss("mse"),

    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,
    t_lon_hr=lon_hr_q,
    t_lat_hr=lat_hr_q,

    device=device
)

print("=" * 70)
print("ZERO-SHOT LR->MR TRANSFER TO MR->HR")
print("=" * 70)

for k, v in zero_metrics.items():
    print(f"{k}: {v:.6f}")

print("ZERO-SHOT RMSE:", zero_metrics["RMSE"])
