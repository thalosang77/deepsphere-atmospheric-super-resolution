# Reload DeepSphere model before Cell 30

deepsphere_model_q = DeepSphereResNet(
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
    use_residual_skip=True,
    residual_scale_init=0.01,
    skip_type="bilinear",
    n_lat_low=spatial_dict_q["n_lat_mr"],
    n_lon_low=spatial_dict_q["n_lon_mr"],
    n_lat_high=spatial_dict_q["n_lat_hr"],
    n_lon_high=spatial_dict_q["n_lon_hr"],
    use_hr_embedding=use_hr_embedding,
    use_local_stencil=use_local_stencil,
    use_flow_blocks=False,
    cond_channels=None,
    edge_index_low=edge_index_mr_q,
    edge_type_low=edge_type_mr_q,
    edge_vectors_low=edge_vectors_mr_q,
    edge_distance_low=edge_distance_mr_q,
    t_kernel_low=t_kernel_mr_q,
).to(device)

ckpt = torch.load(
    CKPT_DIR / "best_temporal_generalization_train_MarJunSep_test_Dec.pt",
    map_location=device
)

deepsphere_model_q.load_state_dict(ckpt["model_state_dict"])
temporal_model_q = deepsphere_model_q
