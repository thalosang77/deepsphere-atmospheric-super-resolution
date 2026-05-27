# ------------------------------------------------------------
# TRAIN: LR -> MR zero-shot source model
# Final architecture-compatible version
# ------------------------------------------------------------

train_model_zero = DeepSphereResNet(
    l_low=L_lr_q,
    l_high=L_mr_q,
    unpool_matrix=unpool_lr_mr_q,

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

    n_lat_low=spatial_dict_q["n_lat_lr"],
    n_lon_low=spatial_dict_q["n_lon_lr"],
    n_lat_high=spatial_dict_q["n_lat_mr"],
    n_lon_high=spatial_dict_q["n_lon_mr"],

    use_local_stencil=use_local_stencil,

    use_flow_blocks=False,
    cond_channels=None,

    edge_index_low=spatial_dict_q["edge_index_lr"],
    edge_type_low=spatial_dict_q["edge_type_lr"],
    edge_vectors_low=spatial_dict_q["edge_vectors_lr"],
    edge_distance_low=spatial_dict_q["edge_distance_lr"],
    t_kernel_low=spatial_dict_q["t_kernel_lr"],
).to(device)

optimizer_zero = torch.optim.AdamW(
    train_model_zero.parameters(),
    lr=4e-4,
    weight_decay=5e-7,
    betas=(0.9, 0.99)
)

scheduler_zero = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_zero,
    T_max=70,
    eta_min=5e-6
)

ckpt_path_zero = CKPT_DIR / "best_lr_to_mr_zero_source.pt"

history_zero, zero_state_dict = training_lr_to_mr(
    train_model=train_model_zero,
    train_loader=train_loader_q,
    val_loader=val_loader_q,

    optimizer=optimizer_zero,

    criterion=SRLoss("mse"),
    val_metric=SRLoss("mse"),

    t_lon_lr=lon_lr_q,
    t_lat_lr=lat_lr_q,
    t_lon_mr=lon_mr_q,
    t_lat_mr=lat_mr_q,

    epochs=70,
    device=device,

    scheduler=scheduler_zero,
    ckpt_path=ckpt_path_zero,
    run_name="lr_to_mr_zero_source_final"
)

train_model_zero.load_state_dict(zero_state_dict)

print("LR -> MR zero-shot source training complete.")
print("Best LR->MR RMSE:", min(history_zero["val_RMSE"]))
print("Checkpoint:", ckpt_path_zero)
