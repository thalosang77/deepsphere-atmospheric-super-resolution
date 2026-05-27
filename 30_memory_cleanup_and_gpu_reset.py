# ============================================================
# Fast ablation study
# ============================================================

import gc
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

gc.collect()
torch.cuda.empty_cache()

ABLATION_BATCH_SIZE = 2
ABLATION_EPOCHS = 5
ABLATION_TRAIN_SAMPLES = 40
ABLATION_VAL_SAMPLES = 16

ablation_train_subset = Subset(
    train_dataset_q_direct,
    list(range(min(ABLATION_TRAIN_SAMPLES, len(train_dataset_q_direct))))
)

ablation_val_subset = Subset(
    val_dataset_q_direct,
    list(range(min(ABLATION_VAL_SAMPLES, len(val_dataset_q_direct))))
)

ablation_train_loader = DataLoader(
    ablation_train_subset,
    batch_size=ABLATION_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

ablation_val_loader = DataLoader(
    ablation_val_subset,
    batch_size=ABLATION_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

ablation_configs = [
    {
        "name": "Full_model",
        "use_local_stencil": True,
        "use_flow_blocks": True,
        "use_absolute_embedding": True,
        "use_hr_embedding": True,
        "use_residual_skip": True,
        "skip_type": "bilinear",
    },
    {
        "name": "No_flow_blocks",
        "use_local_stencil": True,
        "use_flow_blocks": False,
        "use_absolute_embedding": True,
        "use_hr_embedding": True,
        "use_residual_skip": True,
        "skip_type": "bilinear",
    },
    {
        "name": "No_local_stencil",
        "use_local_stencil": False,
        "use_flow_blocks": False,
        "use_absolute_embedding": True,
        "use_hr_embedding": True,
        "use_residual_skip": True,
        "skip_type": "bilinear",
    },
    {
        "name": "No_geo_embedding",
        "use_local_stencil": True,
        "use_flow_blocks": False,
        "use_absolute_embedding": False,
        "use_hr_embedding": True,
        "use_residual_skip": True,
        "skip_type": "bilinear",
    },
    {
        "name": "Nearest_skip",
        "use_local_stencil": True,
        "use_flow_blocks": False,
        "use_absolute_embedding": True,
        "use_hr_embedding": True,
        "use_residual_skip": True,
        "skip_type": "nearest",
    },
]

def make_ablation_model(cfg):
    return DeepSphereResNet(
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

        use_absolute_embedding=cfg["use_absolute_embedding"],
        use_hr_embedding=cfg["use_hr_embedding"],

        use_residual_skip=cfg["use_residual_skip"],
        residual_scale_init=0.01,
        skip_type=cfg["skip_type"],

        n_lat_low=spatial_dict_q["n_lat_mr"],
        n_lon_low=spatial_dict_q["n_lon_mr"],
        n_lat_high=spatial_dict_q["n_lat_hr"],
        n_lon_high=spatial_dict_q["n_lon_hr"],

        use_local_stencil=cfg["use_local_stencil"],

        use_flow_blocks=cfg["use_flow_blocks"],
        cond_channels=cond_channels if cfg["use_flow_blocks"] else None,

        edge_index_low=edge_index_mr_q,
        edge_type_low=edge_type_mr_q,
        edge_vectors_low=edge_vectors_mr_q,
        edge_distance_low=edge_distance_mr_q,
        t_kernel_low=t_kernel_mr_q,
    ).to(device)

ablation_results = []

for cfg in ablation_configs:

    print("\n" + "=" * 70)
    print("FAST ABLATION:", cfg["name"])
    print("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()

    model = make_ablation_model(cfg)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=5e-7,
        betas=(0.9, 0.99)
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=ABLATION_EPOCHS,
        eta_min=5e-6
    )

    hist, state = training_direct_mr_to_hr(
        train_model=model,
        train_loader=ablation_train_loader,
        val_loader=ablation_val_loader,
        optimizer=optimizer,
        criterion=SRLoss(loss_type="mse"),
        val_metric=SRLoss(loss_type="mse"),
        t_lon_mr=lon_mr_q,
        t_lat_mr=lat_mr_q,
        t_lon_hr=lon_hr_q,
        t_lat_hr=lat_hr_q,
        epochs=ABLATION_EPOCHS,
        device=device,
        scheduler=scheduler,
        ckpt_path=None,
        run_name=f"fast_ablation_{cfg['name']}"
    )

    best_rmse = min(hist["val_RMSE"])
    best_epoch = int(torch.tensor(hist["val_RMSE"]).argmin().item()) + 1

    ablation_results.append({
        "Variant": cfg["name"],
        "Best_RMSE": best_rmse,
        "Best_epoch": best_epoch,
        "use_local_stencil": cfg["use_local_stencil"],
        "use_flow_blocks": cfg["use_flow_blocks"],
        "use_absolute_embedding": cfg["use_absolute_embedding"],
        "use_hr_embedding": cfg["use_hr_embedding"],
        "skip_type": cfg["skip_type"],
    })

    del model, optimizer, scheduler, hist, state
    gc.collect()
    torch.cuda.empty_cache()

ablation_df = pd.DataFrame(ablation_results).sort_values("Best_RMSE").reset_index(drop=True)

display(ablation_df)

ablation_path = ARTIFACT_DIR / "fast_ablation_results_q_regular3h.csv"
ablation_df.to_csv(ablation_path, index=False)

print("Saved:", ablation_path)
print("CELL 28 PASSED")
