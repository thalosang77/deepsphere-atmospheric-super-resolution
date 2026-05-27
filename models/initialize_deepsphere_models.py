# ============================================================
# Initialize improved local-stencil DeepSphere models
#
# Target:
#   Push MR -> HR RMSE below CNN baseline (~0.022)
#
# Strategy:
#   - stronger CNN-like local stencil branch
#   - flow branch OFF for speed/stability
#   - larger hidden dimension
#   - lower dropout
#   - bilinear residual skip retained
# ============================================================

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# Coordinates
# ------------------------------------------------------------
lon_lr_q = torch.tensor(spatial_dict_q["lon_lr"], dtype=torch.float32, device=device)
lat_lr_q = torch.tensor(spatial_dict_q["lat_lr"], dtype=torch.float32, device=device)

lon_mr_q = torch.tensor(spatial_dict_q["lon_mr"], dtype=torch.float32, device=device)
lat_mr_q = torch.tensor(spatial_dict_q["lat_mr"], dtype=torch.float32, device=device)

lon_hr_q = torch.tensor(spatial_dict_q["lon_hr"], dtype=torch.float32, device=device)
lat_hr_q = torch.tensor(spatial_dict_q["lat_hr"], dtype=torch.float32, device=device)

# ------------------------------------------------------------
# Static graph structures
# ------------------------------------------------------------
L_lr_q = spatial_dict_q["L_lr"]
L_mr_q = spatial_dict_q["L_mr"]
L_hr_q = spatial_dict_q["L_hr"]

unpool_lr_mr_q = spatial_dict_q["unpool_lr_mr"]
unpool_mr_hr_q = spatial_dict_q["unpool_mr_hr"]

# ------------------------------------------------------------
# Local stencil graph structures
# ------------------------------------------------------------
edge_index_lr_q = spatial_dict_q["edge_index_lr"]
edge_type_lr_q = spatial_dict_q["edge_type_lr"]

edge_index_mr_q = spatial_dict_q["edge_index_mr"]
edge_type_mr_q = spatial_dict_q["edge_type_mr"]

# ------------------------------------------------------------
# Flow-ready structures, kept available but flow is OFF
# ------------------------------------------------------------
edge_vectors_lr_q = spatial_dict_q["edge_vectors_lr"]
edge_distance_lr_q = spatial_dict_q["edge_distance_lr"]
t_kernel_lr_q = spatial_dict_q["t_kernel_lr"]

edge_vectors_mr_q = spatial_dict_q["edge_vectors_mr"]
edge_distance_mr_q = spatial_dict_q["edge_distance_mr"]
t_kernel_mr_q = spatial_dict_q["t_kernel_mr"]

# ------------------------------------------------------------
# Channel count
# ------------------------------------------------------------
if X_q_grid.ndim == 5:
    _, _, _, P_q, C_q = X_q_grid.shape
    in_channels = P_q * C_q
else:
    in_channels = X_q_grid.shape[-1]

out_channels = in_channels

assert in_channels == 37
assert out_channels == 37

# ------------------------------------------------------------
# Architecture controls
# ------------------------------------------------------------
hidden_dim = 96
n_resblock = 5
poly_order = 3
dropout_rate = 0.05
embed_dim = 8

# ------------------------------------------------------------
# Residual skip
# ------------------------------------------------------------
use_residual_skip = True
skip_type = "bilinear"
residual_scale_init = 0.15

# ------------------------------------------------------------
# HR geographical embedding
# ------------------------------------------------------------
use_hr_embedding = True

# ------------------------------------------------------------
# CNN-like local spherical stencil branch
# ------------------------------------------------------------
use_local_stencil = True

# ------------------------------------------------------------
# Flow-aware branch
# OFF for this run: faster and more stable.
# Re-enable later after residual-target training is confirmed.
# ------------------------------------------------------------
use_flow_blocks = False
cond_channels = in_channels

# ------------------------------------------------------------
# Config printout
# ------------------------------------------------------------
print("=" * 70)
print("MODEL CONFIG — LOCAL STENCIL FAST RUN")
print("=" * 70)

print("in_channels       :", in_channels)
print("out_channels      :", out_channels)
print("hidden_dim        :", hidden_dim)
print("n_resblock        :", n_resblock)
print("poly_order        :", poly_order)
print("dropout           :", dropout_rate)
print("embed_dim         :", embed_dim)

print("residual skip     :", use_residual_skip)
print("skip type         :", skip_type)
print("residual scale    :", residual_scale_init)

print("HR embedding      :", use_hr_embedding)
print("local stencil CNN :", use_local_stencil)
print("flow-aware blocks :", use_flow_blocks)
print("cond_channels     :", cond_channels)

print("laplacian type    :", spatial_dict_q["laplacian_type"])
print("graph type        :", spatial_dict_q["graph_type"])
print("k_transport       :", spatial_dict_q["k_transport"])

# ============================================================
# MODEL 1 — LR -> MR
# ============================================================

train_model_q = DeepSphereResNet(
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

    use_residual_skip=use_residual_skip,
    residual_scale_init=residual_scale_init,
    skip_type=skip_type,

    n_lat_low=spatial_dict_q["n_lat_lr"],
    n_lon_low=spatial_dict_q["n_lon_lr"],
    n_lat_high=spatial_dict_q["n_lat_mr"],
    n_lon_high=spatial_dict_q["n_lon_mr"],

    use_hr_embedding=use_hr_embedding,

    use_local_stencil=use_local_stencil,

    use_flow_blocks=use_flow_blocks,
    cond_channels=cond_channels,

    edge_index_low=edge_index_lr_q,
    edge_type_low=edge_type_lr_q,

    edge_vectors_low=edge_vectors_lr_q,
    edge_distance_low=edge_distance_lr_q,
    t_kernel_low=t_kernel_lr_q,
).to(device)

# ============================================================
# MODEL 2 — MR -> HR
# ============================================================

test_model_q = DeepSphereResNet(
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

    use_residual_skip=use_residual_skip,
    residual_scale_init=residual_scale_init,
    skip_type=skip_type,

    n_lat_low=spatial_dict_q["n_lat_mr"],
    n_lon_low=spatial_dict_q["n_lon_mr"],
    n_lat_high=spatial_dict_q["n_lat_hr"],
    n_lon_high=spatial_dict_q["n_lon_hr"],

    use_hr_embedding=use_hr_embedding,

    use_local_stencil=use_local_stencil,

    use_flow_blocks=use_flow_blocks,
    cond_channels=cond_channels,

    edge_index_low=edge_index_mr_q,
    edge_type_low=edge_type_mr_q,

    edge_vectors_low=edge_vectors_mr_q,
    edge_distance_low=edge_distance_mr_q,
    t_kernel_low=t_kernel_mr_q,
).to(device)

# ------------------------------------------------------------
# Parameter count
# ------------------------------------------------------------
def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

print("\nParameter count")
print("LR -> MR model:", count_parameters(train_model_q))
print("MR -> HR model:", count_parameters(test_model_q))

# ============================================================
# SANITY PASS 1 — LR -> MR
# ============================================================

batch = next(iter(train_loader_q))
hr_v, mr_v, lr_v, _, _, _ = batch

print("\n" + "=" * 70)
print("LR -> MR BATCH SHAPES")
print("=" * 70)

print("hr_v:", hr_v.shape)
print("mr_v:", mr_v.shape)
print("lr_v:", lr_v.shape)

lr_v = lr_v.to(device)
mr_v = mr_v.to(device)

with torch.no_grad():
    pred_mr = train_model_q(
        x_low=lr_v,
        lon_low=lon_lr_q,
        lat_low=lat_lr_q,
        lon_high=lon_mr_q,
        lat_high=lat_mr_q,
        cond_low=lr_v
    )

print("\nLR -> MR forward output:", pred_mr.shape)
print("Expected MR target     :", mr_v.shape)

assert pred_mr.shape == mr_v.shape
assert torch.isfinite(pred_mr).all(), "LR->MR prediction contains NaN/Inf"

# ============================================================
# SANITY PASS 2 — MR -> HR
# ============================================================

batch_direct = next(iter(train_loader_q_direct))
hr_v_d, mr_v_d, _, _ = batch_direct

print("\n" + "=" * 70)
print("DIRECT MR -> HR BATCH SHAPES")
print("=" * 70)

print("hr_v_direct:", hr_v_d.shape)
print("mr_v_direct:", mr_v_d.shape)

mr_v_d = mr_v_d.to(device)
hr_v_d = hr_v_d.to(device)

with torch.no_grad():
    pred_hr = test_model_q(
        x_low=mr_v_d,
        lon_low=lon_mr_q,
        lat_low=lat_mr_q,
        lon_high=lon_hr_q,
        lat_high=lat_hr_q,
        cond_low=mr_v_d
    )

print("\nMR -> HR forward output:", pred_hr.shape)
print("Expected HR target     :", hr_v_d.shape)

assert pred_hr.shape == hr_v_d.shape
assert torch.isfinite(pred_hr).all(), "MR->HR prediction contains NaN/Inf"

# ------------------------------------------------------------
# Residual scale diagnostics
# ------------------------------------------------------------
print("\nInitial residual scale:")

print("LR -> MR mean:", train_model_q.residual_scale.mean().item())
print("LR -> MR min :", train_model_q.residual_scale.min().item())
print("LR -> MR max :", train_model_q.residual_scale.max().item())

print("MR -> HR mean:", test_model_q.residual_scale.mean().item())
print("MR -> HR min :", test_model_q.residual_scale.min().item())
print("MR -> HR max :", test_model_q.residual_scale.max().item())

# ------------------------------------------------------------
# Local stencil diagnostics
# ------------------------------------------------------------
print("\nLocal stencil diagnostics:")

print("LR edge_index shape :", edge_index_lr_q.shape)
print("LR edge_type shape  :", edge_type_lr_q.shape)
print("MR edge_index shape :", edge_index_mr_q.shape)
print("MR edge_type shape  :", edge_type_mr_q.shape)

print("Unique LR edge types:", np.unique(edge_type_lr_q))
print("Unique MR edge types:", np.unique(edge_type_mr_q))

# ------------------------------------------------------------
# Flow diagnostics
# ------------------------------------------------------------
print("\nFlow-aware diagnostics:")
print("Flow branch active  :", use_flow_blocks)
print("LR edge vectors     :", edge_vectors_lr_q.shape)
print("MR edge vectors     :", edge_vectors_mr_q.shape)
print("LR t_kernel         :", t_kernel_lr_q)
print("MR t_kernel         :", t_kernel_mr_q)

# ------------------------------------------------------------
# GPU memory check
# ------------------------------------------------------------
if torch.cuda.is_available():
    print("\nGPU memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
    print("GPU memory reserved :", torch.cuda.memory_reserved() / 1024**3, "GB")

print("\nCELL 10 PASSED")
