# ============================================================
# Gradient/front preservation diagnostics
# ============================================================

def spatial_gradients(x, n_lat, n_lon):
    """
    x: [B, N, C]
    returns dx, dy
    """
    B, N, C = x.shape
    grid = x.reshape(B, n_lat, n_lon, C)

    dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]

    return dx, dy


dx_gt, dy_gt = spatial_gradients(direct_targets_q, n_lat_hr, n_lon_hr)
dx_model, dy_model = spatial_gradients(direct_preds_q, n_lat_hr, n_lon_hr)
dx_bil, dy_bil = spatial_gradients(baseline_preds_q, n_lat_hr, n_lon_hr)

grad_rmse_model = torch.sqrt(
    torch.mean((dx_model - dx_gt) ** 2) +
    torch.mean((dy_model - dy_gt) ** 2)
)

grad_rmse_bil = torch.sqrt(
    torch.mean((dx_bil - dx_gt) ** 2) +
    torch.mean((dy_bil - dy_gt) ** 2)
)

grad_improvement = 100 * (grad_rmse_bil.item() - grad_rmse_model.item()) / grad_rmse_bil.item()

print("=" * 70)
print("GRADIENT / FRONT PRESERVATION")
print("=" * 70)
print(f"Gradient RMSE DeepSphere : {grad_rmse_model.item():.6f}")
print(f"Gradient RMSE Bilinear   : {grad_rmse_bil.item():.6f}")
print(f"Improvement              : {grad_improvement:.2f}%")

# Visualize gradient magnitude for one sample/channel
sample_idx = 0
pressure_idx = 20

def grad_mag_single(x):
    field = x[sample_idx, :, pressure_idx].reshape(n_lat_hr, n_lon_hr)
    gy = field[1:, :] - field[:-1, :]
    gx = field[:, 1:] - field[:, :-1]

    gx_crop = gx[:-1, :]
    gy_crop = gy[:, :-1]
    return torch.sqrt(gx_crop ** 2 + gy_crop ** 2).numpy()

gm_gt = grad_mag_single(direct_targets_q)
gm_model = grad_mag_single(direct_preds_q)
gm_bil = grad_mag_single(baseline_preds_q)

vmax = max(gm_gt.max(), gm_model.max(), gm_bil.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

for ax, data, title in zip(
    axes,
    [gm_gt, gm_bil, gm_model],
    ["GT Gradient Magnitude", "Bilinear Gradient Magnitude", "DeepSphere Gradient Magnitude"]
):
    im = ax.imshow(data, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.show()

print("CELL 22 PASSED")
