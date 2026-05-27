# ============================================================
# CNN baseline: bilinear upsample + residual CNN
# ============================================================

class ResidualCNNBaseline(nn.Module):
    def __init__(self, channels=37, hidden=96, depth=6):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(channels, hidden, kernel_size=3, padding=1))
        layers.append(nn.GELU())

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(hidden, hidden, kernel_size=3, padding=1))
            layers.append(nn.GELU())

        layers.append(nn.Conv2d(hidden, channels, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, mr_v):
        B, N, C = mr_v.shape

        x = mr_v.reshape(B, spatial_dict_q["n_lat_mr"], spatial_dict_q["n_lon_mr"], C)
        x = x.permute(0, 3, 1, 2)

        x_up = F.interpolate(
            x,
            size=(spatial_dict_q["n_lat_hr"], spatial_dict_q["n_lon_hr"]),
            mode="bilinear",
            align_corners=False
        )

        residual = self.net(x_up)
        y = x_up + self.residual_scale.to(residual.dtype) * residual

        y = y.permute(0, 2, 3, 1).reshape(
            B,
            spatial_dict_q["n_hr"],
            C
        )

        return y


def train_cnn_baseline(
    model,
    train_loader,
    val_loader,
    epochs=50,
    lr=5e-4,
    weight_decay=1e-5
):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )
    criterion = SRLoss(loss_type="combined")
    metric = SRLoss(loss_type="mse")

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_rmse = float("inf")
    best_state = None
    history = {"train_loss": [], "val_RMSE": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for batch in train_loader:
            hr_v, mr_v, _, _ = batch
            inputs = mr_v.to(device, non_blocking=True)
            targets = hr_v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pred = model(inputs)
                loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        preds, targets_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                hr_v, mr_v, _, _ = batch
                inputs = mr_v.to(device, non_blocking=True)
                targets = hr_v.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    pred = model(inputs)

                preds.append(pred.float().cpu())
                targets_all.append(targets.float().cpu())

        preds = torch.cat(preds, dim=0)
        targets_all = torch.cat(targets_all, dim=0)

        metrics = metric.compute_metrics(preds, targets_all)
        scheduler.step(metrics["RMSE"])

        history["train_loss"].append(avg_loss)
        history["val_RMSE"].append(metrics["RMSE"])

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch+1:03d}/{epochs:03d} | "
            f"Loss {avg_loss:.6f} | "
            f"Val RMSE {metrics['RMSE']:.6f} | "
            f"R2 {metrics['R2']:.6f} | "
            f"{time.time()-start:.2f}s"
        )

    model.load_state_dict(best_state)
    return history, best_state


cnn_model_q = ResidualCNNBaseline(
    channels=in_channels,
    hidden=96,
    depth=6
).to(device)

cnn_history_q, cnn_state_q = train_cnn_baseline(
    cnn_model_q,
    train_loader_q_direct,
    val_loader_q_direct,
    epochs=80,
    lr=5e-4,
    weight_decay=1e-5
)

cnn_model_q.load_state_dict(cnn_state_q)

cnn_metrics_q, cnn_preds_q, cnn_targets_q = None, [], []

cnn_model_q.eval()
with torch.no_grad():
    for batch in val_loader_q_direct:
        hr_v, mr_v, _, _ = batch
        pred = cnn_model_q(mr_v.to(device)).float().cpu()
        cnn_preds_q.append(pred)
        cnn_targets_q.append(hr_v.float())

cnn_preds_q = torch.cat(cnn_preds_q, dim=0)
cnn_targets_q = torch.cat(cnn_targets_q, dim=0)

cnn_metrics_q = SRLoss(loss_type="mse").compute_metrics(cnn_preds_q, cnn_targets_q)

print("\nCNN BASELINE METRICS")
for k, v in cnn_metrics_q.items():
    print(f"{k}: {v:.6f}")

print("CELL 29 PASSED")
