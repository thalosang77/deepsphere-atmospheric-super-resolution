# ============================================================
# Pointwise win-rate: how often DeepSphere beats bilinear?
# ============================================================

model_abs_err = torch.abs(direct_preds_q - direct_targets_q)
bil_abs_err = torch.abs(baseline_preds_q - baseline_targets_q)

win_mask = model_abs_err < bil_abs_err
tie_mask = model_abs_err == bil_abs_err

win_rate = win_mask.float().mean().item()
tie_rate = tie_mask.float().mean().item()
loss_rate = 1.0 - win_rate - tie_rate

improvement = bil_abs_err - model_abs_err

print("=" * 70)
print("POINTWISE ERROR COMPARISON")
print("=" * 70)
print(f"DeepSphere win rate : {100 * win_rate:.2f}%")
print(f"Tie rate            : {100 * tie_rate:.2f}%")
print(f"Bilinear win rate   : {100 * loss_rate:.2f}%")
print(f"Mean abs-error gain : {improvement.mean().item():.6f}")
print(f"Median gain         : {improvement.median().item():.6f}")

plt.figure(figsize=(8, 5))
plt.hist(improvement.reshape(-1).numpy(), bins=100)
plt.axvline(0, linestyle="--")
plt.xlabel("Bilinear |error| - DeepSphere |error|")
plt.ylabel("Count")
plt.title("Pointwise Improvement Distribution")
plt.grid(True, alpha=0.4)
plt.show()

print("CELL 20 PASSED")
