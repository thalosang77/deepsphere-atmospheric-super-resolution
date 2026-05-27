# ============================================================
# GLOBAL PRED/TARGET DEFINITIONS (ROBUST)
# Ensures all downstream cells use the final DeepSphere output
# ============================================================

import torch

# ------------------------------------------------------------
# DeepSphere predictions
# Priority:
#   1. conservation-corrected + clipped
#   2. conservation-corrected
#   3. raw DeepSphere
# ------------------------------------------------------------
if "direct_preds_conservative_clipped_q" in globals():
    final_deepsphere_preds_q = direct_preds_conservative_clipped_q
    print("Using conservation-corrected + clipped DeepSphere predictions.")

elif "direct_preds_conservative_q" in globals():
    final_deepsphere_preds_q = direct_preds_conservative_q
    print("Using conservation-corrected DeepSphere predictions.")

elif "direct_preds_q" in globals():
    final_deepsphere_preds_q = direct_preds_q
    print("Using raw DeepSphere predictions.")

else:
    raise RuntimeError("No DeepSphere predictions found.")

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
if "direct_targets_q" in globals():
    final_deepsphere_targets_q = direct_targets_q

elif "baseline_targets_q" in globals():
    final_deepsphere_targets_q = baseline_targets_q

elif "cnn_targets_q" in globals():
    final_deepsphere_targets_q = cnn_targets_q

else:
    raise RuntimeError("No target tensor found.")

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
print("DeepSphere preds shape :", final_deepsphere_preds_q.shape)
print("Targets shape          :", final_deepsphere_targets_q.shape)

assert final_deepsphere_preds_q.shape == final_deepsphere_targets_q.shape
assert torch.isfinite(final_deepsphere_preds_q).all()
assert torch.isfinite(final_deepsphere_targets_q).all()

print("GLOBAL DEFINITIONS READY")
