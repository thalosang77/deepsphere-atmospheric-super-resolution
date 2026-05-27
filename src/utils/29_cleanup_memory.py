import gc
import torch

for name in [
    "test_model_q",
    "train_model_q",
    "temporal_model_q",
    "optimizer_q_direct",
    "optimizer_temporal",
    "scheduler_q_direct",
    "scheduler_temporal",
    "direct_state_dict_q",
    "temporal_state_dict",
    "history_q_direct",
    "history_temporal",
    "direct_preds_q",
    "direct_targets_q",
    "baseline_preds_q",
    "baseline_targets_q",
    "temporal_preds_q",
    "temporal_targets_q",
    "temporal_baseline_preds_q",
    "temporal_baseline_targets_q",
]:
    if name in globals():
        del globals()[name]

gc.collect()
torch.cuda.empty_cache()

print("GPU memory allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
print("GPU memory reserved :", torch.cuda.memory_reserved() / 1024**3, "GB")
