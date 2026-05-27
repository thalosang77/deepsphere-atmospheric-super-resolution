# ============================================================
# Environment setup, imports, reproducibility checks
# ============================================================

import os
import sys
import json
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path("/content/deepsphere_resnet_clean")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
STATIC_DIR = ARTIFACT_DIR / "static_structures"
FIG_DIR = PROJECT_ROOT / "figures"
CKPT_DIR = PROJECT_ROOT / "checkpoints"

for d in [PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, ARTIFACT_DIR, STATIC_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Global config
# -----------------------------
CONFIG = {
    "seed": SEED,
    "device": str(device),

    "project_root": str(PROJECT_ROOT),
    "data_dir": str(DATA_DIR),
    "raw_data_dir": str(RAW_DATA_DIR),
    "artifact_dir": str(ARTIFACT_DIR),
    "static_dir": str(STATIC_DIR),
    "figure_dir": str(FIG_DIR),
    "checkpoint_dir": str(CKPT_DIR),

    # data / geometry
    "variable": "specific_humidity",
    "short_name": "q",
    "scale_factor": 2,
    "use_unit_sphere": True,
    "sphere_radius_km": 6371.0,

    # q experiment after ERA5 pressure-level reshape
    "in_channels": 37,
    "out_channels": 37,

    # training defaults
    "batch_size": 16,
    "num_workers": 0,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs_smoke": 5,
    "epochs_full": 100,

    # model defaults
    "hidden_channels": 64,
    "poly_order": 3,
    "num_resblocks": 2,
    "dropout_rate": 0.3,
    "geo_embed_dim": 8,
}

with open(PROJECT_ROOT / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# -----------------------------
# Hard-fail environment checks
# -----------------------------
print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)
print(f"Python version   : {sys.version.split()[0]}")
print(f"PyTorch version  : {torch.__version__}")
print(f"NumPy version    : {np.__version__}")
print(f"Pandas version   : {pd.__version__}")
print(f"Xarray version   : {xr.__version__}")
print(f"Device           : {device}")
print(f"CUDA available   : {torch.cuda.is_available()}")

gpu_name = None
gpu_memory_gb = None

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU name         : {gpu_name}")
    print(f"GPU memory       : {gpu_memory_gb:.2f} GB")
else:
    print("WARNING: running on CPU. Full training will be slow.")

# -----------------------------
# cfgrib registration check
# -----------------------------
available_engines = list(xr.backends.list_engines().keys())
print(f"Xarray engines   : {available_engines}")

if "cfgrib" not in available_engines:
    print("WARNING: cfgrib engine not registered. Install/restart before opening GRIB files.")

# -----------------------------
# Tensor sanity check
# -----------------------------
x = torch.randn(2, 16, CONFIG["in_channels"], device=device)
assert x.shape == (2, 16, CONFIG["in_channels"]), "Tensor shape sanity check failed"
assert torch.isfinite(x).all(), "Tensor contains NaN/Inf"

# -----------------------------
# Audit record
# -----------------------------
audit = {
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "xarray_version": xr.__version__,
    "xarray_engines": available_engines,
    "device": str(device),
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": gpu_name,
    "gpu_memory_gb": gpu_memory_gb,
    "seed": SEED,
    "config": CONFIG,
}

with open(ARTIFACT_DIR / "environment_audit.json", "w") as f:
    json.dump(audit, f, indent=2)

print("\nBasic tensor test passed.")
print("Config saved to:", PROJECT_ROOT / "config.json")
print("Audit saved to :", ARTIFACT_DIR / "environment_audit.json")
print("\nCELL 1 PASSED")
