# ============================================================
# CDataset + normalization
# Physics-aware q super-resolution dataset
#
# Supports:
#   mode="train" -> LR -> MR training
#   mode="val"   -> MR -> HR evaluation / testing
# ============================================================

import json
import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


# ------------------------------------------------------------
# Physical treatment for specific humidity
# ------------------------------------------------------------
PHYSICAL_CLIP_Q = True
Q_MIN_PHYSICAL = 0.0


def maybe_clip_specific_humidity(x, clip=True, q_min=0.0):
    """
    Specific humidity should be non-negative.
    ERA5 may contain tiny negative numerical artifacts.
    """
    if clip:
        return np.maximum(x, q_min)
    return x


# ------------------------------------------------------------
# Flatten pressure/variable tensor
# ------------------------------------------------------------
def flatten_pressure_variable_channels(X):
    """
    Input:
        X: [T, Y, X, P, C]

    Output:
        X_flat: [T, Y, X, P*C]
    """
    assert X.ndim == 5, f"Expected [T, Y, X, P, C], got {X.shape}"

    T, Y, Xdim, P, C = X.shape

    X_flat = X.reshape(T, Y, Xdim, P * C)

    assert np.isfinite(X_flat).all()

    return X_flat.astype(np.float32)


# ------------------------------------------------------------
# Compute normalization from MR training data only
# ------------------------------------------------------------
def compute_norm_params(dataloader, channels, save_path=None):
    """
    Compute per-channel z-score normalization from MR training data only.
    This prevents validation/test leakage.
    """

    print("=" * 70)
    print("COMPUTING NORMALIZATION FROM TRAINING MR DATA ONLY")
    print("=" * 70)

    mr_sum = torch.zeros(channels, dtype=torch.float64)
    mr_sum_sq = torch.zeros(channels, dtype=torch.float64)

    mr_min = torch.full((channels,), float("inf"), dtype=torch.float64)
    mr_max = torch.full((channels,), -float("inf"), dtype=torch.float64)

    mr_count = 0

    for batch in dataloader:

        _, mr_v, _, _, _, _ = batch

        assert mr_v.ndim == 3, (
            f"Expected MR tensor [B, N, C], got {mr_v.shape}"
        )

        B, N, C = mr_v.shape

        assert C == channels, (
            f"Channel mismatch: got {C}, expected {channels}"
        )

        mr_v64 = mr_v.double()

        mr_sum += mr_v64.sum(dim=(0, 1))
        mr_sum_sq += (mr_v64 ** 2).sum(dim=(0, 1))

        mr_min = torch.minimum(
            mr_min,
            mr_v64.amin(dim=(0, 1))
        )

        mr_max = torch.maximum(
            mr_max,
            mr_v64.amax(dim=(0, 1))
        )

        mr_count += B * N

    assert mr_count > 0, "No MR samples found"

    mean = mr_sum / mr_count
    var = mr_sum_sq / mr_count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-10))

    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)

    norm_params = {
        "method": "zscore",
        "mean": mean.float().view(1, 1, -1),
        "std": std.float().view(1, 1, -1),
        "min": mr_min.float().view(1, 1, -1),
        "max": mr_max.float().view(1, 1, -1),
        "count": int(mr_count),
    }

    print("Normalization complete.")
    print("mean shape      :", norm_params["mean"].shape)
    print("std shape       :", norm_params["std"].shape)
    print("std min/max     :", std.min().item(), std.max().item())
    print("MR value min/max:", mr_min.min().item(), mr_max.max().item())

    if save_path is not None:

        serializable = {
            "method": "zscore",
            "mean": norm_params["mean"].cpu().numpy().tolist(),
            "std": norm_params["std"].cpu().numpy().tolist(),
            "min": norm_params["min"].cpu().numpy().tolist(),
            "max": norm_params["max"].cpu().numpy().tolist(),
            "count": int(mr_count),
        }

        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print("Saved normalization params to:", save_path)

    return norm_params


# ------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------
class DownsampleERA5(Dataset):
    """
    Dataset for spherical atmospheric SR.

    Expected input:
        X_raw: [T, Y, X, P, C] or [T, Y, X, C_flat]

    Internally:
        [T, Y, X, C_flat]

    mode="train":
        HR -> MR -> LR
        returns:
            hr_vertices, mr_vertices, lr_vertices,
            hr_raw, mr_raw, lr_raw

        Used for LR -> MR training.

    mode="val":
        HR -> MR
        returns:
            hr_vertices, mr_vertices,
            hr_raw, mr_raw

        Used for MR -> HR evaluation.
    """

    def __init__(
        self,
        X_raw,
        spatial_dict,
        mode="train",
        clip_specific_humidity=True,
        q_min=0.0,
    ):

        assert mode in ["train", "val"], (
            f"Unsupported mode: {mode}"
        )

        # Accept [T, Y, X, P, C]
        if X_raw.ndim == 5:
            X_raw = flatten_pressure_variable_channels(X_raw)

        assert X_raw.ndim == 4, (
            f"Expected X_raw [T, Y, X, C], got {X_raw.shape}"
        )

        X_raw = X_raw.astype(np.float32)

        X_raw = maybe_clip_specific_humidity(
            X_raw,
            clip=clip_specific_humidity,
            q_min=q_min
        )

        assert np.isfinite(X_raw).all(), (
            "X_raw contains NaN/Inf"
        )

        self.X = X_raw
        self.timesteps = X_raw.shape[0]
        self.channels = X_raw.shape[-1]

        self.spatial = spatial_dict
        self.mode = mode

        # ----------------------------------------------------
        # Required spatial keys
        # ----------------------------------------------------
        required_keys = [
            "n_hr", "n_mr", "n_lr",
            "n_lat_hr", "n_lon_hr",
            "n_lat_mr", "n_lon_mr",
            "n_lat_lr", "n_lon_lr",
            "pool_hr_mr",
            "pool_mr_lr",
        ]

        for key in required_keys:
            assert key in spatial_dict, (
                f"Missing spatial_dict key: {key}. "
                "CELL 5 must produce HR, MR and LR hierarchy."
            )

        self.n_hr = spatial_dict["n_hr"]
        self.n_mr = spatial_dict["n_mr"]
        self.n_lr = spatial_dict["n_lr"]

        self.n_lat_hr = spatial_dict["n_lat_hr"]
        self.n_lon_hr = spatial_dict["n_lon_hr"]

        self.n_lat_mr = spatial_dict["n_lat_mr"]
        self.n_lon_mr = spatial_dict["n_lon_mr"]

        self.n_lat_lr = spatial_dict["n_lat_lr"]
        self.n_lon_lr = spatial_dict["n_lon_lr"]

        assert self.X.shape[1] >= self.n_lat_hr
        assert self.X.shape[2] >= self.n_lon_hr

        self.pool_hr_mr = self._sparse_to_torch(
            spatial_dict["pool_hr_mr"]
        ).coalesce()

        self.pool_mr_lr = self._sparse_to_torch(
            spatial_dict["pool_mr_lr"]
        ).coalesce()

        self.norm_params = None

    def _sparse_to_torch(self, sp_mat):

        sp_mat = sp_mat.tocoo()

        indices = torch.LongTensor(
            np.vstack((sp_mat.row, sp_mat.col))
        )

        values = torch.FloatTensor(sp_mat.data)

        size = tuple(sp_mat.shape)

        return torch.sparse_coo_tensor(
            indices,
            values,
            size
        )

    def downsample_features_3d(self, data_3d, pool_matrix):
        """
        data_3d:
            [Y, X, C]

        pool_matrix:
            sparse [N_low, N_high]
        """

        assert data_3d.ndim == 3, (
            f"Expected [Y, X, C], got {data_3d.shape}"
        )

        n_lat, n_lon, channels = data_3d.shape
        n_high = n_lat * n_lon

        assert pool_matrix.shape[1] == n_high, (
            f"Pool matrix expects {pool_matrix.shape[1]} high nodes, "
            f"got {n_high}"
        )

        high_vertices = data_3d.reshape(n_high, channels)

        low_vertices = torch.sparse.mm(
            pool_matrix,
            high_vertices
        )

        assert torch.isfinite(low_vertices).all(), (
            "Downsampled tensor has NaN/Inf"
        )

        return low_vertices

    def _apply_norm(self, *arrays):

        assert self.norm_params is not None

        mean = self.norm_params["mean"].view(1, -1)
        std = self.norm_params["std"].view(1, -1)

        normalized = []

        for arr in arrays:

            if arr.ndim == 2:

                out = (arr - mean) / std

            elif arr.ndim == 3:

                out = (
                    arr
                    - mean.view(1, 1, -1)
                ) / std.view(1, 1, -1)

            else:
                raise ValueError(
                    f"Unsupported tensor shape: {arr.shape}"
                )

            assert torch.isfinite(out).all(), (
                "Normalized tensor has NaN/Inf"
            )

            normalized.append(out)

        return normalized

    def __len__(self):

        return self.timesteps

    def __getitem__(self, idx):

        hr_raw_np = self.X[
            idx,
            :self.n_lat_hr,
            :self.n_lon_hr,
            :
        ]

        hr_raw = torch.tensor(
            hr_raw_np,
            dtype=torch.float32
        )

        channels = hr_raw.shape[-1]

        assert channels == self.channels

        hr_vertices = hr_raw.reshape(
            self.n_hr,
            channels
        )

        # ----------------------------------------------------
        # HR -> MR
        # ----------------------------------------------------
        mr_vertices = self.downsample_features_3d(
            hr_raw,
            self.pool_hr_mr
        )

        mr_raw = mr_vertices.reshape(
            self.n_lat_mr,
            self.n_lon_mr,
            channels
        )

        # ----------------------------------------------------
        # Training mode: LR -> MR
        # ----------------------------------------------------
        if self.mode == "train":

            lr_vertices = self.downsample_features_3d(
                mr_raw,
                self.pool_mr_lr
            )

            lr_raw = lr_vertices.reshape(
                self.n_lat_lr,
                self.n_lon_lr,
                channels
            )

            if self.norm_params is not None:

                hr_vertices, mr_vertices, lr_vertices = self._apply_norm(
                    hr_vertices,
                    mr_vertices,
                    lr_vertices
                )

                hr_raw = hr_vertices.reshape(
                    self.n_lat_hr,
                    self.n_lon_hr,
                    channels
                )

                mr_raw = mr_vertices.reshape(
                    self.n_lat_mr,
                    self.n_lon_mr,
                    channels
                )

                lr_raw = lr_vertices.reshape(
                    self.n_lat_lr,
                    self.n_lon_lr,
                    channels
                )

            return (
                hr_vertices,
                mr_vertices,
                lr_vertices,
                hr_raw,
                mr_raw,
                lr_raw
            )

        # ----------------------------------------------------
        # Validation / test mode: MR -> HR
        # ----------------------------------------------------
        if self.mode == "val":

            if self.norm_params is not None:

                hr_vertices, mr_vertices = self._apply_norm(
                    hr_vertices,
                    mr_vertices
                )

                hr_raw = hr_vertices.reshape(
                    self.n_lat_hr,
                    self.n_lon_hr,
                    channels
                )

                mr_raw = mr_vertices.reshape(
                    self.n_lat_mr,
                    self.n_lon_mr,
                    channels
                )

            return (
                hr_vertices,
                mr_vertices,
                hr_raw,
                mr_raw
            )


print("CELL 7 PASSED")
