# ============================================================
# Load ERA5 q dataset and build physics-aware tensor
# Future-proofed for flow-aware DeepSphere-ResNet-Note
# ============================================================

import numpy as np
import xarray as xr
from pathlib import Path

# ------------------------------------------------------------
# Load ERA5 GRIB file
# ------------------------------------------------------------
Q_FILE = Path(
    "/content/deepsphere_resnet_clean/data/raw/"
    "era5_specific_humidity_2025_europe_regular3h.grib"
)

assert Q_FILE.exists(), f"Missing file: {Q_FILE}"

ds_q = xr.open_dataset(str(Q_FILE), engine="cfgrib")

print("=" * 70)
print("RAW DATASET SUMMARY")
print("=" * 70)
print(ds_q)

# ------------------------------------------------------------
# Required coordinates / variables
# ------------------------------------------------------------
required_coords = [
    "longitude",
    "latitude",
    "isobaricInhPa"
]

required_var = "q"

for name in required_coords:
    assert (
        (name in ds_q.coords)
        or (name in ds_q.variables)
        or (name in ds_q)
    ), f"Missing required coordinate: {name}"

assert required_var in ds_q.data_vars, (
    f"Missing required variable: {required_var}"
)

# ------------------------------------------------------------
# Extract arrays
# ------------------------------------------------------------
lon_1d = ds_q["longitude"].values.astype(np.float32)
lat_1d = ds_q["latitude"].values.astype(np.float32)
plev_1d = ds_q["isobaricInhPa"].values.astype(np.float32)

# q shape:
# [time, pressure, latitude, longitude]
q_raw = ds_q["q"].values.astype(np.float32)

print("\n" + "=" * 70)
print("BASIC SHAPE CHECKS")
print("=" * 70)

print(f"longitude shape : {lon_1d.shape}")
print(f"latitude shape  : {lat_1d.shape}")
print(f"pressure shape  : {plev_1d.shape}")
print(f"q raw shape     : {q_raw.shape}")

# ------------------------------------------------------------
# Expected ERA5 dimension order
# ------------------------------------------------------------
expected_dims = (
    "time",
    "isobaricInhPa",
    "latitude",
    "longitude"
)

actual_dims = ds_q["q"].dims

print(f"q dims          : {actual_dims}")

assert actual_dims == expected_dims, (
    f"Unexpected q dims: {actual_dims} "
    f"| expected {expected_dims}"
)

# ------------------------------------------------------------
# Numerical sanity checks
# ------------------------------------------------------------
assert lon_1d.ndim == 1
assert lat_1d.ndim == 1
assert plev_1d.ndim == 1
assert q_raw.ndim == 4

assert np.isfinite(lon_1d).all(), "longitude contains NaN/Inf"
assert np.isfinite(lat_1d).all(), "latitude contains NaN/Inf"
assert np.isfinite(plev_1d).all(), "pressure contains NaN/Inf"
assert np.isfinite(q_raw).all(), "q contains NaN/Inf"

print(f"q min           : {q_raw.min():.6e}")
print(f"q max           : {q_raw.max():.6e}")
print(f"q mean          : {q_raw.mean():.6e}")
print(f"q std           : {q_raw.std():.6e}")

# ------------------------------------------------------------
# Coordinate monotonicity checks
# ------------------------------------------------------------
def check_monotonic(arr, name):

    diffs = np.diff(np.asarray(arr))

    increasing = np.all(diffs > 0)
    decreasing = np.all(diffs < 0)

    assert increasing or decreasing, (
        f"{name} is not strictly monotonic"
    )

    direction = "increasing" if increasing else "decreasing"

    print(f"{name} monotonic : OK ({direction})")


check_monotonic(lon_1d, "longitude")
check_monotonic(lat_1d, "latitude")
check_monotonic(plev_1d, "pressure")

# ------------------------------------------------------------
# IMPORTANT:
# Standardize latitude orientation
#
# ERA5 is usually north -> south.
# For directional graph transport operators,
# we standardize to south -> north.
# ------------------------------------------------------------
if lat_1d[0] > lat_1d[-1]:

    print("\nReversing latitude orientation:")
    print("north->south  --> south->north")

    lat_1d = lat_1d[::-1]

    # Reverse latitude dimension
    q_raw = q_raw[:, :, ::-1, :]

# ------------------------------------------------------------
# Final post-reversal validation
# ------------------------------------------------------------
assert lat_1d[0] < lat_1d[-1], (
    "Latitude orientation normalization failed."
)

print(f"\nLatitude range  : {lat_1d[0]} -> {lat_1d[-1]}")

# ------------------------------------------------------------
# Build variable tensor
#
# Current:
# [T, P, Y, X]
#
# Add variable channel dimension:
# [T, P, Y, X, C]
#
# This is future-proof for:
# q, u, v, temperature, etc.
# ------------------------------------------------------------
X_q = np.expand_dims(q_raw, axis=-1).astype(np.float32)

print("\n" + "=" * 70)
print("FEATURE TENSOR")
print("=" * 70)

print(f"X_q shape       : {X_q.shape}")

# ------------------------------------------------------------
# Tensor dimensions
# ------------------------------------------------------------
n_time, n_plev, n_lat, n_lon, n_var = X_q.shape

assert n_var == 1, (
    "q-only tensor must contain one variable channel."
)

assert n_plev == 37, (
    f"Expected 37 pressure levels, got {n_plev}"
)

# ------------------------------------------------------------
# Expected timesteps
#
# 4 months
# x 7 days
# x 8 timestamps/day
#
# = 224
# ------------------------------------------------------------
expected_timesteps = 224

assert n_time == expected_timesteps, (
    f"Expected {expected_timesteps} timesteps, "
    f"got {n_time}"
)

# ------------------------------------------------------------
# Reorder dimensions
#
# FROM:
# [T, P, Y, X, C]
#
# TO:
# [T, Y, X, P, C]
#
# This preserves:
# - pressure structure
# - variable structure
#
# DO NOT flatten yet.
# ------------------------------------------------------------
X_q_grid = X_q.transpose(0, 2, 3, 1, 4)

print(f"X_q_grid shape  : {X_q_grid.shape}")

expected_shape = (
    n_time,
    n_lat,
    n_lon,
    n_plev,
    n_var
)

assert X_q_grid.shape == expected_shape, (
    f"Unexpected X_q_grid shape: "
    f"{X_q_grid.shape}, expected {expected_shape}"
)

# ------------------------------------------------------------
# Build standardized 2D geographic grids
# ------------------------------------------------------------
lat_grid_2d, lon_grid_2d = np.meshgrid(
    lat_1d,
    lon_1d,
    indexing="ij"
)

print("\n" + "=" * 70)
print("GRID CHECK")
print("=" * 70)

print(f"lat_grid_2d shape : {lat_grid_2d.shape}")
print(f"lon_grid_2d shape : {lon_grid_2d.shape}")
print(f"expected grid     : ({n_lat}, {n_lon})")

assert lat_grid_2d.shape == (n_lat, n_lon)
assert lon_grid_2d.shape == (n_lat, n_lon)

# ------------------------------------------------------------
# Preserve atmospheric metadata
# IMPORTANT for:
# - pressure-aware kernels
# - transport-aware graph attention
# - future multi-variable experiments
# ------------------------------------------------------------
ATMOS_METADATA = {
    "pressure_levels_hPa": plev_1d,
    "longitude": lon_1d,
    "latitude": lat_1d,
    "n_timesteps": n_time,
    "n_pressure_levels": n_plev,
    "n_latitude": n_lat,
    "n_longitude": n_lon,
    "variables": ["specific_humidity"]
}

print("\n" + "=" * 70)
print("ATMOSPHERIC METADATA")
print("=" * 70)

print(f"Pressure levels : {plev_1d.shape}")
print(f"Variables       : {ATMOS_METADATA['variables']}")

# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------
assert np.isfinite(X_q_grid).all(), (
    "Tensor contains NaN/Inf values."
)

print("\nCELL 4 PASSED")

print(f"Timesteps        : {n_time}")
print(f"Pressure levels  : {n_plev}")
print(f"Latitudes        : {n_lat}")
print(f"Longitudes       : {n_lon}")

print("\nFinal tensor format:")
print("[T, Y, X, P, C]")

print(f"Tensor shape     : {X_q_grid.shape}")
