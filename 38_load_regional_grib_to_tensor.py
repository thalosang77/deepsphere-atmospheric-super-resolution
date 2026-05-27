# ============================================================
# Load any regional GRIB into model-ready tensor
# ============================================================

def load_q_region_grib(grib_path):
    grib_path = Path(grib_path)
    assert grib_path.exists(), f"Missing file: {grib_path}"

    ds = xr.open_dataset(str(grib_path), engine="cfgrib")

    lon = ds["longitude"].values
    lat = ds["latitude"].values
    plev = ds["isobaricInhPa"].values
    q_raw = ds["q"].values

    assert ds["q"].dims == ("time", "isobaricInhPa", "latitude", "longitude")
    assert q_raw.ndim == 4

    X = np.expand_dims(q_raw, axis=-1).astype(np.float32)
    T, P, Y, Xlon, C = X.shape

    X_grid = X.transpose(0, 2, 3, 1, 4).reshape(T, Y, Xlon, P * C)

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    print("=" * 70)
    print("REGION LOADED")
    print("=" * 70)
    print("File:", grib_path)
    print("Raw q:", q_raw.shape)
    print("Model tensor:", X_grid.shape)
    print("lat/lon grid:", lat_grid.shape, lon_grid.shape)

    return {
        "ds": ds,
        "lon": lon,
        "lat": lat,
        "plev": plev,
        "X_grid": X_grid,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }

print("CELL 36 PASSED")
