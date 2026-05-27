# ============================================================
# Download multi-region ERA5 q datasets
# Europe, SEA, high-latitude, longitude seam
# ============================================================

import cdsapi
from pathlib import Path

PROJECT_ROOT = Path("/content/deepsphere_resnet_clean")
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="5bfff7c9-3b3e-47bd-9a96-3112bc3c3686"
)

pressure_levels = [
    "1","2","3","5","7","10",
    "20","30","50","70","100","125",
    "150","175","200","225","250","300",
    "350","400","450","500","550","600",
    "650","700","750","775","800","825",
    "850","875","900","925","950","975","1000"
]

common_request = {
    "product_type": ["reanalysis"],
    "variable": ["specific_humidity"],
    "year": ["2025"],
    "month": ["03", "06", "09", "12"],
    "day": ["01", "05", "10", "15", "20", "25", "29"],
    "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
    "pressure_level": pressure_levels,
    "data_format": "grib",
    "download_format": "unarchived",
}

regions = {
    "europe": [70, -20, 30, 20],
    "sea": [25, 95, -15, 135],
    "high_latitude": [85, -40, 45, 0],
    "longitude_seam": [20, 160, -20, -160],  # crosses dateline
}

dataset = "reanalysis-era5-pressure-levels"

for region_name, area in regions.items():
    out_file = DATA_RAW_DIR / f"era5_q_2025_{region_name}_regular3h.grib"

    if out_file.exists() and out_file.stat().st_size > 1_000_000:
        print(f"Already exists, skipping: {out_file}")
        continue

    request = dict(common_request)
    request["area"] = area

    print("=" * 70)
    print(f"DOWNLOADING REGION: {region_name}")
    print("Area:", area)
    print("Output:", out_file)
    print("=" * 70)

    c.retrieve(dataset, request).download(str(out_file))

    size_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"Saved: {out_file}")
    print(f"Size : {size_mb:.2f} MB")

print("CELL 35 PASSED")
