# ============================================================
# ERA5 download: performance-improved q dataset
# ============================================================

import json
from pathlib import Path
import cdsapi

DATA_RAW_DIR = RAW_DATA_DIR
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_RAW_DIR / "era5_specific_humidity_2025_europe_regular3h.grib"
REQUEST_FILE = DATA_RAW_DIR / "era5_specific_humidity_2025_europe_regular3h_request.json"

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="5bfff7c9-3b3e-47bd-9a96-3112bc3c3686"
)

area_subset = [70, -20, 30, 20]
dataset = "reanalysis-era5-pressure-levels"

request = {
    "product_type": ["reanalysis"],
    "variable": ["specific_humidity"],
    "year": ["2025"],
    "month": ["03", "06", "09", "12"],
    "day": ["01", "05", "10", "15", "20", "25", "29"],
    "time": [
        "00:00", "03:00", "06:00", "09:00",
        "12:00", "15:00", "18:00", "21:00"
    ],
    "pressure_level": [
        "1", "2", "3", "5", "7", "10",
        "20", "30", "50", "70", "100", "125",
        "150", "175", "200", "225", "250", "300",
        "350", "400", "450", "500", "550", "600",
        "650", "700", "750", "775", "800", "825",
        "850", "875", "900", "925", "950", "975", "1000"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": area_subset
}

with open(REQUEST_FILE, "w") as f:
    json.dump({"dataset": dataset, "request": request, "output_file": str(OUTPUT_FILE)}, f, indent=2)

print("=" * 70)
print("DOWNLOADING ERA5 q DATASET — REGULAR 3-HOURLY SAMPLING")
print("=" * 70)
print("Output:", OUTPUT_FILE)
print("Times :", request["time"])

c.retrieve(dataset, request).download(str(OUTPUT_FILE))

assert OUTPUT_FILE.exists(), "Download failed."
size_mb = OUTPUT_FILE.stat().st_size / (1024 ** 2)
assert size_mb > 1, "Downloaded file too small."

print(f"\nFile size: {size_mb:.2f} MB")
print("CELL 3 PASSED")
