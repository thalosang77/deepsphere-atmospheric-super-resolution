# ============================================================
# Run multi-region transfer benchmark
# ============================================================

import pandas as pd
from pathlib import Path
import gc
import torch

region_files = {
    "europe": DATA_RAW_DIR / "era5_q_2025_europe_regular3h.grib",
    "sea": DATA_RAW_DIR / "era5_q_2025_sea_regular3h.grib",
    "high_latitude": DATA_RAW_DIR / "era5_q_2025_high_latitude_regular3h.grib",

    # This may be missing if CDS failed on dateline-crossing request
    "longitude_seam": DATA_RAW_DIR / "era5_q_2025_longitude_seam_regular3h.grib",
}

region_results = []

print("=" * 70)
print("CELL 38 — MULTI-REGION TRANSFER BENCHMARK")
print("=" * 70)

for region_name, file_path in region_files.items():

    file_path = Path(file_path)

    print("\n" + "=" * 70)
    print("REGION:", region_name)
    print("=" * 70)

    if not file_path.exists():
        print("Missing, skipping:", file_path)
        continue

    if file_path.stat().st_size < 1_000_000:
        print("File too small / likely incomplete, skipping:", file_path)
        continue

    try:
        result = evaluate_region_transfer(
            region_name=region_name,
            grib_file=file_path
        )

        region_results.append(result)

    except Exception as e:
        print(f"FAILED region: {region_name}")
        print("Error:", repr(e))

        region_results.append(
            {
                "region": region_name,
                "status": "failed",
                "error": repr(e),
            }
        )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if len(region_results) == 0:
    raise RuntimeError("No regional results were produced.")

region_results_df = pd.DataFrame(region_results)

display(region_results_df)

region_results_path = (
    ARTIFACT_DIR /
    "multi_region_transfer_results.csv"
)

region_results_df.to_csv(
    region_results_path,
    index=False
)

print("Saved:", region_results_path)
print("CELL 38 PASSED")
