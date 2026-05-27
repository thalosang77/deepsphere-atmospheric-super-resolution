import pandas as pd
from pathlib import Path

region_path = Path(
    "/content/deepsphere_resnet_clean/artifacts/multi_region_transfer_results.csv"
)

assert region_path.exists(), f"Missing file: {region_path}"

region_df = pd.read_csv(region_path)

region_df = region_df.sort_values(
    "DeepSphere_improvement_%",
    ascending=False
).reset_index(drop=True)

display(region_df)

print("=" * 70)
print("MULTI-REGION TRANSFER SUMMARY")
print("=" * 70)

for _, row in region_df.iterrows():

    print(
        f"{row['region']:>20} | "
        f"DeepSphere improvement: "
        f"{row['DeepSphere_improvement_%']:.2f}%"
    )
