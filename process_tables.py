import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------
dense = False
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

base_name = "mnist_standard_mlp_results_seed{}.csv"
output_file = "mnist_mlp_results_avg.csv"

if dense:
    base_name = "mnist_dense_mlp_results_B_is_10_seed{}.csv"
    output_file = "mnist_dense_mlp_results_B_is_10_avg.csv"

# -------------------------------------------------
# Load all CSVs
# -------------------------------------------------
dfs = []
for seed in seeds:
    fname = base_name.format(seed)
    df = pd.read_csv(fname)
    df["Seed"] = seed
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------
# Convert accuracy columns to numeric (important)
# -------------------------------------------------
all_df["Train Acc (%)"] = pd.to_numeric(all_df["Train Acc (%)"])
all_df["Test Acc (%)"] = pd.to_numeric(all_df["Test Acc (%)"])

# -------------------------------------------------
# Group and aggregate
# -------------------------------------------------
agg_df = (
    all_df
    .groupby(["Architecture", "Width", "Parameters"], as_index=False)
    .agg(
        Train_Acc_Mean=("Train Acc (%)", "mean"),
        Train_Acc_Std=("Train Acc (%)", "std"),
        Test_Acc_Mean=("Test Acc (%)", "mean"),
        Test_Acc_Std=("Test Acc (%)", "std"),
    )
)

# -------------------------------------------------
# Optional: format as mean ± std strings (for paper)
# -------------------------------------------------
agg_df["Train Acc (%)"] = agg_df.apply(
    lambda r: f"{r.Train_Acc_Mean:.2f} ± {r.Train_Acc_Std:.2f}", axis=1
)
agg_df["Test Acc (%)"] = agg_df.apply(
    lambda r: f"{r.Test_Acc_Mean:.2f} ± {r.Test_Acc_Std:.2f}", axis=1
)

# Keep both numeric + formatted if you want
final_df = agg_df[[
    "Architecture",
    "Width",
    "Train Acc (%)",
    "Test Acc (%)",
    "Parameters"
]]

# -------------------------------------------------
# Save and display
# -------------------------------------------------
final_df.to_csv(output_file, index=False)
print("Averaged results saved to:", output_file)
print(final_df.to_string(index=False))
