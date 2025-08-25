import pandas as pd

# List input files
files = [
    "data/items_implicature.csv",
    "data/items_indirectness.csv",
    "data/items_politeness.csv"
]

# Read each file into a DataFrame
dfs = [pd.read_csv(f) for f in files]

# Concatenate all DataFrames
all_items = pd.concat(dfs, ignore_index=True)

# Write to the combined file
all_items.to_csv("data/items_all.csv", index=False)
