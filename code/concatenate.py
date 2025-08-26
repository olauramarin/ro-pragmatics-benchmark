import pandas as pd

# Load subsets
politeness = pd.read_csv("data/items_politeness.csv", encoding="utf-8")
indirectness = pd.read_csv("data/items_indirectness.csv", encoding="utf-8")
implicature = pd.read_csv("data/items_implicature.csv", encoding="utf-8")

# Concatenate into one DataFrame
all_items = pd.concat([politeness, indirectness, implicature], ignore_index=True)

# Save combined file
all_items.to_csv("data/items_all.csv", index=False, encoding="utf-8")
