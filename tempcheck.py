import pandas as pd
import numpy as np

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"

# test index 60 -> original row 850+60=910 (0-indexed) -> sample_id = 911
valid_ids = pd.read_csv(f"{WORKDIR}\\valid_samples.csv", header=None)[0].tolist()
sample_id = valid_ids[910]
print("Checking sample_id:", sample_id)

df = pd.read_csv(f"{WORKDIR}\\out_temp_{sample_id}.csv", header=None, names=["x", "y", "T"])

# Look at raw mesh nodes closest to the corner (0, 0.1)
df["dist_to_corner"] = np.sqrt(df["x"]**2 + (df["y"] - 0.1)**2)
nearest = df.nsmallest(10, "dist_to_corner")
print(nearest.to_string())

# Also check this sample's component layout
params = pd.read_csv(f"{WORKDIR}\\parameters.csv")
row = params[params["sample_id"] == sample_id]
print("\nComponent layout for this sample:")
print(row.to_string())