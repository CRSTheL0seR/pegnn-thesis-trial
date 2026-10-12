"""
Stage 1+2: Geometry setup + Parametrization
Generates 1000 valid, non-overlapping component layouts with randomized
power intensity, matching the PEGNN-style 2D steady conduction problem.

Output: parameters.csv -- one row per sample, consumed by the ANSYS
APDL batch driver in the next stage.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------- Board ----------------
BOARD_L = 0.1   # m
BOARD_W = 0.1   # m
MARGIN = 0.003  # keep components this far from board edges
CLEARANCE = 0.002  # minimum gap required between components

# ---------------- Fixed component sizes (w, h), from paper Table 1, scaled down to 4 ----------------
COMPONENTS = [
    {"name": "comp1", "w": 0.020, "h": 0.030, "power_dist": ("uniform", 0, 30000)},
    {"name": "comp2", "w": 0.010, "h": 0.010, "power_dist": ("gumbel", 30000, 3000)},
    {"name": "comp3", "w": 0.020, "h": 0.020, "power_dist": ("normal", 30000, 3000)},
    {"name": "comp4", "w": 0.0225, "h": 0.010, "power_dist": ("uniform", 0, 30000)},
]

N_SAMPLES = 1000


def sample_power(dist):
    kind, a, b = dist
    if kind == "uniform":
        return float(rng.uniform(a, b))
    if kind == "gumbel":
        return float(rng.gumbel(a, b))
    if kind == "normal":
        return float(rng.normal(a, b))
    raise ValueError(kind)


def rects_overlap(r1, r2, clearance):
    # r = (x1, y1, x2, y2)
    return not (
        r1[2] + clearance <= r2[0]
        or r2[2] + clearance <= r1[0]
        or r1[3] + clearance <= r2[1]
        or r2[3] + clearance <= r1[1]
    )


def sample_layout(max_tries=500):
    placed = []
    for comp in COMPONENTS:
        w, h = comp["w"], comp["h"]
        for _ in range(max_tries):
            x1 = rng.uniform(MARGIN, BOARD_L - MARGIN - w)
            y1 = rng.uniform(MARGIN, BOARD_W - MARGIN - h)
            rect = (x1, y1, x1 + w, y1 + h)
            if not any(rects_overlap(rect, p["rect"], CLEARANCE) for p in placed):
                placed.append({"name": comp["name"], "w": w, "h": h, "rect": rect})
                break
        else:
            return None  # failed to place this component -- caller retries whole layout
    return placed


rows = []
sample_id = 1
attempts = 0
while len(rows) < N_SAMPLES:
    attempts += 1
    layout = sample_layout()
    if layout is None:
        continue  # retry whole layout

    row = {"sample_id": sample_id}
    for comp_def, placed in zip(COMPONENTS, layout):
        x1, y1, x2, y2 = placed["rect"]
        name = comp_def["name"]
        row[f"{name}_x1"] = round(x1, 6)
        row[f"{name}_y1"] = round(y1, 6)
        row[f"{name}_x2"] = round(x2, 6)
        row[f"{name}_y2"] = round(y2, 6)
        row[f"{name}_power"] = round(sample_power(comp_def["power_dist"]), 2)

    rows.append(row)
    sample_id += 1

df = pd.DataFrame(rows)
df.to_csv("parameters.csv", index=False)

print(f"Generated {len(df)} valid samples in {attempts} attempts "
      f"({attempts - len(df)} layouts rejected for overlap).")
print(df.head(3).to_string())
