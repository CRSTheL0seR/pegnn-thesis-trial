import pandas as pd
from pathlib import Path

WORKDIR = Path(r"E:\My Thesis\ML Model\Neural Network\Script")
T_SINK = 298

good, bad = [], []
for i in range(1, 1001):
    f = WORKDIR / f"out_temp_{i}.csv"
    if not f.exists():
        bad.append((i, "missing"))
        continue
    try:
        df = pd.read_csv(f, header=None, names=["x", "y", "T"])
    except Exception as e:
        bad.append((i, f"read error: {e}"))
        continue
    if df.empty or df["T"].isna().any():
        bad.append((i, "empty/NaN"))
        continue
    if df["T"].min() < T_SINK - 1 or df["T"].max() > T_SINK + 200:
        bad.append((i, f"out of range [{df['T'].min():.1f}, {df['T'].max():.1f}]"))
        continue
    good.append(i)

print(f"Good: {len(good)}  Bad: {len(bad)}")
for sid, reason in bad[:30]:
    print(f"  sample {sid}: {reason}")

pd.Series(good).to_csv(WORKDIR / "valid_samples.csv", index=False, header=False)