import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

WORKDIR = Path(r"E:\My Thesis\ML Model\Neural Network\Script")
BOARD_L, BOARD_W = 0.1, 0.1
GRID_N = 50
N_SENSORS = 12

gx, gy = np.meshgrid(
    np.linspace(0, BOARD_L, GRID_N),
    np.linspace(0, BOARD_W, GRID_N),
)
grid_points = np.column_stack([gx.ravel(), gy.ravel()])  # (2500, 2)

rng = np.random.default_rng(7)
sensor_xy = rng.uniform(0.01, 0.09, size=(N_SENSORS, 2))
np.save(WORKDIR / "sensor_xy.npy", sensor_xy)
np.save(WORKDIR / "grid_xy.npy", grid_points)

valid_ids = pd.read_csv(WORKDIR / "valid_samples.csv", header=None)[0].tolist()

X_all, Y_all = [], []
for n, sid in enumerate(valid_ids, 1):
    df = pd.read_csv(WORKDIR / f"out_temp_{sid}.csv", header=None, names=["x", "y", "T"])
    pts = df[["x", "y"]].values
    vals = df["T"].values

    Y_field = griddata(pts, vals, grid_points, method="linear")
    Y_field = np.nan_to_num(Y_field, nan=np.nanmean(Y_field))

    X_sensor = griddata(pts, vals, sensor_xy, method="linear")
    X_sensor = np.nan_to_num(X_sensor, nan=np.nanmean(X_sensor))

    X_all.append(X_sensor)
    Y_all.append(Y_field)

    if n % 100 == 0:
        print(f"Processed {n}/{len(valid_ids)}")

X = np.array(X_all)
Y = np.array(Y_all)

print("X shape:", X.shape, " Y shape:", Y.shape)
np.savez(WORKDIR / "pegnn_dataset.npz", X=X, Y=Y, sensor_xy=sensor_xy, grid_xy=grid_points)
print("Saved pegnn_dataset.npz")