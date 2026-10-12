import numpy as np

data = np.load(r"E:\My Thesis\ML Model\Neural Network\Script\pegnn_dataset.npz")
X, Y, sensor_xy, grid_xy = data["X"], data["Y"], data["sensor_xy"], data["grid_xy"]

i = 0
nearest_idx = [np.argmin(np.sum((grid_xy - s) ** 2, axis=1)) for s in sensor_xy]
y_at_sensors = Y[i][nearest_idx]

print("X[0] (sensor readings):      ", X[0].round(1))
print("Y[0] at same locations:      ", y_at_sensors.round(1))
print("Max difference (should be small):", np.abs(X[0] - y_at_sensors).max())