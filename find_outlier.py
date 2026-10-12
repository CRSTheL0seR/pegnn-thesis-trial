import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"
d = np.load(f"{WORKDIR}\\pegnn_dataset.npz")
g = np.load(f"{WORKDIR}\\pegnn_graph.npz")
X, Y = d["X"], d["Y"]
grid_xy = g["grid_xy"]

# Find the single worst node across the whole test set
y_mean, y_std = Y[:700].mean(), Y[:700].std()

# Reuse model classes (copy from train_pegnn.py) -- or just inspect raw Y directly first:
test_Y = Y[850:]
worst_flat_idx = np.argmax(np.abs(test_Y - test_Y.mean()))  # crude first pass: find extreme raw values
sample_idx, node_idx = np.unravel_index(worst_flat_idx, test_Y.shape)

print("Suspicious sample index (within test set):", sample_idx)
print("Suspicious node index:", node_idx)
print("Node coordinates:", grid_xy[node_idx])
print("Raw temperature value at this node:", test_Y[sample_idx, node_idx])
print("Dataset overall mean/std:", Y.mean(), Y.std())
print("Dataset overall min/max:", Y.min(), Y.max())