import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing, GCNConv

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"
GRID_N = 50

d = np.load(f"{WORKDIR}\\pegnn_dataset.npz")
g = np.load(f"{WORKDIR}\\pegnn_graph.npz")
X, Y = d["X"], d["Y"]
edge_index_np = g["edge_index"]
sensor_node_ids = g["sensor_node_ids"]
N_NODES = Y.shape[1]

y_mean, y_std = Y[:700].mean(), Y[:700].std()
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

class GradOp(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr="add")
        self.lin = nn.Linear(channels, channels, bias=False)
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        return self.lin(x_j - x_i)

class PhysicalLayer(nn.Module):
    def __init__(self, channels, step_size=0.1):
        super().__init__()
        self.grad_op = GradOp(channels)
        self.div_op = GradOp(channels)
        self.K = nn.Linear(channels, channels, bias=False)
        self.h = step_size
    def forward(self, x, edge_index):
        gr = F.gelu(self.grad_op(x, edge_index))
        dv = F.gelu(self.K(self.div_op(gr, edge_index)))
        return x + self.h * dv

class PEGNN(nn.Module):
    def __init__(self, hidden=16, n_layers=20, step_size=0.1):
        super().__init__()
        self.readin = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([PhysicalLayer(hidden, step_size) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        x = self.readin(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.readout(x)

pegnn = PEGNN(hidden=16, n_layers=20)
pegnn.load_state_dict(torch.load(f"{WORKDIR}\\pegnn_model.pt"))
pegnn.eval()

test_idx = 0
true_field = Y[850 + test_idx].reshape(GRID_N, GRID_N)
sensor_vals = X[850 + test_idx]

node_feat = torch.zeros(N_NODES, 1, dtype=torch.float32)
node_feat[sensor_node_ids] = torch.tensor(
    (sensor_vals - y_mean) / y_std, dtype=torch.float32
).unsqueeze(1)

with torch.no_grad():
    pegnn_pred = (pegnn(node_feat, edge_index) * y_std + y_mean).numpy().reshape(GRID_N, GRID_N)

error_map = np.abs(pegnn_pred - true_field)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(error_map, origin="lower", cmap="hot")
ax.set_title("PEGNN absolute error per node (Sample 0)")
fig.colorbar(im, label="|Error| (K)")
plt.savefig(f"{WORKDIR}\\error_map_sample0.png", dpi=150)
print("Saved error_map_sample0.png")
print("Max error:", error_map.max(), " Mean error:", error_map.mean())

# Also mark where the sensors actually are, for reference
sensor_rows = sensor_node_ids // GRID_N
sensor_cols = sensor_node_ids % GRID_N
ax.scatter(sensor_cols, sensor_rows, c="cyan", marker="x", s=60, label="Sensors")
ax.legend()
plt.savefig(f"{WORKDIR}\\error_map_sample0.png", dpi=150)
print("Updated with sensor locations marked.")