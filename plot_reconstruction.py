import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"
device = "cpu"

d = np.load(f"{WORKDIR}\\pegnn_dataset.npz")
g = np.load(f"{WORKDIR}\\pegnn_graph.npz")
X, Y = d["X"], d["Y"]
edge_index_np = g["edge_index"]
sensor_node_ids = g["sensor_node_ids"]
N_NODES = Y.shape[1]
GRID_N = 50

y_mean, y_std = Y[:700].mean(), Y[:700].std()
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

# ---- Rebuild model classes (must match train_pegnn.py exactly) ----
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

from torch_geometric.nn import GCNConv
class GCNBaseline(nn.Module):
    def __init__(self, hidden=16, n_layers=20):
        super().__init__()
        self.readin = nn.Linear(1, hidden)
        self.convs = nn.ModuleList([GCNConv(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Linear(hidden, 1)
    def forward(self, x, edge_index):
        x = F.gelu(self.readin(x))
        for conv in self.convs:
            x = x + F.gelu(conv(x, edge_index))
        return self.readout(x)

pegnn = PEGNN(hidden=16, n_layers=20)
pegnn.load_state_dict(torch.load(f"{WORKDIR}\\pegnn_model.pt"))
pegnn.eval()

gcn = GCNBaseline(hidden=16, n_layers=20)
gcn.load_state_dict(torch.load(f"{WORKDIR}\\gcn_model.pt"))
gcn.eval()

# ---- Pick a test sample to visualize ----
test_idx = 60   # same sample we investigated before (sample_id 911) -- pick others too
true_field = Y[850 + test_idx].reshape(GRID_N, GRID_N)
sensor_vals = X[850 + test_idx]

node_feat = torch.zeros(N_NODES, 1, dtype=torch.float32)
node_feat[sensor_node_ids] = torch.tensor((sensor_vals - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)

with torch.no_grad():
    pegnn_pred = (pegnn(node_feat, edge_index) * y_std + y_mean).numpy().reshape(GRID_N, GRID_N)
    gcn_pred = (gcn(node_feat, edge_index) * y_std + y_mean).numpy().reshape(GRID_N, GRID_N)

# ---- Plot side by side ----
vmin, vmax = true_field.min(), true_field.max()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, field, title in zip(axes, [true_field, pegnn_pred, gcn_pred],
                              ["Ground Truth (ANSYS)", "PEGNN Prediction", "GCN Prediction"]):
    im = ax.imshow(field, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x (grid)")
    ax.set_ylabel("y (grid)")

fig.colorbar(im, ax=axes, label="Temperature (K)", shrink=0.8)
plt.savefig(f"{WORKDIR}\\reconstructed_field_sample{test_idx}.png", dpi=150, bbox_inches="tight")
print(f"Saved reconstructed_field_sample{test_idx}.png")