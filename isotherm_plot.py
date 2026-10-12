import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
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

# ---- Model classes (must match train_pegnn.py) ----
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

# ---- Pick test samples to show as rows (indices within the test set, 0-149) ----
test_indices = [0, 25, 60]

n_rows = len(test_indices)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
if n_rows == 1:
    axes = axes.reshape(1, 3)

col_titles = ["Truth", "PEGNN", "GCN"]

for row, test_idx in enumerate(test_indices):
    true_field = Y[850 + test_idx].reshape(GRID_N, GRID_N)
    sensor_vals = X[850 + test_idx]

    node_feat = torch.zeros(N_NODES, 1, dtype=torch.float32)
    node_feat[sensor_node_ids] = torch.tensor(
        (sensor_vals - y_mean) / y_std, dtype=torch.float32
    ).unsqueeze(1)

    with torch.no_grad():
        pegnn_pred = (pegnn(node_feat, edge_index) * y_std + y_mean).numpy().reshape(GRID_N, GRID_N)
        gcn_pred = (gcn(node_feat, edge_index) * y_std + y_mean).numpy().reshape(GRID_N, GRID_N)

    # Light smoothing for DISPLAY only -- does not affect saved metrics/model
    pegnn_pred_smooth = gaussian_filter(pegnn_pred, sigma=0.8)
    gcn_pred_smooth = gaussian_filter(gcn_pred, sigma=0.8)

    fields = [true_field, pegnn_pred_smooth, gcn_pred_smooth]
    vmin = min(f.min() for f in fields)
    vmax = max(f.max() for f in fields)
    levels = np.linspace(vmin, vmax, 8)   # reduced from 15 -- fewer, cleaner contour bands

    for col, (field, title) in enumerate(zip(fields, col_titles)):
        ax = axes[row, col]
        cs = ax.contour(field, levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

        max_idx = np.unravel_index(np.argmax(field), field.shape)
        ax.plot(max_idx[1], max_idx[0], marker="*", color="red", markersize=14)
        ax.annotate(f"{field.max():.2f}K", (max_idx[1], max_idx[0]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=10, color="red", weight="bold")

        if row == 0:
            ax.set_title(title, fontsize=14)
        if col == 0:
            ax.set_ylabel(f"Sample {test_idx}", fontsize=12)
        ax.set_aspect("equal")

fig.text(0.5, -0.01, "Note: PEGNN/GCN panels lightly smoothed (Gaussian, sigma=0.8) for display only -- "
                       "metrics (MAE/MaxAE) are computed from the raw, unsmoothed predictions.",
          ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(f"{WORKDIR}\\isotherm_comparison.png", dpi=150, bbox_inches="tight")
print("Saved isotherm_comparison.png")