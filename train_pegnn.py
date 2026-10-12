
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, GCNConv

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

d = np.load(f"{WORKDIR}\\pegnn_dataset.npz")
g = np.load(f"{WORKDIR}\\pegnn_graph.npz")
X, Y = d["X"], d["Y"]
edge_index_np = g["edge_index"]
sensor_node_ids = g["sensor_node_ids"]
N_NODES = Y.shape[1]

y_mean, y_std = Y[:700].mean(), Y[:700].std()
X_n = (X - y_mean) / y_std
Y_n = (Y - y_mean) / y_std
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

class PEGNNDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X, self.Y = X, Y
    def len(self): return len(self.X)
    def get(self, idx):
        node_feat = torch.zeros(N_NODES, 1, dtype=torch.float32)
        node_feat[sensor_node_ids] = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(self.Y[idx], dtype=torch.float32).unsqueeze(1)
        return Data(x=node_feat, edge_index=edge_index, y=y)

train_ds = PEGNNDataset(X_n[:700], Y_n[:700])
val_ds   = PEGNNDataset(X_n[700:850], Y_n[700:850])
test_ds  = PEGNNDataset(X_n[850:], Y_n[850:])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)
test_loader  = DataLoader(test_ds, batch_size=16)

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

def train_model(model, name, epochs=250, lr=5e-3):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = float("inf"), None
    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss, n_batches = 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = F.l1_loss(pred, batch.y)
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item()
            n_batches += 1
        sched.step()
        train_hist.append(epoch_train_loss / n_batches)

        model.eval()
        val_loss, n = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                val_loss += F.l1_loss(pred, batch.y).item() * batch.num_graphs
                n += batch.num_graphs
        val_loss /= n
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{name}] epoch {epoch}: train MAE = {train_hist[-1]:.4f}  val MAE = {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, train_hist, val_hist

def evaluate(model, loader, name):
    model.eval()
    mae_total, maxae, n = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index)
            pred_real = pred * y_std + y_mean
            true_real = batch.y * y_std + y_mean
            err = (pred_real - true_real).abs()
            mae_total += err.sum().item()
            maxae = max(maxae, err.max().item())
            n += err.numel()
    print(f"[{name}] Test MAE = {mae_total/n:.3f} K   MaxAE = {maxae:.3f} K")

print("\n--- Training PEGNN ---")
pegnn, pegnn_train, pegnn_val = train_model(PEGNN(hidden=16, n_layers=20), "PEGNN", epochs=250, lr=2e-3)
evaluate(pegnn, test_loader, "PEGNN")

print("\n--- Training GCN baseline ---")
gcn, gcn_train, gcn_val = train_model(GCNBaseline(hidden=16, n_layers=20), "GCN", epochs=250, lr=5e-3)
evaluate(gcn, test_loader, "GCN")

torch.save(pegnn.state_dict(), f"{WORKDIR}\\pegnn_model.pt")
torch.save(gcn.state_dict(), f"{WORKDIR}\\gcn_model.pt")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(pegnn_train, label="Train")
axes[0].plot(pegnn_val, label="Validation")
axes[0].set_title("PEGNN: Train vs Validation MAE")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MAE (normalized)")
axes[0].legend()

axes[1].plot(gcn_train, label="Train")
axes[1].plot(gcn_val, label="Validation")
axes[1].set_title("GCN: Train vs Validation MAE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MAE (normalized)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{WORKDIR}\\training_curves.png", dpi=150)
print("\nSaved training_curves.png")