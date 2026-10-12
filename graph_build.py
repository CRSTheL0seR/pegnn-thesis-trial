import numpy as np

WORKDIR = r"E:\My Thesis\ML Model\Neural Network\Script"
data = np.load(f"{WORKDIR}\\pegnn_dataset.npz")
grid_xy = data["grid_xy"]      # (2500, 2)
sensor_xy = data["sensor_xy"]  # (12, 2)

GRID_N = 50  # must match build_dataset.py

def node_id(row, col):
    return row * GRID_N + col

edges = []
for row in range(GRID_N):
    for col in range(GRID_N):
        n = node_id(row, col)
        if col + 1 < GRID_N:            # right neighbor
            m = node_id(row, col + 1)
            edges.append((n, m))
            edges.append((m, n))        # both directions -- undirected graph
        if row + 1 < GRID_N:            # down neighbor
            m = node_id(row + 1, col)
            edges.append((n, m))
            edges.append((m, n))

edge_index = np.array(edges, dtype=np.int64).T   # shape (2, E)
print("edge_index shape:", edge_index.shape)      # expect (2, 9800)

# Map each sensor's physical (x,y) to its nearest grid node index
sensor_node_ids = []
for sx, sy in sensor_xy:
    dists = np.sum((grid_xy - np.array([sx, sy])) ** 2, axis=1)
    sensor_node_ids.append(int(np.argmin(dists)))
sensor_node_ids = np.array(sensor_node_ids)
print("sensor_node_ids:", sensor_node_ids)

np.savez(f"{WORKDIR}\\pegnn_graph.npz",
         edge_index=edge_index,
         sensor_node_ids=sensor_node_ids,
         grid_xy=grid_xy)
print("Saved pegnn_graph.npz")