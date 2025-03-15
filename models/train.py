# models/train.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import os

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def load_dataset():
    dataset = []
    graph_dir = "data/graphs"
    for fname in os.listdir(graph_dir):
        if fname.endswith(".pt"):
            data = torch.load(os.path.join(graph_dir, fname), weights_only=False)
            dataset.append(data)
    return dataset

def train():
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GCN(in_channels=3, hidden_channels=64, out_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/gnn_model.pt")

if __name__ == '__main__':
    train()
