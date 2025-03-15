# scripts/evaluate.py
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import os
import torch.nn.functional as F
from sklearn.metrics import classification_report

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

def evaluate():
    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GCN(in_channels=3, hidden_channels=64, out_channels=3)
    model.load_state_dict(torch.load("models/checkpoints/gnn_model.pt"))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)
            y_true.extend(data.y.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=["Helix", "Strand", "Coil"]))

if __name__ == '__main__':
    evaluate()
