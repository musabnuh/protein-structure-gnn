# models/train.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import os

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        # Initializes a Graph Convolutional Network with two layers.
        # GCNConv aggregates structural features from neighboring nodes (e.g., nearby residues),
        # mimicking how local 3D environments influence residue function in proteins.
        super().__init__()

        # First GCN layer: transforms input features into a hidden representation.
        # Input could include properties like amino acid identity, charge, or position.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Second GCN layer: allows the model to consider broader spatial context
        # by stacking graph convolutions — similar to modeling long-range residue interactions.
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Forward pass through the network:
        # x: Node features (residues with attributes)
        # edge_index: Pairs of connected nodes (based on spatial proximity or chemical bonds)

        # Apply first GCN layer to learn from immediate spatial neighbors
        x = self.conv1(x, edge_index)
        
        # Apply non-linear activation to introduce complexity in feature interactions.
        # Reflects how biochemical influence is often non-linear (e.g., allostery).
        x = F.relu(x)
        
        # Second convolution captures more global context within the protein structure
        x = self.conv2(x, edge_index)
        
        # Output is passed on to the loss function or evaluation logic.
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
    # Load preprocessed graph datasets from disk.
    # Each graph represents a protein structure with residue-level features and edges based on 3D proximity.
    dataset = load_dataset()

    # DataLoader allows batching and shuffling — important for robust model training.
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize the model.
    # in_channels = # of features per residue (e.g., atom type, charge, position)
    # hidden/out_channels determine the depth and output complexity of the GNN.
    model = GCN(in_channels=3, hidden_channels=64, out_channels=3)
    
    # Use Adam optimizer — common for graph learning tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Basic training loop (1 epoch shown here)
    for epoch in range(20):
        total_loss = 0
        for data in loader:
            optimizer.zero_grad()
          
            # Forward pass through the GNN
            out = model(data.x, data.edge_index)
            
            # Compute loss between predicted and actual labels
            # Assumes a classification task (e.g., fold-type, functional class)
            loss = F.cross_entropy(out, data.y)
            
            # Backpropagate and update model weights
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/gnn_model.pt")

if __name__ == '__main__':
    train()
