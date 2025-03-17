# scripts/visualize_graph.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import argparse
import os

# Import the specific class first, then add it to safe globals
from torch_geometric.data.data import DataEdgeAttr
torch.serialization.add_safe_globals([DataEdgeAttr])

# Alternative approach if the above doesn't work
# torch.load with weights_only=False parameter
def visualize_graph(graph_file):
    data = torch.load(graph_file, weights_only=False)
    G = to_networkx(data, to_undirected=True)

    labels = data.y.tolist()
    node_colors = [COLORS.get(label, "gray") for label in labels]

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_color=node_colors, with_labels=False, node_size=80)
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=LABEL_MAP[i],
                                 markerfacecolor=COLORS[i], markersize=8) for i in range(3)]
    plt.legend(handles=legend_patches)
    plt.title("Protein Residue Graph Colored by Secondary Structure")
    
    # Extract filename from path and create output path
    filename = os.path.basename(graph_file).replace('.pt', '')
    output_path = os.path.join('visualizations', f"{filename}_graph.png")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")
    
    plt.show()

LABEL_MAP = {0: "Helix", 1: "Strand", 2: "Coil"}
COLORS = {0: "red", 1: "blue", 2: "green"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', required=True)
    args = parser.parse_args()
    visualize_graph(args.graph_file)
