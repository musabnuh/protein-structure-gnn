# scripts/visualize_graph.py
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import argparse

LABEL_MAP = {0: "Helix", 1: "Strand", 2: "Coil"}
COLORS = {0: "red", 1: "blue", 2: "green"}

def visualize_graph(graph_file):
    data = torch.load(graph_file)
    G = to_networkx(data, to_undirected=True)

    labels = data.y.tolist()
    node_colors = [COLORS.get(label, "gray") for label in labels]

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_color=node_colors, with_labels=False, node_size=80)
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', label=LABEL_MAP[i],
                                  markerfacecolor=COLORS[i], markersize=8) for i in range(3)]
    plt.legend(handles=legend_patches)
    plt.title("Protein Residue Graph Colored by Secondary Structure")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', required=True)
    args = parser.parse_args()
    visualize_graph(args.graph_file)
