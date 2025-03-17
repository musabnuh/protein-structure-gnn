# scripts/visualize_features.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def visualize_features(graph_file):
    """Visualize the distribution of node features in a protein graph."""
    data = torch.load(graph_file, weights_only=False)
    
    # Extract node features and labels
    X = data.x.numpy()
    y = data.y.numpy()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Feature names (assuming these are known)
    feature_names = ["Hydrophobicity", "Charge", "Polarity", "Size", "Aromaticity"]
    
    # Plot feature distributions by secondary structure class
    fig, axes = plt.subplots(len(feature_names), 1, figsize=(10, 15), sharex=True)
    
    for i, (ax, feature_name) in enumerate(zip(axes, feature_names)):
        for label in np.unique(y):
            sns.kdeplot(X[y == label, i], ax=ax, label=f"{LABEL_MAP[label]}")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    filename = os.path.basename(graph_file).replace('.pt', '')
    output_path = os.path.join('visualizations', f"{filename}_feature_dist.png")
    plt.savefig(output_path, dpi=300)
    print(f"Feature distribution saved to {output_path}")
    
    plt.show()

LABEL_MAP = {0: "Helix", 1: "Strand", 2: "Coil"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file', required=True, help='Path to the graph file')
    args = parser.parse_args()
    visualize_features(args.graph_file)