# scripts/visualize_embeddings.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import os

def visualize_embeddings(model_file, data_file):
    """Visualize node embeddings using t-SNE."""
    # Load model
    model = torch.load(model_file)
    model.eval()
    
    # Load data
    data = torch.load(data_file, weights_only=False)
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    
    # Convert to numpy
    embeddings_np = embeddings.numpy()
    labels_np = data.y.numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels_np):
        mask = labels_np == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            label=LABEL_MAP[label],
            alpha=0.7
        )
    
    plt.legend()
    plt.title("t-SNE Visualization of Node Embeddings")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Save plot
    filename = os.path.basename(data_file).replace('.pt', '')
    output_path = os.path.join('visualizations', f"{filename}_embeddings.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Embeddings visualization saved to {output_path}")
    
    plt.show()

LABEL_MAP = {0: "Helix", 1: "Strand", 2: "Coil"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, help='Path to trained model')
    parser.add_argument('--data_file', required=True, help='Path to graph data file')
    args = parser.parse_args()
    visualize_embeddings(args.model_file, args.data_file)