# Protein Secondary Structure Classification with GNNs

## 🔬 Motivation
This project demonstrates key concepts in:
- **Geometric Deep Learning / Graph Neural Networks**: Apply graph-based deep learning to structural biology problems
- **Structural Biology & Computational Biology**: Analyze protein structures using computational methods
- **Protein Ensemble Modeling**: Represent protein structures as graphs to capture spatial relationships

## 🚀 Project Pipeline
Run everything using:
```bash
python run_pipeline.py
```

### Pipeline Steps
1. **Download PDB files**: Retrieves protein structures from the RCSB Protein Data Bank
2. **Run DSSP** for secondary structure annotation: Labels residues as helix, strand, or coil
3. **Construct graphs** from residue coordinates: Creates graph representations with spatial relationships
4. **Train a GNN** (GCN) to classify structure types: Uses Graph Convolutional Networks
5. **Evaluate** model and report classification performance: Precision, recall, F1-score metrics
6. **Visualize** results: Graph structure, embeddings, and prediction accuracy

## 📂 Folder Structure
```
.
├── data/             # Raw PDBs, DSSP files, and graphs
│   ├── raw/          # PDB files downloaded from RCSB
│   ├── dssp/         # DSSP annotation files
│   └── graphs/       # Processed graph representations (PyTorch .pt files)
├── scripts/          # Graph construction, evaluation, visualization
│   ├── construct_graph.py         # Create protein graphs from structures
│   ├── evaluate.py                # Evaluate model performance
│   ├── visualize_3d_structure.py  # View 3D protein structures
│   ├── visualize_embeddings.py    # Visualize learned node embeddings
│   ├── visualize_features.py      # Analyze feature distributions
│   ├── visualize_graph.py         # Render protein graphs with labels
│   └── visualize_training.py      # Plot training/validation metrics
├── models/           # GNN training code and checkpoints
│   ├── train.py      # Training loop and model definition
│   └── checkpoints/  # Saved model weights
├── visualizations/   # Generated images and plots
├── run_pipeline.py   # Full pipeline orchestrator
├── colab_pipeline.py # Notebook-friendly version for Google Colab
└── README.md
```

## 📦 Requirements
Install dependencies:
```bash
# Option 1: Using pip directly
pip install torch torchvision torchaudio
pip install torch-geometric
pip install biopython scikit-learn networkx matplotlib notebook

# Option 2: Using requirements.txt
pip install -r requirements.txt

# Install DSSP (required for secondary structure annotation)
sudo apt install dssp  # For Ubuntu/Debian
# For macOS: brew install brewsci/bio/dssp (requires Homebrew)
```

Additional requirements for visualization scripts:
```bash
# For 3D structure visualization
pip install nglview biotite

# For embedding visualization
pip install scikit-learn pandas
```

## 🔍 Evaluation Output
After training, you’ll see a classification report in the following format:
```
              precision    recall  f1-score   support

       Helix                                           ...
      Strand                                           ...
        Coil                                           ...
```

## 📊 Visualization
The project provides multiple visualization tools:

### Graph Visualization
Visualize a protein's graph representation with secondary structure labels:
```bash
python scripts/visualize_graph.py --graph_file data/graphs/1CRN.pt
```

### 3D Structure Visualization
View the 3D protein structure (requires nglview):
```bash
python scripts/visualize_3d_structure.py --pdb_file data/raw/1CRN.pdb
```

### Feature Distributions
Analyze node feature distributions across structure classes:
```bash
python scripts/visualize_features.py --graph_file data/graphs/1CRN.pt
```

### Embedding Visualization
Visualize the learned node embeddings using t-SNE:
```bash
python scripts/visualize_embeddings.py --model_file models/checkpoints/gnn_model.pt --data_file data/graphs/1CRN.pt
```

### Training Progress
Monitor training metrics over epochs:
```bash
python scripts/visualize_training.py --log_file models/training_log.csv
```

All visualizations are saved to the `visualizations/` directory.

## 📁 Colab Version
To run in Google Colab:
- Upload your `.pdb` files directly to Colab
- Use the included `colab_pipeline.py` script instead of `run_pipeline.py`
- Skip DSSP (optional fallback: label residues manually)
- Comment out system calls and use `!`-based bash commands

Example notebook setup:
```python
# Upload PDB files
from google.colab import files
uploaded = files.upload()  # Select your PDB files

# Run the Colab-friendly pipeline
%run colab_pipeline.py
```

## 🧪 Example PDB IDs Used
The project includes example data for two proteins:
- `1CRN` (Crambin): A small, well-characterized protein (46 residues)
- `4HHB` (Hemoglobin subunit): A classic oxygen-binding protein (574 residues)

You can add additional proteins by adding their PDB IDs to the list in `run_pipeline.py`.

## 📚 Technical Details

### Graph Construction
- **Nodes**: Protein residues (with Cα atom coordinates as positions)
- **Edges**: Connected if Cα atoms are within 8Å of each other
- **Node Features**: XYZ coordinates (can be extended with residue properties)
- **Node Labels**: Secondary structure type (Helix=0, Strand=1, Coil=2)

### Model Architecture
- **GNN Type**: Graph Convolutional Network (GCN)
- **Hidden Dimensions**: 32
- **Layers**: 2 GCN layers with ReLU activation
- **Training**: Adam optimizer, Cross-entropy loss

## 🔄 Contributing
Contributions are welcome! Some ideas for extension:
- Add more node features (hydrophobicity, charge, etc.)
- Implement alternative GNN architectures (GAT, GraphSAGE)
- Add ensemble prediction across multiple models
- Support additional secondary structure notation schemes

---

© Muhammad Abdullah – GitHub: [@muhabdullahd](https://github.com/muhabdullahd)

![Protein Graph Example](/visualizations/1CRN_graph.png)