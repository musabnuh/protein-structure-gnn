# Protein Secondary Structure Classification with GNNs

This project classifies secondary structure elements (helix, strand, coil) in proteins by representing them as residue-level graphs and applying a Graph Neural Network (GNN).

## 🔬 Motivation
This project demonstrates key concepts in:
- **Geometric Deep Learning / Graph Neural Networks**
- **Structural Biology & Computational Biology**
- **Protein Ensemble Modeling**

## 🚀 Project Pipeline
Run everything using:
```bash
python run_pipeline.py
```

### Pipeline Steps
1. **Download PDB files**
2. **Run DSSP** for secondary structure annotation
3. **Construct graphs** from residue coordinates
4. **Train a GNN** (GCN) to classify structure types
5. **Evaluate** model and report classification performance

## 📂 Folder Structure
```
.
├── data/             # Raw PDBs, DSSP files, and graphs
├── scripts/          # Graph construction, evaluation, visualization
├── models/           # GNN training code and checkpoints
├── run_pipeline.py   # Full pipeline orchestrator
└── README.md
```

## 📦 Requirements
Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install biopython scikit-learn networkx matplotlib
sudo apt install dssp  # for mkdssp command
```

## 🔍 Evaluation Output
After training, you’ll see a classification report:
```
              precision    recall  f1-score   support

       Helix                                           ...
      Strand                                           ...
        Coil                                           ...
```

## 📊 Visualization
To visualize a graph and see secondary structure labels by color:
```bash
python scripts/visualize_graph.py --graph_file data/graphs/1CRN.pt
```

## 📁 Colab Version
To run in Colab:
- Upload your `.pdb` files directly
- Skip DSSP (optional fallback: label residues manually)
- Comment out system calls and use `!`-based bash commands

## 🧪 Example PDB IDs Used
- `1CRN` (Crambin)
- `4HHB` (Hemoglobin subunit)

---

© Muhammad Abdullah – GitHub: [@muhabdullahd](https://github.com/muhabdullahd)
