# Protein-Structure‑GNN  
A Graph Neural Network Approach to Protein Structure Modeling

---

## Project Summary

This project leverages Graph Neural Networks (GNNs) to model and interpret 3D protein structures. By representing proteins as graphs, where nodes are residues or atoms, and edges capture spatial or physicochemical proximity. The model can learn structure-function relationships in a data-driven way.

While the core model was built by my collaborator [@muhabdullahd](https://github.com/muhabdullahd), I joined to contribute meaningful documentation and annotations. I also helped refine biological and structural concepts in the codebase using insights drawn from my coursework in **Computational Structural Biochemistry** (CHEM 5420).

---

## Scientific Context

Proteins are not linear — their biological roles are encoded in their 3D structures, not just their sequences. Traditional bioinformatics approaches rely on alignments and homology modeling. This project takes a modern machine learning approach by encoding spatial interactions as graphs, then using GNN layers to learn meaningful patterns across protein structures.

Concepts relevant to this project that I studied in CHEM 5420 include:
- Structure prediction and comparative modeling
- Molecular dynamics and local conformational analysis
- Feature engineering from protein sequence and structural data
- Structure-based drug design, docking, and virtual screening
- Recent developments in AI/ML for protein folding and function prediction

---

## Features

- Converts PDB protein files into graph-based representations
- Implements a GNN architecture using PyTorch Geometric (e.g., `GCNConv`)
- Trains the model on labeled graph data for structure-related tasks
- Provides a modular, extensible framework for future development

---
