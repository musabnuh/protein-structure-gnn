# scripts/construct_graph.py
import os
import argparse
import numpy as np
from Bio.PDB import PDBParser
import torch
import networkx as nx
from torch_geometric.data import Data

# Simplified secondary structure map from DSSP
SS_MAP = {"H": 0, "B": 1, "E": 1, "G": 0, "I": 0, "T": 2, "S": 2, "-": 2}

def load_dssp_labels(dssp_file):
    labels = []
    with open(dssp_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("  #  RESIDUE AA"):
                break
        for line in lines[lines.index(line)+1:]:
            if len(line) > 16:
                ss = line[16]
                labels.append(SS_MAP.get(ss, 2))  # Default to 'coil'
    return labels

def extract_residue_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
    return np.array(coords)

def build_graph(coords, labels, threshold=8.0):
    edge_index = []
    num_nodes = len(coords)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.linalg.norm(coords[i] - coords[j]) < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(coords, dtype=torch.float)
    y = torch.tensor(labels[:len(coords)], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

def main(pdb_id):
    pdb_file = f"data/raw/{pdb_id}.pdb"
    dssp_file = f"data/dssp/{pdb_id}.dssp"
    coords = extract_residue_coords(pdb_file)
    labels = load_dssp_labels(dssp_file)
    graph = build_graph(coords, labels)
    os.makedirs("data/graphs", exist_ok=True)
    torch.save(graph, f"data/graphs/{pdb_id}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_id', required=True)
    args = parser.parse_args()
    main(args.pdb_id)
