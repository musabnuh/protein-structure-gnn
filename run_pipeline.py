# Project: Protein Secondary Structure Classification using GNNs
# Structure: Scaffold for Automation

# Directory Structure:
# - data/: Raw and processed protein structure files
# - scripts/: Python scripts for each pipeline step
# - models/: Model definitions and training code
# - results/: Output files, metrics, and visualizations
# - run_pipeline.py: Orchestration script

# run_pipeline.py
import os
import subprocess

# Step 1: Download PDB Files
def download_pdb(pdb_ids):
    for pdb_id in pdb_ids:
        os.makedirs("data/raw", exist_ok=True)
        os.system(f"wget https://files.rcsb.org/download/{pdb_id}.pdb -O data/raw/{pdb_id}.pdb")

# Step 2: Run DSSP
def run_dssp(pdb_id):
    os.makedirs("data/dssp", exist_ok=True)
    pdb_path = f"data/raw/{pdb_id}.pdb"
    dssp_path = f"data/dssp/{pdb_id}.dssp"
    os.system(f"mkdssp -i {pdb_path} -o {dssp_path}")

# Step 3: Graph Construction
def construct_graph(pdb_id):
    script = "scripts/construct_graph.py"
    os.system(f"python {script} --pdb_id {pdb_id}")

# Step 4: Train Model
def train_model():
    os.system("python models/train.py")

# Step 5: Evaluate and Visualize
def evaluate():
    os.system("python scripts/evaluate.py")

if __name__ == "__main__":
    pdb_ids = ["1CRN", "4HHB"]  # Sample PDB IDs

    download_pdb(pdb_ids)
    for pdb_id in pdb_ids:
        run_dssp(pdb_id)
        construct_graph(pdb_id)

    train_model()
    evaluate()
