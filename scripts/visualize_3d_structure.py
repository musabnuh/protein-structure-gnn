# scripts/visualize_3d_structure.py
import nglview as nv
import biotite.structure.io as bsio
import argparse

def visualize_3d_structure(pdb_file):
    """Visualize a protein's 3D structure from a PDB file."""
    # Load structure
    structure = bsio.load_structure(pdb_file)
    
    # Create viewer
    view = nv.show_structure(structure)
    view.add_representation('cartoon', selection='protein')
    view.add_representation('ball+stick', selection='hetero')
    
    return view

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', required=True, help='Path to PDB file')
    args = parser.parse_args()
    
    view = visualize_3d_structure(args.pdb_file)
    view._display_image()