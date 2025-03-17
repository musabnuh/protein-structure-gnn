# scripts/visualize_training.py
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

def visualize_training(log_file):
    """Visualize training progress from a log file."""
    # Load log data
    df = pd.read_csv(log_file)
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Save plot
    output_path = os.path.join('visualizations', "training_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training visualization saved to {output_path}")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', required=True, help='Path to training log CSV file')
    args = parser.parse_args()
    visualize_training(args.log_file)