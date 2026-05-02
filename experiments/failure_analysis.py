import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import calculate_shannon_entropy
from src.dataset import get_splits

def find_failures():
    # Load data
    true_dist = np.load('data/cifar10h_labels.npy')[50000:]
    pred_dist = np.load('outputs/test_predictions.npy')
    
    # Calculate Entropies
    true_ent = calculate_shannon_entropy(true_dist)
    pred_ent = calculate_shannon_entropy(pred_dist)
    
    # 1. Model is certain, but Humans are confused (Underestimation)
    under_idx = np.argsort(true_ent - pred_ent)[-3:] 
    
    # 2. Model is confused, but Humans are certain (Overestimation)
    over_idx = np.argsort(pred_ent - true_ent)[-3:]
    
    # Load raw images for saving
    transform = transforms.Compose([transforms.ToTensor()])
    _, _, test_set = get_splits(root_dir='data', transform=transform)

    os.makedirs('outputs/failures', exist_ok=True)

    def save_set(indices, folder_name):
        for idx in indices:
            img, _ = test_set[idx]
            plt.imshow(img.permute(1, 2, 0))
            plt.title(f"Index {idx} | Human Ent: {true_ent[idx]:.2f} | Model Ent: {pred_ent[idx]:.2f}")
            plt.savefig(f'outputs/failures/{folder_name}_{idx}.png')
            plt.close()

    save_set(under_idx, "underestimated")
    save_set(over_idx, "overestimated")
    print(f"Failure images saved to outputs/failures/")

if __name__ == "__main__":
    find_failures()