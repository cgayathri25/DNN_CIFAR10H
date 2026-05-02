import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import os

def get_splits(root_dir='data', transform=None, val_split=0.1):
    # Dynamic path resolution to handle running from project root or experiments folder
    current_path = os.path.abspath(os.getcwd())
    base_path = os.path.dirname(current_path) if current_path.endswith('experiments') else current_path
    final_root = os.path.join(base_path, 'data')
    
    cifar_path = os.path.join(final_root, 'cifar10')
    
    # Load CIFAR-10
    full_dataset = datasets.CIFAR10(root=cifar_path, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=cifar_path, train=False, download=False, transform=transform)
    
    # Load CIFAR-10H soft labels
    soft_labels_path = os.path.join(final_root, 'cifar10h_labels.npy')
    soft_labels_all = np.load(soft_labels_path)
    
    # Generate consistent indices for Train/Val split
    num_train = len(full_dataset)
    indices = list(range(num_train))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(np.floor(val_split * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    
    class CIFAR10H(torch.utils.data.Dataset):
        def __init__(self, base_dataset, labels=None):
            self.base_dataset = base_dataset
            # Convert labels to float for KL/JSD loss compatibility
            self.labels = torch.from_numpy(labels).float() if labels is not None else None
            
        def __getitem__(self, index):
            img, hard_label = self.base_dataset[index]
            # Return soft labels if available (Test set), else return hard labels (Train/Val)
            label = self.labels[index] if self.labels is not None else hard_label
            return img, label
            
        def __len__(self):
            return len(self.base_dataset)

    # SUCCESS LOGIC: Wrap training and val with standard hard labels
    train_set = Subset(CIFAR10H(full_dataset, labels=None), train_idx)
    val_set = Subset(CIFAR10H(full_dataset, labels=None), val_idx)
    
    # Test set gets the final 10,000 soft labels from the .npy file
    test_set = CIFAR10H(test_dataset, labels=soft_labels_all[-10000:])
    
    return train_set, val_set, test_set