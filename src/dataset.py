import torch
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class CIFAR10HDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform
        
        labels_path = os.path.join(self.root_dir, 'cifar10h_labels.npy')
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Label file not found at {labels_path}")
            
        self.soft_labels = np.load(labels_path)
        self.soft_labels = self.soft_labels / self.soft_labels.sum(axis=1, keepdims=True)
        
        cifar_path = os.path.join(self.root_dir, 'cifar-10-batches-py', 'test_batch')
        alt_path = os.path.join(self.root_dir, 'cifar-10-python', 'cifar-10-batches-py', 'test_batch')
        
        if os.path.exists(cifar_path):
            final_path = cifar_path
        elif os.path.exists(alt_path):
            final_path = alt_path
        else:
            raise FileNotFoundError(f"CIFAR-10 test_batch not found.")

        with open(final_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.images = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_splits(root_dir, transform=None):
    datasets.CIFAR10(root=root_dir, train=False, download=True)
    full_dataset = CIFAR10HDataset(root_dir, transform)
    
    indices = np.arange(10000)
    train_idx, temp_idx = train_test_split(indices, train_size=6000, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, train_size=2000, random_state=42)
    
    return Subset(full_dataset, train_idx), Subset(full_dataset, val_idx), Subset(full_dataset, test_idx)