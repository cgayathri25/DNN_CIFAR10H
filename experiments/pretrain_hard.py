import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import argparse

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_splits
from src.models.backbone import get_backbone

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=str, default='cifar10', choices=['random', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Path Resolution
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, 'backbone_hard.pth')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load splits - dataset.py handles returning integers for this script
    train_set, val_set, _ = get_splits(root_dir=os.path.join(base_dir, 'data'), transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Initialize Model
    if args.init == 'random':
        model = get_backbone(num_classes=10, pretrained=False).to(device)
    else:
        model = get_backbone(num_classes=10, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting Pretraining (Init: {args.init})...")
    print(f"Checkpoints will save to: {save_path}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # labels are integers here, so CrossEntropyLoss works directly
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # SAVE EVERY EPOCH: Ensures the file exists even if Stage 3 crashes later
        torch.save(model.state_dict(), save_path)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f} - Model Saved")

    print(f"\n!!! PRETRAINING COMPLETE !!!")
    print(f"Final model verified at: {save_path}")

if __name__ == "__main__":
    train()