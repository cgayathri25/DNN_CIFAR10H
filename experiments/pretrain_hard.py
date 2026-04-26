import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt # Added for plotting
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import get_splits
from src.models.backbone import get_backbone

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20 
    batch_size = 64
    lr = 0.001
    
    # List to store history for plotting
    loss_history = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Path fix: ../data to go up from experiments/ to the root
    train_set, val_set, _ = get_splits(root_dir='../data', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = get_backbone(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Training on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            hard_labels = torch.argmax(labels, dim=1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, hard_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # --- SAVE WEIGHTS ---
    os.makedirs('../checkpoints', exist_ok=True) # Up one level to root checkpoints
    torch.save(model.state_dict(), '../checkpoints/backbone_hard.pth')
    print("Model saved to checkpoints/backbone_hard.pth")

    # --- GENERATE LOSS PLOT ---
    os.makedirs('../outputs', exist_ok=True)
    plt.figure()
    plt.plot(loss_history, 'o-', color='C0')
    plt.title('losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('../outputs/loss_curves.png')
    plt.show()
    print("Plot saved to outputs/loss_curves.png")

if __name__ == "__main__":
    train()