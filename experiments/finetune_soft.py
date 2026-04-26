import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_splits
from src.models.backbone import get_backbone
from src.losses import SoftLabelLoss

def finetune():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_set, val_set, _ = get_splits(root_dir='../data', transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = get_backbone(num_classes=10).to(device)
    
    checkpoint_path = '../checkpoints/backbone_hard.pth'
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Successfully loaded the Hard-Pretrained Backbone!")
    else:
        print(f"Error: Could not find {checkpoint_path}")
        return

    criterion = SoftLabelLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Starting Fine-tuning on Human Soft Labels...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Fine-tune Epoch [{epoch+1}/10], Soft Loss: {running_loss/len(train_loader):.4f}")

    os.makedirs('../checkpoints', exist_ok=True)
    torch.save(model.state_dict(), '../checkpoints/model_soft_tuned.pth')
    print("Final Model Saved: ../checkpoints/model_soft_tuned.pth")

if __name__ == "__main__":
    finetune()