import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
import argparse
import numpy as np

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_splits
from src.models.backbone import get_backbone
from src.losses import KLDivLoss, JSDLoss, CustomEntropyWeightedLoss

def finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='kl', choices=['kl', 'jsd', 'custom'])
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Absolute Path Resolution
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    output_dir = os.path.join(base_dir, 'outputs')
    data_dir = os.path.join(base_dir, 'data')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Use the absolute data path
    _, _, test_set = get_splits(root_dir=data_dir, transform=transform)
    soft_loader = DataLoader(test_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = get_backbone(num_classes=10).to(device)
    
    # Load the backbone from Stage 2
    backbone_path = os.path.join(checkpoint_dir, 'backbone_hard.pth')
    if os.path.exists(backbone_path):
        model.load_state_dict(torch.load(backbone_path, map_location=device))
        print(f"Loaded backbone from: {backbone_path}")
    else:
        print(f"CRITICAL ERROR: Backbone not found at {backbone_path}")
        return

    if args.loss == 'kl':
        criterion = KLDivLoss()
    elif args.loss == 'jsd':
        criterion = JSDLoss()
    elif args.loss == 'custom':
        criterion = CustomEntropyWeightedLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"\n--- Starting Fine-tuning: {args.loss.upper()} Loss ---")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in soft_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(soft_loader):.4f}")

    # Evaluation phase
    print("Generating predictions...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
    
    # Save predictions
    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    np.save(pred_path, np.concatenate(all_preds, axis=0))
    
    # Save the model with absolute path
    save_name = f'model_soft_{args.loss}.pth' if args.loss != 'kl' else 'model_soft_tuned.pth'
    final_save_path = os.path.join(checkpoint_dir, save_name)
    torch.save(model.state_dict(), final_save_path)
    
    print(f"SUCCESS: {args.loss.upper()} model saved to: {final_save_path}")
    print(f"Predictions updated at: {pred_path}\n")

if __name__ == "__main__":
    finetune()