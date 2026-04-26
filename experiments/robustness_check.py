import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_splits
from src.models.backbone import get_backbone

def add_noise(inputs, noise_factor=0.3):
    noise = torch.randn_like(inputs) * noise_factor
    return torch.clamp(inputs + noise, 0., 1.)

def evaluate(model, loader, device, noise_level=0.0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if noise_level > 0:
                images = add_noise(images, noise_level)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            hard_labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == hard_labels).sum().item()
    return 100 * correct / total

def run_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    _, _, test_set = get_splits(root_dir='../data', transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    hard_model = get_backbone(num_classes=10).to(device)
    soft_model = get_backbone(num_classes=10).to(device)
    
    hard_model.load_state_dict(torch.load('../checkpoints/backbone_hard.pth', map_location=device))
    soft_model.load_state_dict(torch.load('../checkpoints/model_soft_tuned.pth', map_location=device))

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    hard_accs, soft_accs = [], []

    print("Running Robustness Comparison...")
    for level in noise_levels:
        h_acc = evaluate(hard_model, test_loader, device, noise_level=level)
        s_acc = evaluate(soft_model, test_loader, device, noise_level=level)
        hard_accs.append(h_acc)
        soft_accs.append(s_acc)
        print(f"Noise {level}: Hard Model {h_acc:.2f}% | Soft Model {s_acc:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, hard_accs, 'o-', label='Hard Baseline', color='red')
    plt.plot(noise_levels, soft_accs, 's-', label='Soft-Tuned (Human)', color='green')
    plt.title('Model Robustness: Hard vs. Soft Labels')
    plt.xlabel('Gaussian Noise Level')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../outputs/robustness_plot.png')
    plt.show()

if __name__ == "__main__":
    run_comparison()