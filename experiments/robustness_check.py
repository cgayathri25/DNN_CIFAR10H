import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataset import get_splits
from src.models.backbone import get_backbone
from src.utils import calculate_shannon_entropy

def add_noise(inputs, noise_factor=0.3):
    noise = torch.randn_like(inputs) * noise_factor
    return torch.clamp(inputs + noise, 0., 1.)

def evaluate_robustness(model, loader, device, noise_level=0.0):
    model.eval()
    total_entropy = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if noise_level > 0:
                images = add_noise(images, noise_level)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            entropy = calculate_shannon_entropy(probs)
            total_entropy.extend(entropy)
            
            _, predicted = torch.max(outputs.data, 1)
            hard_labels = torch.argmax(labels, dim=1)
            total += labels.size(0)
            correct += (predicted == hard_labels).sum().item()
            
    return 100 * correct / total, np.mean(total_entropy)

def run_comparison():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.ToTensor()])
    _, _, test_set = get_splits(root_dir='../data', transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    hard_model = get_backbone(num_classes=10).to(device)
    soft_model = get_backbone(num_classes=10).to(device)
    
    hard_model.load_state_dict(torch.load('../checkpoints/backbone_hard.pth', map_location=device))
    soft_model.load_state_dict(torch.load('../checkpoints/model_soft_tuned.pth', map_location=device))

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    hard_accs, soft_accs = [], []
    hard_entropies, soft_entropies = [], []

    print("Running Robustness Comparison...")
    for level in noise_levels:
        h_acc, h_ent = evaluate_robustness(hard_model, test_loader, device, noise_level=level)
        s_acc, s_ent = evaluate_robustness(soft_model, test_loader, device, noise_level=level)
        
        hard_accs.append(h_acc)
        soft_accs.append(s_acc)
        hard_entropies.append(h_ent)
        soft_entropies.append(s_ent)
        
        print(f"Noise {level}: Acc (H/S) {h_acc:.1f}/{s_acc:.1f}% | Entropy (H/S) {h_ent:.2f}/{s_ent:.2f}")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Gaussian Noise Level')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(noise_levels, hard_accs, 'o--', label='Hard Baseline (Acc)', color='tab:red')
    ax1.plot(noise_levels, soft_accs, 's-', label='Soft-Tuned (Acc)', color='tab:green')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Mean Predicted Entropy', color='tab:purple')
    ax2.plot(noise_levels, hard_entropies, 'x--', label='Hard Baseline (Entropy)', color='salmon')
    ax2.plot(noise_levels, soft_entropies, 'd-', label='Soft-Tuned (Entropy)', color='purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')

    plt.title('Robustness Check: Accuracy and Entropy Response to Noise')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('../outputs/robustness_plot.png')
    plt.show()

if __name__ == "__main__":
    run_comparison()