import torch
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.backbone import get_backbone
from src.dataset import get_splits

def run_grad_cam():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = get_backbone(num_classes=10).to(device)
    model.load_state_dict(torch.load('../checkpoints/model_soft_tuned.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    _, _, test_set = get_splits(root_dir='../data', transform=transform)
    
    img_tensor, label_probs = test_set[4331] 
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    target_class = torch.argmax(label_probs).item()
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = img_tensor.permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.imshow(visualization)
    plt.title(f"Grad-CAM for Class {target_class}")
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/grad_cam_high_ent.png')
    print("Grad-CAM saved to outputs/grad_cam_results.png")

if __name__ == "__main__":
    run_grad_cam()