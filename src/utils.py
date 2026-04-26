import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def calculate_shannon_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    
    epsilon = 1e-12
    entropy = -np.sum(probs * np.log2(probs + epsilon), axis=1)
    return entropy

def generate_entropy_histogram(entropies, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(entropies, bins=30, kde=True, color='teal')
    plt.title("Distribution of Human Annotator Disagreement (CIFAR-10H)")
    plt.xlabel("Shannon Entropy (Bits)")
    plt.ylabel("Number of Images")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path)
    print(f"Histogram saved successfully to: {save_path}")
    plt.close()

def get_ambiguity_indices(entropies, n=4):
    sorted_idx = np.argsort(entropies)
    low_entropy_idx = sorted_idx[:n]
    high_entropy_idx = sorted_idx[-n:]
    return low_entropy_idx, high_entropy_idx

def plot_soft_distribution(prob_vector, class_names, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, prob_vector, color='orange')
    plt.xticks(rotation=45)
    plt.ylabel("Probability")
    plt.title("Human Annotator Distribution")
    if save_path:
        plt.savefig(save_path)
    plt.show()