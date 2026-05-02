import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F

def calculate_shannon_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    epsilon = 1e-12
    entropy = -np.sum(probs * np.log2(probs + epsilon), axis=1)
    return entropy

def calculate_jsd(p, q):
    if isinstance(p, np.ndarray): p = torch.from_numpy(p)
    if isinstance(q, np.ndarray): q = torch.from_numpy(q)
    m = 0.5 * (p + q)
    def kl(a, b):
        return torch.sum(a * torch.log((a + 1e-12) / (b + 1e-12)), dim=1)
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return jsd.numpy()

def calculate_cosine_similarity(p, q):
    if isinstance(p, torch.Tensor): p = p.detach().cpu().numpy()
    if isinstance(q, torch.Tensor): q = q.detach().cpu().numpy()
    norm_p = np.linalg.norm(p, axis=1, keepdims=True)
    norm_q = np.linalg.norm(q, axis=1, keepdims=True)
    return np.sum(p * q, axis=1) / (norm_p.flatten() * norm_q.flatten() + 1e-12)

def calculate_precision_at_k(true_entropy, pred_entropy, k_list=[100, 200, 500]):
    results = {}
    true_top_indices = np.argsort(true_entropy)[::-1]
    for k in k_list:
        pred_top_indices = np.argsort(pred_entropy)[::-1][:k]
        actual_top_k = set(true_top_indices[:k])
        matches = len([i for i in pred_top_indices if i in actual_top_k])
        results[f"Precision@{k}"] = matches / k
    return results

def calculate_entropy_correlations(true_entropy, pred_entropy):
    pearson_val, _ = pearsonr(true_entropy, pred_entropy)
    spearman_val, _ = spearmanr(true_entropy, pred_entropy)
    return {"Pearson": pearson_val, "Spearman": spearman_val}

def generate_entropy_histogram(entropies, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(entropies, bins=30, kde=True, color='teal')
    plt.title("Histogram of True Entropy Values")
    plt.xlabel("Shannon Entropy (Bits)")
    plt.ylabel("Number of Images")
    plt.savefig(save_path)
    plt.close()

def plot_per_class_entropy(labels, entropies, class_names, save_path):
    avg_entropies = []
    for i in range(10):
        avg_entropies.append(np.mean(entropies[labels == i]))
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, avg_entropies, color='salmon')
    plt.title("Per-Class Average Entropy Plot")
    plt.ylabel("Average Entropy")
    plt.savefig(save_path)
    plt.close()

def get_ambiguity_indices(entropies, n=4):
    sorted_idx = np.argsort(entropies)
    return sorted_idx[:n], sorted_idx[-n:]

def plot_soft_distribution(prob_vector, class_names, title="Distribution", save_path=None):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, prob_vector, color='orange')
    plt.title(title)
    plt.ylabel("Probability")
    if save_path: plt.savefig(save_path)
    plt.close()