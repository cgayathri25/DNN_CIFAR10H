import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', type=str, default=None, choices=['init', 'head', 'data'])
    args = parser.parse_args()

    print("--- DNN PROJECT: HUMAN DISAGREEMENT PIPELINE ---")
    
    print("\n[1/5] Running Compulsory Data Exploration & Visualizations...")
    os.system("python3 -c \"import sys; sys.path.append('src'); from utils import calculate_shannon_entropy, generate_entropy_histogram, plot_per_class_entropy; import numpy as np; labels = np.load('data/cifar10h_labels.npy'); e = calculate_shannon_entropy(labels); generate_entropy_histogram(e, 'outputs/entropy_histogram.png'); classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']; true_labels = np.argmax(labels, axis=1); plot_per_class_entropy(true_labels, e, classes, 'outputs/per_class_entropy.png')\"")

    if args.ablation == 'init':
        print("\n[Ablation] Running Backbone Initialization Comparison (Compulsory A)...")
        os.system("python3 experiments/pretrain_hard.py --init random --epochs 5")
        os.system("python3 experiments/pretrain_hard.py --init cifar10 --epochs 5")
    else:
        print("\n[2/5] Running Mandatory Hard-Label Pretraining...")
        os.system("python3 experiments/pretrain_hard.py")

    print("\n[3/5] Running Multi-Loss Soft-Label Finetuning (KL, JSD, Custom)...")
    os.system("python3 experiments/finetune_soft.py --loss kl")
    os.system("python3 experiments/finetune_soft.py --loss jsd")
    os.system("python3 experiments/finetune_soft.py --loss custom")

    print("\n[4/5] Running Robustness Checks (Gaussian Noise & Entropy Response)...")
    os.system("python3 experiments/robustness_check.py")

    print("\n[5/5] Performing Final Evaluation (Precision@K & Failure Case Analysis)...")
    eval_cmd = (
        "import sys; sys.path.append('src'); "
        "from utils import calculate_shannon_entropy, calculate_precision_at_k, calculate_entropy_correlations; "
        "import torch; import numpy as np; "
        "true_dist = np.load('data/cifar10h_labels.npy')[-10000:]; "
        "true_ent = calculate_shannon_entropy(true_dist); "
        "pred_dist = np.load('outputs/test_predictions.npy'); "
        "pred_ent = calculate_shannon_entropy(pred_dist); "
        "print('Entropy Correlation:', calculate_entropy_correlations(true_ent, pred_ent)); "
        "print('Precision Results:', calculate_precision_at_k(true_ent, pred_ent))"
    )
    os.system(f"python3 -c \"{eval_cmd}\"")

    print("\nPIPELINE COMPLETE")

if __name__ == "__main__":
    main()