import os
import sys
import argparse

def main():
    print("--- DNN PROJECT: CIFAR-10H PIPELINE ---")
    
    # Run Entropy Analysis (Phase 1)
    print("\n[1/4] Running Data Exploration & Entropy Analysis...")
    os.system("python -c \"import sys; sys.path.append('src'); from utils import calculate_shannon_entropy, generate_entropy_histogram; import numpy as np; labels = np.load('data/cifar10h_labels.npy'); e = calculate_shannon_entropy(labels); generate_entropy_histogram(e, 'outputs/entropy_histogram.png')\"")

    # Run Pretraining (Phase 2)
    print("\n[2/4] Running Hard-Label Pretraining...")
    os.system("python experiments/pretrain_hard.py")

    # Run Finetuning (Phase 3)
    print("\n[3/4] Running Soft-Label Finetuning...")
    os.system("python experiments/finetune_soft.py")

    # Run Robustness Check (Phase 4)
    print("\n[4/4] Generating Robustness Comparison...")
    os.system("python experiments/robustness_check.py")

    print("\n--- ALL PHASES COMPLETE. CHECK THE 'outputs/' FOLDER ---")

if __name__ == "__main__":
    main()