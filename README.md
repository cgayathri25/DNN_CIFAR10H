# Human-Centric Uncertainty Alignment in Image Classification

This project explores the alignment between Deep Learning models and human perceptual uncertainty. By leveraging the CIFAR-10H dataset, we move beyond traditional "hard-label" classification to train a model that understands when humans disagree and why.

## Project Overview

Traditional AI models are often forced to be 100% confident in a single class, even when an image is blurry or ambiguous. This project implements a Soft-Label Fine-Tuning approach using a modified ResNet-18 architecture. The goal is to minimize the distributional gap between model predictions and human annotator disagreement.

### Key Features

* **Architecture**: ResNet-18 modified with an Identity Maxpool and 3x3 initial convolution for 32x32 image compatibility.
* **Loss Function**: KL-Divergence based soft-label training.
* **Evaluation**: Multi-metric approach focusing on distribution matching (KL, JSD, Cosine Similarity), entropy correlation (Pearson, Spearman), and ranking precision.
* **Explainability**: Grad-CAM heatmaps used to analyze model focus in high-entropy failure cases.

## Core Metrics and Results

### Metric 1: Distribution Matching

We evaluate how closely the model's output probability distribution matches the human soft-labels.

| Metric | Result (Mean ± Std) |
| :--- | :--- |
| KL Divergence | 1.6065 ± 1.3607 |
| JS Divergence | 0.3376 ± 0.2018 |
| Cosine Similarity | 0.5355 ± 0.3591 |

### Metric 2: Entropy Correlation

To measure if the model "knows what it doesn't know," we correlate predicted entropy against human entropy.

* **Pearson Correlation**: 0.1217
* **Spearman Correlation**: 0.1147

### Metric 3: Precision @ K

We test the model's ability to identify the top $K$ most "difficult" (high-entropy) images in the dataset.

* **Precision@100**: 0.0400
* **Precision@500**: 0.3080

## Explainability and Failure Analysis

Using Grad-CAM, we identified that model failures often stem from "Representational Ambiguity." For example, in Image #364, the model correctly identifies high uncertainty because the vertical silhouette of a rider at 32x32 resolution shares features with both 'Horse' and 'Bird' classes.

### Ablation Study

We compared a Randomly Initialized backbone against a Pretrained (CIFAR-10) backbone. The pretrained version achieved a 100% improvement in Pearson Correlation, proving that robust feature representation is essential for learning human-like uncertainty.

## Technical Implementation

### Mathematical Framework

The model is trained to minimize the KL Divergence between predicted probabilities $P$ and human soft-labels $Q$:

$$D_{KL}(Q \parallel P) = \sum_{i} Q(i) \log \left( \frac{Q(i)}{P(i) + \epsilon} \right)$$

Predictive uncertainty is quantified using Shannon Entropy:

$$H(P) = -\sum_{i} P(i) \log_2(P(i) + \epsilon)$$

## Installation and Usage

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/cgayathri25/DNN_CIFAR10H
    ```

2.  **Ensure Data Structure**:
    * `data/`: Place `test_images.npy` and `test_labels_soft.npy`.
    * `checkpoints/`: Place `model_soft_tuned.pth`.
    * `outputs/`: Directory for generated plots.

3.  **Run the Analysis**: Open the Jupyter Notebook and execute all cells. The code includes `np.errstate` handling to ensure a warning-free execution environment.

## Acknowledgments

This project builds upon the foundations laid by Northcutt et al. regarding label errors and the destabilization of benchmarks, using the CIFAR-10H dataset to provide a more robust ground truth for human-AI alignment.