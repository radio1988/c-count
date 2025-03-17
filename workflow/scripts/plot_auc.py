"""
This script computes and plots the Precision-Recall curve for a binary classification problem.

python plot_auc.py -prediction res/2_count_on_validationSet/1.0.rep5.probs.txt -groundtruth data/AF.val.npy.gz

prob.txt:
p_no p_yes
0.999999 0.000001
0.999999 0.000001
0.999998 0.000002
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import argparse
import sys
import os
from ccount_utils.blob import load_blobs
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Precision-Recall curve")
    parser.add_argument('-prediction', type=str, required=True, help='probs.txt')
    parser.add_argument('-groundtruth', type=str, required=True, help='val.npy.gz')
    parser.add_argument('-output', type=str, default='precision_recall_curve.pdf', help='Output PDF file for the plot')
    return parser.parse_args()


def main():
    args = parse_args()

    blobs = load_blobs(args.groundtruth)
    y_true = blobs[:, 3]
    y_scores = pd.read_csv(args.prediction,  delimiter=' ', header=0).iloc[:, 1].values  # second column

    if y_scores.shape[0] != y_true.shape[0]:
        print(f"Ground truth shape: {y_true.shape}, Predictions shape: {y_scores.shape}")
        sys.exit("Error: The number of predictions and ground truth labels must match.")

    if not np.all(np.isin(y_true, [0, 1])):
        print(f"Ground truth labels: {np.unique(y_true)}")
        sys.exit("Error: Ground truth labels must be binary (0 or 1).")

    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    # Compute AUC-PR
    auc_pr = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, marker='.', label=f'AUC-PR = {auc_pr:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    plt.legend()

    outname = 'output/test.pdf'
    output_dir = os.path.dirname(outname)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(outname, format="pdf", bbox_inches="tight")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(args.output, format="pdf", bbox_inches="tight")

    plt.figure(figsize=(12, 6))  # Set figure size
    sns.histplot(y_scores, bins=50, log_scale=(True, False))
    plt.ylim(1, 500)  # Example: from 1 to 1000, adjust based on your data
    plt.xlabel("P-value (log scale)")
    plt.ylabel("Count (log scale)")
    plt.title("Histogram of P-values (Log-Log Scale)")
    plt.savefig(outname.replace("pdf", "hist.pdf"), format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
