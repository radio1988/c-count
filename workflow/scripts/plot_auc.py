"""
This script computes and plots the Precision-Recall curve for a binary classification problem.

python plot_auc.py -prediction res/2_count_on_validationSet/1.0.rep5.probs.txt -groundtruth data/AF.val.npy.gz
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import argparse
import sys
from ccount_utils.blob import load_blobs


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Plot Precision-Recall curve")
    parser.add_argument('-prediction', type=str, required=True, help='Path to predicted probabilities file')
    parser.add_argument('-groundtruth', type=str, required=True, help='Path to ground truth file')
    return parser.parse_args()


def main():
    args = parse_args()

    y_true = load_blobs(args.groundtruth)
    y_scores = pd.read_csv(fname,  delimiter=' ', header=1).iloc[:, 1].values

    if y_scores.shape[0] != y_true.shape[0]:
        sys.exit("Error: The number of predictions and ground truth labels must match.")

    if not np.all(np.isin(y_true, [0, 1])):
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
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()

    # Save plot to PDF
    plt.savefig("precision_recall_curve.pdf", format="pdf", bbox_inches="tight")

    # Show the plot (optional)
    plt.show()

if __name__ == "__main__":
    main()
