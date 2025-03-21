"""
Evaluate classification performance using F1 score, AUC-PR, and MCC
"""

import sys
import numpy as np
import argparse

from ccount_utils.blob import load_blobs
from ccount_utils.clas import F1_calculation
from ccount_utils.blob import intersect_blobs
from ccount_utils.clas import calculate_AUC_PR, calculate_MCC_Max
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def read_labels(blobs):
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate classification performance")
    parser.add_argument('-pred', type=str, required=True, help='Path to predicted blobs (npy.gz)')
    parser.add_argument('-truth', type=str, required=True, help='Path to ground truth blobs (npy.gz)')
    parser.add_argument('-output', type=str, default='results.txt', help='Output file for results')
    return parser.parse_args()

args = parse_args()
name2 = args.truth
name1 = args.pred

blobs1 = load_blobs(name1)
locs1 = blobs1[:, :4]
print()
blobs2 = load_blobs(name2)
locs2 = blobs2[:, :4]  # yxrL
print()

blobs1b, blobs2b = intersect_blobs(blobs1, blobs2)

labels1b = read_labels(blobs1b)
labels2b = read_labels(blobs2b)

y_true = labels2b
y_pred = labels1b

precision, recall, F1 = F1_calculation(y_pred, y_true) # will be printed

mcc_min = calculate_MCC_Max(y_true, y_pred)
print(f"MCC-MAX: {mcc_min:.4f}")

# Compute precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_pred)

# Compute AUC-PR
auc_pr = auc(recall, precision)
print(f"AUC-PR: {auc_pr:.4f}")

# Plot Precision-Recall curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, marker='.', label=f'AUC-PR = {auc_pr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig(args.output.replace('.txt', '.pdf'), bbox_inches='tight')
