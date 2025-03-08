import sys
import numpy as np

from ccount_utils.blob import load_blobs
from ccount_utils.clas import F1_calculation
from ccount_utils.blob import intersect_blobs


print("usage:  python f1_score.py prediction.npy.gz truth.npy.gz > f1_score.t.txt")
print("note0: the two files should have blobs or crops in them")
print("note1: calculate based on the intersection of blobs in two files, \
    different blobs will be discarded\n\n")

def read_labels(blobs):
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels

name1 = sys.argv[1]  # prediction
name2 = sys.argv[2]  # truth


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

precision, recall, F1 = F1_calculation(y_pred, y_true)


from ccount_utils.clas import calculate_AUC_PR, calculate_MCC_Max
auc_pr = calculate_AUC_PR(y_true, y_pred)  # average_precision_score computes the precision-recall AUC, which is equivalent to AUC-PR.
print(f"AUC-PR: {auc_pr:.4f}")

mcc_min = calculate_MCC_Max(y_true, y_pred)
print(f"MCC-min: {mcc_min:.4f}")
