import numpy as np
import pandas as pd
import sys

from ccount_utils.blob import load_blobs, save_crops
from ccount_utils.blob import get_blob_statistics
from ccount_utils.clas import F1_calculation
from ccount_utils.blob import intersect_blobs


def logical_and(list1, list2):
    """
    Performs element-wise logical AND operation on two lists.

    Args:
      list1: The first list of boolean values.
      list2: The second list of boolean values.

    Returns:
      A new list containing the result of the logical AND operation for each element.
    """

    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")

    return [a and b for a, b in zip(list1, list2)]


print("usage:  python crops_vote.py A.npy.gz B.npy.gz output.npy.gz")
print("note0: A and B should be for the same blobs")
print("note1: calculate based on the intersection of blobs in two files, \
    different blobs will be discarded\n")
print("note2: only output yes when both yes\n\n")


def read_labels(blobs):
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels


sys.exit("This is obsolete")

name1 = sys.argv[1]  # label set A
name2 = sys.argv[2]  # label set B
outname = sys.argv[3]

blobs1 = load_blobs(name1)
blobs2 = load_blobs(name2)

labels1 = read_labels(blobs1)
labels2 = read_labels(blobs2)
precision, recall, F1 = F1_calculation(labels1, labels2)

binary1 = [x == 1 for x in labels1]
binary2 = [x == 1 for x in labels2]
binary_output = logical_and(binary1, binary2)
label_output = [int(x) for x in binary_output]

blobs1[:, 3] = label_output

save_crops(blobs1, outname)
