import sys, gc
import numpy as np
import pandas as pd
from ccount.blob.io import load_blobs, save_crops
from ccount.blob.misc import get_label_statistics
from ccount.blob.intersect import intersect_blobs
from ccount.clas.metrics import F1_calculation


def read_labels(blobs):
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels


def count_votes(labels):
    """
    Sum elements from multiple lists, e.g. [L1, L2, L3]

    Args:
      lists: A list of lists.
      e.g.
      L1 = [0, 1, 0]
      L2 = [0, 1, 0]
      L3 = [0, 1, 1]
      labels = [L1, L2, L3]

    Returns:
      A new list containing the combined elements.
      e.g.
      [0, 3, 1]
    """
    return [sum(values) for values in zip(*labels)]


def threshold_counts(COUNTS, n=1):
    print("threshold", n)
    L = [1 if value >= n else 0 for value in COUNTS]
    print("num positive output votes:", sum(L))
    return L


def main():
    # Check if there are enough arguments
    if len(sys.argv) != 6:
        print("Usage: vote.py <output.npy.gz> <min-vote> <A.npy.gz> <B.npy.gz> <C.npy.gz>")
        sys.exit("cmd error")
    else:
        print(sys.argv)

    IN1 = load_blobs(sys.argv[2 + 1])
    L1 = read_labels(IN1)
    del IN1
    gc.collect()

    IN2 = load_blobs(sys.argv[2 + 2])
    L2 = read_labels(IN2)
    del IN2
    gc.collect()

    IN3 = load_blobs(sys.argv[2 + 3])
    L3 = read_labels(IN3)

    COUNTS = count_votes([L1, L2, L3])
    L_FINAL = threshold_counts(COUNTS, n=int(sys.argv[2]))

    IN3[:, 3] = L_FINAL
    print("\noutput crops stats:")
    get_label_statistics(IN3)
    save_crops(IN3, sys.argv[1])

# Run the main function
if __name__ == "__main__":
    main()
