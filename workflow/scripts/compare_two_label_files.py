"""
true: tname
pred: pname
"""
from random import sample

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef

from ccount_utils.blob import load_blobs, show_rand_crops, plot_flat_crop, sort_blobs, intersect_blobs
from ccount_utils.blob import setdiff_blobs, get_blob_statistics
from ccount_utils.img import equalize, read_czi, parse_image_obj

def argparse_args():
    parser = argparse.ArgumentParser(description="Compare two label files (crops.npy.gz or blobs.npy.gz)")
    parser.add_argument('-t', type=str, required=True, help='Path to ground truth blobs (npy.gz)')
    parser.add_argument('-tname', type=str, required=True, help='Name of the ground truth blobs in plots')
    parser.add_argument('-p', type=str, required=True, help='Path to predicted blobs (npy.gz)')
    parser.add_argument('-pname', type=str, required=True, help='Name of the predicted blobs in plots')
    parser.add_argument('-sample', type=str, required=True, help='Sample name for output files')
    return parser.parse_args()


def list_stats(list):
    # Get unique values and their counts
    unique_values, counts = np.unique(list, return_counts=True)

    # Print the results
    for value, count in zip(unique_values, counts):
        print(f"Value {value}: {count} occurrences")


def get_labels(blobs):
    labels = blobs[:, 3]
    print('before processing')
    list_stats(labels)

    labels[labels == 9] = 1
    print('after processing')
    list_stats(labels)
    print()
    return labels


def compare(labels_t, labels_p, tname, pname, sample):
    # Compute confusion matrix
    cm = confusion_matrix(labels_t, labels_p)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel(pname)  # labels_p
    plt.ylabel("GT: " + tname)  # labels_t
    plt.title(sample)
    plt.savefig(f'{sample}.{tname}.{pname}.confusion.pdf')

    # Compute metrics
    precision = precision_score(labels_t, labels_p, zero_division=0)
    recall = recall_score(labels_t, labels_p, zero_division=0)
    f1 = f1_score(labels_t, labels_p, zero_division=0)
    mcc = matthews_corrcoef(labels_t, labels_p)

    # Print results
    print(f'Truth: {tname}')
    print(f'Pred:  {pname}')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    # Save to CSV using pandas
    metrics_dict = {
        "Truth": [tname],
        "Pred": [pname],
        "Precision": [precision],
        "Recall": [recall],
        "F1-score": [f1],
        "MCC": [mcc]
    }

    df = pd.DataFrame(metrics_dict)
    csv_filename = f"{sample}.{tname}_vs_{pname}_metrics.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Metrics saved to {csv_filename}")

def main():
    args = argparse_args()

    blobs_t = load_blobs(args.t)
    blobs_p = load_blobs(args.p)

    tname = args.tname
    pname = args.pname
    sample = args.sample

    blobs_t2, blobs_p2 = intersect_blobs(blobs_t, blobs_p)
    if len(blobs_t2) == 0 or len(blobs_p2) == 0:
        raise Exception("Blobs do not intersect")
    if len(blobs_t) != len(blobs_p):
        raise Exception("Blobs do not match")
    if len(blobs_t) != len(blobs_t2) or len(blobs_p) != len(blobs_p2):
        raise Exception("Blobs intersection smaller than original")

    labels_t = get_labels(blobs_t2)
    labels_p = get_labels(blobs_p2)

    compare(labels_t, labels_p, tname, pname, sample)




if __name__ == "__main__":
    main()