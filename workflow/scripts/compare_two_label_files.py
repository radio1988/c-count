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
    parser.add_argument('-pred', type=str, required=True, help='Path to predicted blobs (npy.gz)')
    parser.add_argument('-truth', type=str, required=True, help='Path to ground truth blobs (npy.gz)')
    parser.add_argument('-sample', type=str, required=True, help='Sample name')
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


def compare(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["neg", "pos"], yticklabels=["neg", "pos"])
    plt.xlabel(name2)  # y_pred
    plt.ylabel(name1)  # y_true
    plt.title(sample)
    plt.savefig(f'{sample}.{name1}.{name2}.confusion.pdf')

    # Compute metrics
    precision = precision_score(labels1, labels2, zero_division=0)
    recall = recall_score(labels1, labels2, zero_division=0)
    f1 = f1_score(labels1, labels2, zero_division=0)
    mcc = matthews_corrcoef(labels1, labels2)

    # Print results
    print(f'between {name1} & {name2}')
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    # Save to CSV using pandas
    metrics_dict = {
        "Comparison": [f"{name1} vs {name2}"],
        "Precision": [precision],
        "Recall": [recall],
        "F1-score": [f1],
        "MCC": [mcc]
    }

    df = pd.DataFrame(metrics_dict)
    csv_filename = f"{sample}.{name1}.{name2}_metrics.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Metrics saved to {csv_filename}")

def main():
    args = argparse_args()
    name1 = args.pred
    name2 = args.truth
    sample = args.sample

    blobs1 = load_blobs(name1)
    locs1 = blobs1[:, :4]
    print()
    blobs2 = load_blobs(name2)
    locs2 = blobs2[:, :4]  # yxrL
    print()

    blobs1b, blobs2b = intersect_blobs(blobs1, blobs2)
    if blobs1b is None or blobs2b is None:
        raise Exception("No intersection found.")
    if len(blobs1b) != len(blobs1) or len(blobs2b) != len(blobs2):
        raise Exception("Warning: The number of blobs in the intersection is not equal to the original blobs.")

    labels1b = get_labels(blobs1b)
    labels2b = get_labels(blobs2b)

    compare(labels1b, labels2b)