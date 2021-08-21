from ccount import load_blobs_db
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import sys 

print("usage:  python f1_score.py output/FL.t.pred.npy.gz ../data/FL.t.npy.gz  > f1_score.t.txt")
#todo: calculate mean/median of f1_scores of many pairs of blobs

def read_labels(fname):
    blobs = load_blobs_db(fname)
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels

def F1_calculation(predictions, labels):
    print("F1_calculation for sure labels only")
    idx = (labels == 1) | (labels == 0)  # sure only
    labels = labels[idx, ]
    predictions = predictions[idx, ]


    TP = np.sum(np.round(predictions * labels))
    PP = np.sum(np.round(labels))
    recall = TP / (PP + 1e-7)

    PP2 = np.sum(np.round(predictions))
    precision = TP/(PP2 + 1e-7)

    F1 = 2*((precision*recall)/(precision+recall+1e-7))

    print('Precition: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'.format(precision*100, recall*100, F1*100))

    return F1

name1 = sys.argv[1]  # prediction
name2 = sys.argv[2]  # truth
print(name1, name2)
labels1 = read_labels(name1)
labels2 = read_labels(name2)
print("F1: ", F1_calculation(labels1, labels2))
