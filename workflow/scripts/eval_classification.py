import numpy as np
import pandas as pd
import sys 

from ccount.blob.io import load_crops
from ccount.clas.metrics import F1_calculation


print("usage:  python f1_score.py output/FL.t.pred.npy.gz ../data/FL.t.npy.gz  > f1_score.t.txt")
#todo: calculate mean/median of f1_scores of many pairs of blobs

def read_labels(fname):
    blobs = load_crops(fname)
    labels = blobs[:, 3]
    labels[labels == 9] = 0  # for 2020 April labeled data (0, 1, -2 , 9)
    return labels

name1 = sys.argv[1]  # prediction
name2 = sys.argv[2]  # truth
print(name1, name2)
labels1 = read_labels(name1)
labels2 = read_labels(name2)
precision, recall, F1 = F1_calculation(labels1, labels2)
