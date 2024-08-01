# from ccount.blob.intersect import intersect_blobs
import sys


def sort_blobs(blobs):
    blobs = blobs[blobs[:, 2].argsort()]
    blobs = blobs[blobs[:, 1].argsort(kind='mergesort')]
    blobs = blobs[blobs[:, 0].argsort(kind='mergesort')]
    return blobs


def intersect_blobs(blobs1, blobs2):
    blobs1 = sort_blobs(blobs1)
    blobs2 = sort_blobs(blobs2)
    tup1 = [tuple(x[0:3]) for x in blobs1]
    tup2 = [tuple(x[0:3]) for x in blobs2]
    set1 = set(tup1)
    set2 = set(tup2)
    set_overlap = set1 & set2
    if len(set1) != len(set_overlap):
        sys.stderr.write("blobs1 and blobs2 are different, intersection is taken after sorting\n\n")
    idx1 = [x in set_overlap for x in tup1]
    idx2 = [x in set_overlap for x in tup2]
    blobs1b = blobs1[idx1, :]
    blobs2b = blobs2[idx2, :]
    return (blobs1b, blobs2b)


def setdiff_blobs(blobs1, blobs2):
    blobs1 = sort_blobs(blobs1)
    blobs2 = sort_blobs(blobs2)
    tup1 = [tuple(x[0:2]) for x in blobs1]
    tup2 = [tuple(x[0:2]) for x in blobs2]
    set1 = set(tup1)
    set2 = set(tup2)
    setout = set1 - set2
    idx1 = [x in setout for x in tup1]
    blobsout = blobs1[idx1, :]
    return (blobsout)
