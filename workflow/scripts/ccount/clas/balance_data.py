from ..blob.misc import crops_stat
import numpy as np



def balance_by_removal(blobs):
    '''
    balance yes/no ratio to 1, by removing excess NO samples
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    crops_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)

    if N_No > N_Yes:
        print('number of No matched to yes by sub-sampling')
        idx_no = np.random.choice(idx_no, N_Yes, replace=False)
        idx_choice = np.concatenate([idx_yes, idx_no])
        np.random.seed(2)
        np.random.shuffle(idx_choice)
        np.random.seed()
        blobs = blobs[idx_choice,]

    print("After balancing by removing neg samples")
    crops_stat(blobs)

    return blobs


def balance_by_duplication(blobs, maxN=160000):
    '''
    balance yes/no ratio to 1, by duplicating blobs in the under-represented group
    only yes/no considered
    undistinguishable not altered
    result randomized to avoid problems in training
    :return: balanced blobs (with less samples)
    '''
    print('Before balancing:')
    crops_stat(blobs)

    idx_yes = np.arange(0, blobs.shape[0])[blobs[:, 3] == 1]
    idx_no = np.arange(0, blobs.shape[0])[blobs[:, 3] == 0]
    idx_unsure = np.arange(0, blobs.shape[0])[blobs[:, 3] == -2]
    N_Yes = len(idx_yes)
    N_No = len(idx_no)
    # N_unsure = len(idx_unsure)

    if N_No > maxN//2:
        idx_no = np.random.choice(idx_no, maxN//2, replace=False)
    if N_Yes > maxN//2:
        idx_yes = np.random.choice(idx_yes, maxN//2, replace=False)
    if N_Yes < maxN//2:
        idx_yes = np.random.choice(idx_yes, maxN//2, replace=True)
    if N_No > maxN//2:
        idx_no = np.random.choice(idx_no, maxN//2, replace=True)

    idx_choice = np.concatenate([idx_yes, idx_no])
    np.random.shuffle(idx_choice)
    blobs = blobs[idx_choice, ]

    print("After balancing by adding positive samples")
    crops_stat(blobs)

    return blobs

