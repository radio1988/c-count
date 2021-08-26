def blobs_stat(blobs):
    '''
    print summary of labels in blobs
    :param blobs:
    :return:
    '''
    print("{} Yes, {} No, {} Uncertain, {} Unlabeled".format(
        sum(blobs[:, 3] == 1),
        sum(blobs[:, 3] == 0),
        sum(blobs[:, 3] == -2),
        sum(blobs[:, 3] == -1),))
    print("Total:", blobs.shape[0])


def hist_blobsize(blobs):
    '''
    show blob size distribution with histogram
    '''
    plt.title("Histogram of blob radius")
    plt.hist(blobs[:, 2], 40)
    plt.show()


def blob_width(image_flat_crops):
    from math import sqrt
    return  int(sqrt(image_flat_crops.shape[1] - 6))


def filter_blobs(blobs, r_min, r_max):
    '''
    filter blobs based on size of r
    '''
    flitered_blobs = blobs[blobs[:, 2] >= r_min,]
    flitered_blobs = flitered_blobs[flitered_blobs[:, 2] < r_max,]
    print("Filtered blobs:", len(flitered_blobs))
    return flitered_blobs


def flat_label_filter(flats, label_filter = 1):
    if (label_filter != 'na'):
        filtered_idx = flats[:, 3] == label_filter
        flats = flats[filtered_idx, :]
    return (flats)


def reshape_img_from_flat(flat_crop):
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image


def sub_sample(A, n, seed=1):
    if n < A.shape[0]:
        np.random.seed(seed=seed)
        A = A[np.random.choice(A.shape[0], n, replace=False), :]
        np.random.seed(seed=None)
    else:
        pass
    return (A)


def sample_crops(crops, proportion, seed):
    np.random.seed(seed)
    crops = np.random.permutation(crops)
    sample = crops[range(int(len(crops)*proportion)), :]
    np.random.seed(seed=None)
    print(len(sample), "samples taken")
    return sample
    