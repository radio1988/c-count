## blob_locs and crops

def sub_sample(A, n, seed=1):
    if n < A.shape[0]:
        np.random.seed(seed=seed)
        A = A[np.random.choice(A.shape[0], n, replace=False), :]
        np.random.seed(seed=None)
    else:
        pass
    return (A)


def blob_radius_histogram(blobs):
    '''
    show blob size distribution with histogram
    works on blob_locs or crops
    '''
    plt.title("Histogram of blob radius")
    plt.hist(crops[:, 2], 40)
    plt.show()


def blob_radius_filter(blobs, r_min, r_max):
    '''
    filter blobs based on size of r
    '''
    flitered_blobs = blobs[blobs[:, 2] >= r_min,]
    flitered_blobs = flitered_blobs[flitered_blobs[:, 2] < r_max,]
    print("Filtered blobs:", len(flitered_blobs))
    return flitered_blobs


## crops

def crops_stat(crops):
    '''
    print summary of labels in crops
    :param crops:
    :return:
    '''
    print("{} Yes, {} No, {} Uncertain, {} Unlabeled".format(
        sum(crops[:, 3] == 1),
        sum(crops[:, 3] == 0),
        sum(crops[:, 3] == -2),
        sum(crops[:, 3] == -1),))
    print("Total:", crops.shape[0])


def crop_width(image_flat_crops):
    from math import sqrt
    return  int(sqrt(crops.shape[1] - 6) / 2)


def parse_crops(crops):
    '''
    parse crops into images, labels, rs
    :param crops:
    :return:  images, labels, rs
    '''
    flats = crops[:, 6:]
    w = crop_width(crops)  # width of img
    images = flats.reshape(len(flats), 2*w, 2*w)
    labels = crops[:, 3]
    rs = crops[:, 2]

    return images, labels, rs


def flat_label_filter(flats, label_filter = 1):
    if (label_filter != 'na'):
        filtered_idx = flats[:, 3] == label_filter
        flats = flats[filtered_idx, :]
    return (flats)


def flat2image(flat_crop):
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image
    

def sample_crops(crops, proportion, seed):
    np.random.seed(seed)
    crops = np.random.permutation(crops)
    sample = crops[range(int(len(crops)*proportion)), :]
    np.random.seed(seed=None)
    print(len(sample), "samples taken")
    return sample
    