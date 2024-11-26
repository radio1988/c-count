## blob_locs and crops

def sub_sample(A, n, seed=1):
    '''
    replace=False
    return shape (n, w**2 + 6)
    '''
    import numpy as np
    if n <= 0:
        raise Exception('n must be float between 0-1 or int >=1')
    if n < 1:
        n = int(A.shape[0] * n)

    if n <= A.shape[0]:
        np.random.seed(seed=seed)
        A = A[np.random.choice(A.shape[0], n, replace=False), :]
        np.random.seed(seed=None)
    else:
        raise Exception("more samples than data asked for sub_sampling process")
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
    if crops.shape[1] > 3:
        [yes, no, uncertain, artifact, unlabeled] = [
            sum(crops[:, 3] == 1), sum(crops[:, 3] == 0),
            sum(crops[:, 3] == 3), sum(crops[:, 3] == 4),
            sum(crops[:, 3] == 5)
        ]
        print("{} Yes(1), {} No(0), {} Uncertain(3), {} artifacts(4), {} Unlabeled(5)\n".format(
            yes, no, uncertain, artifact, unlabeled))
    else:
        raise Exception("Crops does not contain label column\n")
    return {'yes': yes, "no": no, 'uncertain': uncertain, 'artifact': artifact, 'unlabeled': unlabeled}


def crop_width(image_flat_crops):
    from math import sqrt
    if image_flat_crops.shape[1] <= 6 + 4:
        raise Exception("this file is locs file, not crops file\n")
    else:
        return int(sqrt(image_flat_crops.shape[1] - 6) / 2)


def parse_crops(crops):
    '''
    parse crops into images, labels, rs
    :param crops:
    :return:  images, labels, rs
    '''
    flats = crops[:, 6:]
    w = crop_width(crops)  # width of img
    images = flats.reshape(len(flats), 2 * w, 2 * w)
    labels = crops[:, 3]
    rs = crops[:, 2]

    return images, labels, rs


def remove_edge_crops(flat_blobs):
    """
    some crops of blobs contain edges, because they are from the edge of scanned areas or on the edge of the well
    use this function to remove blobs with obvious long straight black/white lines
    """
    import cv2
    import numpy as np
    from .plot import flat2image
    from .mask_image import mask_image
    good_flats = []
    bad_flats = []
    for i in range(0, flat_blobs.shape[0]):
        flat = flat_blobs[i,]
        crop = flat2image(flat)
        crop = mask_image(crop, r=flat[2])
        crop = crop * 255
        crop = crop.astype(np.uint8)

        crop = cv2.blur(crop, (4, 4))  # 4 is good
        # https://www.pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
        edges = cv2.Canny(crop, 240, 250, apertureSize=7)  # narrow (240, 250) is good, 7 is good
        lines = cv2.HoughLinesP(edges,
                                rho=1, theta=np.pi / 180,
                                threshold=30, minLineLength=20,
                                maxLineGap=2)  # threashold 30 is sensitive, minLineLength20 is good

        if lines is not None:  # has lines
            bad_flats.append(flat)
        else:  # no lines
            good_flats.append(flat)
    if len(good_flats) > 0:
        good_flats = np.stack(good_flats)
    if len(bad_flats) > 0:
        bad_flats = np.stack(bad_flats)
    return (good_flats, bad_flats)


## flats    

def flat_label_filter(flats, label_filter=1):
    if (label_filter != 'na'):
        filtered_idx = flats[:, 3] == label_filter
        flats = flats[filtered_idx, :]
    return flats
