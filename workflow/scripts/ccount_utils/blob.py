import gzip, os, sys, subprocess
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path
from scipy import ndimage
from skimage import filters  # io skipped todo check
from skimage.feature import blob_log  # blob_doh, blob_dog
from skimage.draw import disk
from ccount_utils.img import down_scale, equalize, float_image_auto_contrast
from IPython.display import clear_output


def sub_sample(A, n, seed=1):
    """
    subsample n rows from np.array A
    replace=False
    default seed is 1 (repeatable)

    A: np.array, can be  (locs, labeled_locs, crops)
    n: num of sub-samples needed
        - if n is an integer between 1 and nrow(A), n rows will be sampled
        - if n is a float between 0 and 1, nrow(A) * n will be sampled
    """
    print("\n<sub_sample>")
    n_rows = A.shape[0]

    if n <= 0:
        raise Exception('n must be float between 0-1 or int >=1')
    elif 0 < n < 1:
        n = int(A.shape[0] * n)
    elif 1 <= n <= n_rows:
        n = int(n)
    elif n > n_rows:
        warnings.warn("WARNING: n-subsample is larger than n_row of array, \n\
        but we simply randomized the rows and let it pass\n")
        n = n_rows

    np.random.seed(seed)
    idx = np.random.choice(A.shape[0], size=n, replace=False)
    np.random.seed(None)

    return A[idx]


def get_blob_statistics(blobs):
    """
    Input: blobs, can be locs (yxr) labeled_locs (yxrL) or crops (yxrL + flattened_crop_img)
    Output: a dictionary of count for each type of labels

    previously called 'crop_stats'
    """
    print("\n<get_blob_statistics>")

    n_cols = blobs.shape[1]
    n_blobs = blobs.shape[0]

    if n_cols > 10:
        print('yxrL+crops format, crop width =', crop_width(blobs))
    elif n_cols >= 4:
        print('yxrL format')
        [negative, positive, maybe, artifact, unlabeled] = [
            sum(blobs[:, 3] == 0),
            sum(blobs[:, 3] == 1),
            sum(blobs[:, 3] == 3),
            sum(blobs[:, 3] == 4),
            sum(blobs[:, 3] == -1)
        ]
        print("Negatives: {}, Positives: {}, Maybes: {}, Artifacts: {}, Unlabeled: {}\n".format(
            negative, positive, maybe, artifact, unlabeled))
    elif n_cols == 3:
        print('yxr format, all {} blobs unlabeled'.format(n_blobs))
        [negative, positive, maybe, artifact, unlabeled] = [0, 0, 0, 0, n_blobs]
    else:
        raise Exception("blobs array format error: has less than 3 columns")

    return {'positive': positive,
            "negative": negative,
            'maybe': maybe,
            'artifact': artifact,
            'unlabeled': unlabeled}


def crop_width(crops):
    """
    Input crops: yxrL + flattened crop in an np.array
    """
    if crops.shape[1] <= 6 + 4:
        raise Exception("this file is locs file, not crops file\n")
    else:
        return int(sqrt(crops.shape[1] - 6) / 2)


def parse_crops(crops):
    """
    input:  crops [yxrL ...]
    output:  images, labels, rs
    """
    flats = crops[:, 6:]
    w = crop_width(crops)
    images = flats.reshape(len(flats), 2 * w, 2 * w)
    labels = crops[:, 3]
    rs = crops[:, 2]

    return images, labels, rs


# def remove_edge_crops(flat_blobs):
#     """
#     some crops of blobs contain edges, because they are from the edge of scanned areas or on the edge of the well
#     use this function to remove blobs with obvious long straight black/white lines
#     verified no need to use this 2024/11, as ccount can safely mark these kind of negatives with very little FP
#     """
#     import cv2
#     from .plot import flat2image
#     from .mask_blob_img import mask_blob_img
#     good_flats = []
#     bad_flats = []
#     for i in range(0, flat_blobs.shape[0]):
#         flat = flat_blobs[i,]
#         crop = flat2image(flat)
#         crop = mask_blob_img(crop, r=flat[2])
#         crop = crop * 255
#         crop = crop.astype(np.uint8)
#
#         crop = cv2.blur(crop, (4, 4))  # 4 is good
#         # https://www.pyimagesearch.com/2021/05/12/opencv-edge-detection-cv2-canny/
#         edges = cv2.Canny(crop, 240, 250, apertureSize=7)  # narrow (240, 250) is good, 7 is good
#         lines = cv2.HoughLinesP(edges,
#                                 rho=1, theta=np.pi / 180,
#                                 threshold=30, minLineLength=20,
#                                 maxLineGap=2)  # threashold 30 is sensitive, minLineLength20 is good
#
#         if lines is not None:  # has lines
#             bad_flats.append(flat)
#         else:  # no lines
#             good_flats.append(flat)
#     if len(good_flats) > 0:
#         good_flats = np.stack(good_flats)
#     if len(bad_flats) > 0:
#         bad_flats = np.stack(bad_flats)
#     return (good_flats, bad_flats)

def load_blobs(fname):
    """
    input: npy or npy.gz file
    output: np.array of locs or crops

    gives statistics if Label is present
    """
    print("\n<load_blobs>")
    if not os.path.isfile(fname):
        raise Exception('input file', fname, 'not found')

    if fname.endswith('npy'):
        array = np.load(fname)
    elif fname.endswith('npy.gz'):
        f = gzip.GzipFile(fname, "r")
        array = np.load(f)
    else:
        raise Exception("blob file format not recognized, suffix not npy nor npy.gz")

    get_blob_statistics(array)
    return array


def save_locs(crops, fname):
    """
    Input: np.array of blobs (crops, locs)
    Output: locs.npy.gz file ( 4 columns only, yxrL)

    Note:
    - if inputs are crops, trim to xyrL formatted locs (to save space)
    - if inputs are yxr formatted locs, padding to yxrL format with -1(unlabeled) as labels
    """

    print('\n<save_locs>')

    if not fname.endswith('.npy.gz'):
        raise Exception("file format for <save_locs> not npy.gz:{}\n".format(fname))

    if crops.shape[0] < 1:
        raise Exception("n_crops equal to zero\n")

    # Trim crops to locs
    w = crops.shape[1]
    if w > 4:
        print('input locs array are crops (np.array with flattened crop image)')
        print('trimming crops to locs in yxrL format')
        locs = crops[:, 0:4]
    elif w == 4:
        print('input locs array is in yxrL format')
        locs = crops
    elif w == 3:
        print('input locs array is in yxr format, it does not have an L (labels) column')
        print('padding with "-1" (i.e. unlabeled) as labels')
        padding = np.full((crops.shape[0], 1), -1)  # 5:unlabeled
        locs = np.hstack([crops, padding])
    else:
        sys.exit("locs/crops format error")

    print('num of blob locs: {}'.format(locs.shape[0]))
    print('blob locs head:', locs[0:4, ])

    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    tempName = fname.replace(".npy.gz", ".npy")
    np.save(tempName, locs)
    subprocess.run('gzip -f ' + tempName, shell=True, check=True)
    print(fname, 'saved\n')


def save_crops(crops, fname):
    """
    Input: crops
    Output: npy.gz or npy files
    """
    print("\n<save_crops>")

    get_blob_statistics(crops)

    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)

    if crops.shape[1] > 4:
        print("saving crops:", fname)
    else:
        print("saving locs:", fname)

    if fname.endswith('.npy.gz'):
        fname = fname.replace(".npy.gz", ".npy")
        np.save(fname, crops)
        subprocess.run("gzip -f " + fname, shell=True, check=True)
    elif fname.endswith('.npy'):
        np.save(fname, crops)
    else:
        raise Exception('crop output suffix not .npy nor .npy.gz')


def find_blobs(image_neg, scaling_factor=4,
               max_sigma=12, min_sigma=3, num_sigma=20, threshold=0.1, overlap=.2):
    """
    Input:
    gray scaled image with bright blob on dark background (image_neg)

    Output:
    [n, 3] array of blob information, [y, x, r]

    Method:
    https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html
    https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log

    Steps:
    1. scale down for faster processing
    2. blob detection
    3. scale back [y,x,r] and output

    Params:
    - max_sigma, min_sigma: the max/min size of blobs able to be detected
    - num_sigma: the accuracy, larger slower, more accurate
    - threshold: the min contrast for a blob
    - overlap: how much overlap of blobs allowed before merging

    # larger num_sigma: more accurate boundry, slower, try 15
    # larger max_sigma: larger max blob size, slower
    # threshold: larger, less low contrast blobs

    Default Params (Rui 2024)
    blob_detection_scaling_factor: 4  # 1, 2, 4 (08/23/2021 for blob_detection)
    max_sigma: 12 # 6 for 8 bit, larger for detecting larger blobs
    min_sigma: 3  # 2-8
    num_sigma: 20  # smaller->faster, less accurate, 5-20
    threshold: 0.1  # 0.02 too sensitive, 0.1 to ignore debris
    overlap: .2 # overlap larger than this, smaller blob gone, not sensitive
    blob_extention_ratio: 1.4 # for vis in jpg
    blob_extention_radius: 10 # for vis in jpg
    crop_width: 80  # padding width, which is cropped img width/2 (50), in blob_cropping.py
    """
    print("\n<find_blobs>")

    tic = time.time()
    print('image size:', image_neg.shape)
    print("scaling factor = {}".format(scaling_factor))
    image_neg = down_scale(image_neg, scaling_factor)
    print('scaled image size for faster blob detection:', image_neg.shape)

    blobs = blob_log(
        image_neg,
        max_sigma=max_sigma, min_sigma=min_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
        exclude_border=False
    )

    blobs[:, 2] = blobs[:, 2] * sqrt(2)  # adjust r
    blobs = blobs * scaling_factor  # scale back coordinates
    toc = time.time()

    print("blob detection time: {}s".format(round(toc - tic), 2))
    print("{} blobs detected\n".format(blobs.shape[0]))
    return blobs


def pad_with(vector, pad_width, iaxis, kwargs):
    """
    to make np.pad in crop_blobs work
    """
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def crop_blobs(locs, image, area=0, place_holder=0, crop_width=80):
    """
    locs: locs [n, 0:3], [y, x, r] or labels [n, 0:4], [y, x, r, L]
    image: image, corresponding image

    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers

    Algorithm:
    1. White padding
    2. Crop for each blob

    @type crop_width: int
    """
    print("\n<crop_blobs> cropping..")
    # White padding so that locs on the edge can get cropped image
    padder = max(np.max(image), 1)
    padded_img = np.pad(image, crop_width, pad_with, padder=padder)  # 1 = white padding, 0 = black padding

    num_locs = locs.shape[0]
    crop_size = crop_width * 2  # todo seems confusing
    flat_crop_size = crop_size * crop_size + 6  # 6 for y, x, r, L, area, place_holder
    print("flat_crop_size: ", flat_crop_size)

    # Preallocate output array to save time
    crops = np.empty((num_locs, flat_crop_size), dtype=padded_img.dtype)
    print("empty crops shape: ", crops.shape)

    for i, blob in enumerate(locs):
        y, x, r = blob[0:3]  # Counter-intuitive order, historical reasons

        L = blob[3] if locs.shape[1] >= 4 else -1  # Unlabeled
        area = -1
        place_holder = -1
        y_ = int(y + crop_width)
        x_ = int(x + crop_width)  # Adjust for padding

        # Crop image
        cropped_img = padded_img[
                      y_ - crop_width:y_ + crop_width,
                      x_ - crop_width:x_ + crop_width
                      ]

        # Flatten crop and concatenate metadata
        flat_crop = cropped_img.flatten()
        crops[i,] = np.concatenate(([y, x, r, L, area, place_holder], flat_crop))

    return crops


def sort_blobs(blobs):
    """
    blobs: yxrL
    sorted by r, x, y
    """
    print("\n<sort_blobs>")
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
    if len(set_overlap) < max(len(set1), len(set2)):
        print('two crops are different, intersect less than input')
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


def mask_blob_img(image, r=10, blob_extention_ratio=1, blob_extention_radius=0):
    """
    input: one image [100, 100], and radius of the blob
    return: hard-masked image of [0,1] scale for training
    """
    print("\n<mask_blob_img>")

    image = float_image_auto_contrast(image)

    r_ = r * blob_extention_ratio + blob_extention_radius
    w = int(image.shape[0] / 2)
    mask = np.zeros((2 * w, 2 * w))  # zeros are masked to be black
    rr, cc = disk((w - 1, w - 1), min(r_, w - 1))
    mask[rr, cc] = 1  # 1 is white
    hard_masked = (1 - (1 - image) * mask)

    return hard_masked


def area_calculation(img, r,
                     plotting=False, outname=None,
                     blob_extention_ratio=1.4, blob_extention_radius=10):
    """
    read one image
    output area-of-pixels as int
    outname: outname for plotting, e.g. 'view_area_cal.pdf'
    """
    # automatic thresholding method such as Otsu's (avaible in scikit-image)
    img = float_image_auto_contrast(img)

    try:
        val = filters.threshold_yen(img)
    except ValueError:
        # print("Ops, got blank blob crop")
        return (0)

    r = r * blob_extention_ratio + blob_extention_radius

    # cells as 1 (white), background as 0 (black)
    drops = ndimage.binary_fill_holes(img < val)

    # mask out of the circle to be zero
    w = int(img.shape[0] / 2)
    mask = np.zeros((2 * w, 2 * w))
    rr, cc = disk((w - 1, w - 1), min(r, w - 1))
    mask[rr, cc] = 1  # 1 is white

    # apply mask on binary image
    masked = abs(drops * mask)

    if (plotting):
        plt.subplot(1, 2, 1)
        plt.imshow(img, 'gray', clim=(0, 1))
        plt.subplot(1, 2, 2)
        plt.imshow(masked, cmap='gray')
        if outname:
            plt.savefig(outname)
        else:
            plt.show()

    return int(sum(sum(masked)))


def area_calculations(crops,
                      blob_extention_ratio=1.4, blob_extention_radius=10,
                      plotting=False):
    """
    only calculate for blobs matching the filter
    """
    print("\n<area_calculations>")
    images, labels, rs = parse_crops(crops)
    areas = [area_calculation(image, r=rs[ind], plotting=plotting,
                              blob_extention_ratio=blob_extention_ratio,
                              blob_extention_radius=blob_extention_radius)
             for ind, image in enumerate(images)]
    print("labels (top 5):", [str(int(x)) for x in labels][0:min(5, len(labels))])
    print('areas (top 5):', areas[0:min(5, len(labels))])
    return areas


def flat2image(flat_crop):
    flat = flat_crop[6:]
    w = int(sqrt(len(flat)) / 2)
    image = np.reshape(flat, (w + w, w + w))
    return image


def visualize_blobs_on_img(image, blob_locs,
                           blob_extention_ratio=1.4, blob_extention_radius=10, fname=None):
    """
    image: image where blobs were detected from
    blob_locs: blob info array n x 4 [x, y, r, label], crops also works, only first 3 columns used
    blob_locs: if labels in the fourth column provided, use that to give colors to blobs

    output: image with yellow circles around blobs
    """
    print("\n<visualize_blobs_on_img>")
    px = 1 / plt.rcParams['figure.dpi']

    print("image shape:", image.shape)

    fig, ax = plt.subplots(figsize=(image.shape[1] * px + 0.5, image.shape[0] * px + 0.5))
    ax.imshow(image, 'gray')

    print("blob shape:", blob_locs.shape)
    if blob_locs.shape[1] >= 4:
        labels = blob_locs[:, 3]

        idx0_negative = labels == 0
        idx1_positive = labels == 1
        idx3_maybe = labels == 3
        idx5_nolabel = labels == -1
        idx_others = [x not in [0, 1, 3, -1] for x in labels]

        ax.set_title('Blob Visualization\n\
            Red: Positive {}, Blue: Negative {}, Green: Maybe {}, Black: Others {}, Gray: Not-Labeled {}'.format(
            sum(idx1_positive),
            sum(idx0_negative),
            sum(idx3_maybe),
            sum(idx_others),
            sum(idx5_nolabel)
        ))

        for loc in blob_locs[idx0_negative, 0:3]:
            y, x, r = loc
            p_negative = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(0, 0, 1, 1),  # blue
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_negative)

        for loc in blob_locs[idx1_positive, 0:3]:
            y, x, r = loc
            p_positive = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(1, 0, 0, 1),  # red
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_positive)

        for loc in blob_locs[idx3_maybe, 0:3]:
            y, x, r = loc
            p_maybe = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(0, 1, 0, 1),  # green
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_maybe)

        for loc in blob_locs[idx5_nolabel, 0:3]:
            y, x, r = loc
            p_nolabel = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(0.5, 0.5, 0.5, 1),  # gray
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_nolabel)

        for loc in blob_locs[idx_others, 0:3]:
            y, x, r = loc
            p_others = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(1, 1, 1, 1),  # black
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_others)
    else:
        print('no labels available, all circles will be gray')
        ax.set_title('Visualizing {} blobs (All are Not Labeled)'.format(blob_locs.shape[0]))
        for loc in blob_locs[:, 0:3]:
            y, x, r = loc
            p_yellow = plt.Circle(
                (x, y),
                r * blob_extention_ratio + blob_extention_radius,
                color=(0.5, 0.5, 0.5, 1),  # gray
                linewidth=2,
                fill=False
            )
            ax.add_patch(p_yellow)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def visualize_blob_compare(image, blob_locs, blob_locs2,
                           blob_extention_ratio=1.4, blob_extention_radius=10, fname=None):
    """
    image: image where blobs were detected from
    blob_locs: blob info array n x 4 [x, y, r, label]
    blob_locs2: ground truth

    output: image with colored circles around blobs
        GT, Label
        0, 0, blue
        1, 1, pink
        0, 1, red
        1, 0, purple
    """
    from ccount_utils.clas import F1_calculation

    print("\n<visualize_blob_compare>")
    px = 1 / plt.rcParams['figure.dpi']

    print("image shape:", image.shape)
    print("blob shape:", blob_locs.shape, blob_locs2.shape)

    if not blob_locs2.shape[1] == blob_locs.shape[1]:
        raise ValueError('num of locs in crops and crops2 different')

    if blob_locs.shape[1] <= 3 or blob_locs2.shape[1] <= 3:
        raise ValueError('crop or crops2 has no label')

    labels = blob_locs[:, 3]
    labels2 = blob_locs2[:, 3]
    precision, recall, F1 = F1_calculation(labels, labels2)

    fig, ax = plt.subplots(figsize=(image.shape[1] * px + 0.5, image.shape[0] * px + 0.5))
    ax.imshow(image, 'gray')

    ax.set_title('Visualizing blobs:\n\
        Red: FP, Yellow: FN, Green: TP, Blue: TN\n\
        Precision: {:.3f}, Recall: {:.3f}, F1: , {:.3f}'.format(precision, recall, F1))

    fp = [gt == 0 and clas == 1 for gt, clas in zip(labels2, labels)]
    fn = [gt == 1 and clas == 0 for gt, clas in zip(labels2, labels)]
    tp = [gt == 1 and clas == 1 for gt, clas in zip(labels2, labels)]
    tn = [gt == 0 and clas == 0 for gt, clas in zip(labels2, labels)]

    for loc in blob_locs[fp, 0:3]:
        y, x, r = loc
        FP = plt.Circle((x, y),
                        r * blob_extention_ratio + blob_extention_radius,
                        color=(1, 0, 0, 0.7), linewidth=2,
                        fill=False)
        ax.add_patch(FP)

    for loc in blob_locs[fn, 0:3]:
        y, x, r = loc
        FN = plt.Circle((x, y),
                        r * blob_extention_ratio + blob_extention_radius,
                        color=(1, 1, 0, 0.7), linewidth=2,
                        fill=False)
        ax.add_patch(FN)

    for loc in blob_locs[tp, 0:3]:
        y, x, r = loc
        TP = plt.Circle((x, y),
                        r * blob_extention_ratio + blob_extention_radius,
                        color=(0, 1, 0, 0.7), linewidth=2,
                        fill=False)
        ax.add_patch(TP)

    for loc in blob_locs[tn, 0:3]:
        y, x, r = loc
        TN = plt.Circle((x, y),
                        r * blob_extention_ratio + blob_extention_radius,
                        color=(0, 0, 1, 0.7), linewidth=2,
                        fill=False)
        ax.add_patch(TN)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()


def plot_flat_crop(flat_crop, blob_extention_ratio=1.4, blob_extention_radius=10,
                   image_scale=1, fname=None, equalization=False):
    '''
    input: flat_crop of a blob, e.g. (160006,)
    output: two plots
        - left: original image with yellow circle
        - right: binary for area calculation
    '''
    if len(flat_crop) >= 6:
        [y, x, r, label, area, place_holder] = flat_crop[0:6]
    elif len(flat_crop) >= 4:
        [y, x, r, label] = flat_crop[0:4]
        area = -1
    elif len(flat_crop) >= 3:
        [y, x, r] = flat_crop[0:3]
        label = -1
        area = -1
    else:
        raise Exception('this blob does not have y,x,r info, useless\n')

    r = r * blob_extention_ratio + blob_extention_radius
    image = flat2image(flat_crop)
    image = float_image_auto_contrast(image)
    w = sqrt(len(flat_crop) - 6) / 2
    W = w * image_scale / 30
    area = flat_crop[6]

    if equalization:
        image = equalize(image)

    fig, ax = plt.subplots(figsize=(W, W))
    ax.set_title('Image for Labeling\ncurrent label:{}\n\
        x:{}, y:{}, r:{}, area:{}'.format(int(label), x, y, r, area))
    ax.imshow(image, 'gray')
    c = plt.Circle((w, w), r,
                   color=(1, 1, 0, 0.7), linewidth=2,
                   fill=False)
    ax.add_patch(c)

    if fname:
        plt.savefig(fname)
    else:
        plt.show()

    plt.close('all')

    return image


def plot_flat_crops(crops, blob_extention_ratio=1, blob_extention_radius=0, fname=None):
    """
    input: crops
    task: call plot_flat_crop many times
    """
    for i, flat_crop in enumerate(crops):
        if fname:
            plot_flat_crop(flat_crop,
                           blob_extention_ratio=blob_extention_ratio,
                           blob_extention_radius=blob_extention_radius,
                           fname=fname + '.rnd' + str(i) + '.jpg')
        else:
            plot_flat_crop(flat_crop,
                           blob_extention_ratio=blob_extention_ratio,
                           blob_extention_radius=blob_extention_radius)


def show_rand_crops(crops, label_filter="na", num_shown=1,
                    blob_extention_ratio=1, blob_extention_radius=0, seed=None, fname=None):
    """
    crops: the blob crops
    label_filter: 0, 1, -1; "na" means no filter
    fname: None, plot.show(); if fname provided, saved to png
    """
    print("\n<show_rand_crops>")
    if (label_filter != 'na'):
        filtered_idx = [str(int(x)) == str(label_filter) for x in crops[:, 3]]
        crops = crops[filtered_idx, :]

    if len(crops) == 0:
        print('num_blobs after filtering is 0')
        return False

    if (len(crops) >= num_shown):
        print("Samples of {} blobs will be plotted".format(num_shown))
        crops = sub_sample(crops, num_shown, seed)
    else:
        print("all {} blobs will be plotted".format(len(crops)))

    plot_flat_crops(
        crops,
        blob_extention_ratio=blob_extention_ratio,
        blob_extention_radius=blob_extention_radius,
        fname=fname)

    images, labels, rs = parse_crops(crops)

    return True


def pop_label_flat_crops(crops, random=True, seed=1, skip_labels=[0, 1, 2, 3]):
    """
    input:
        crops
    task:
        plot padded crop, let user label them
    labels:
        no: 0, yes: 1, groupB: 2, uncertain: 3, artifacts: 4, unlabeled: -1
        never use neg values
    skipLablels:
        crops with current labels in these will be skipped, to save time
    output:
        labeled array in the original order
    """

    N = len(crops)
    if random:
        np.random.seed(seed=seed)
        idx = np.random.permutation(N)
        np.random.seed()
    else:
        idx = np.arange(N)

    labels = crops[idx, 3]
    keep = [x not in skip_labels for x in labels]
    idx = idx[keep]

    num_to_label = sum(keep)
    print("Input: there are {} blobs to label in {} blobs". \
          format(num_to_label, len(crops)))

    i = -1
    while i < len(idx):
        i += 1
        if i >= len(idx):
            break

        plot_flat_crop(crops[idx[i], :])

        label = input('''labeling for the {}/{} blob, \
            yes=1, no=0, skip=s, go-back=b, excape(pause)=e'''. \
                      format(i + 1, num_to_label))

        if label == '1':
            crops[idx[i], 3] = 1
        elif label == '0':
            crops[idx[i], 3] = 0
        elif label == '2':
            crops[idx[i], 3] = 2
        elif label == '3':
            crops[idx[i], 3] = 3
        elif label == 's':
            pass
        elif label == 'b':
            i -= 2
        elif label == 'e':
            label = input('are you sure to quit?(y/n)')
            if label == 'y':
                print("labeling stopped manually")
                break
            else:
                print('continued')
                i -= 1
        else:
            print('invalid input, please try again')
            i -= 1

        print('new label: ', label, crops[idx[i], 0:4])
        clear_output()

    return crops
