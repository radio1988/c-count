import gzip, os, sys, subprocess
import warnings
import numpy as np

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


def load_locs(fname):
    '''
    assuming reading fname.locs.npy.gz
    read into np.array
    can also read crops without giving statistics
    y,x,r,L,crop-flatten
    '''
    if os.path.isfile(fname):
        if fname.endswith('npy'):
            array = np.load(fname)
        elif fname.endswith('npy.gz'):
            f = gzip.GzipFile(fname, "r")
            array = np.load(f)
        else:
            raise Exception("suffix not npy nor npy.gz")

        if array.shape[1] > 3:  # with label
            crops_stat(array)
        if array.shape[1] > 4:  # is crop, not only loc
            print("n-crop: {}, crop width: {}\n".format(len(array), crop_width(array)))
        return array
    else:
        raise Exception('input file', fname, 'not found')


def load_crops(in_db_name):
    '''
    alias for load_locs
    '''
    image_flat_crops = load_locs(in_db_name)
    return image_flat_crops


def save_locs(crops, fname):
    """
    Input: np.array of crops or locs
    Output: npy file (not npy.gz)

    Note:
    - if input are crops, trim to xyrL formatted locs (to save space)
    - if input are yxr formatted locs, padding to yxrL format with 5(unlabeled) labels
    - even if fname is x.npy.gz will save into x.npy (locs file not big anyway)
    """
    from .misc import crops_stat, crop_width
    from pathlib import Path

    print('<save_locs>')

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
        print('padding with "5" (i.e. unlabeled) as labels')
        padding = np.full((crops.shape[0], 1), 5)  # 5:unlabeled
        locs = np.hstack([crops, padding])
    else:
        sys.exit("locs/crops format error")

    print('num of blob locs: {}'.format(locs.shape[0]))

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
    from .misc import crops_stat, crop_width
    from pathlib import Path
    Path(os.path.dirname(fname)).mkdir(parents=True, exist_ok=True)
    print('dim:', crops.shape)
    if crops.shape[1] > 4:
        print('width:', crop_width(crops))
        print("Saving crops:", fname)
    else:
        print("Saving locs:", fname)

    if fname.endswith('.npy.gz'):
        fname = fname.replace(".npy.gz", ".npy")
        np.save(fname, crops)
        subprocess.run("gzip -f " + fname, shell=True, check=True)
    elif fname.endswith('.npy'):
        np.save(fname, crops)
    else:
        raise Exception('crop output suffix not .npy nor .npy.gz')
