def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    to make np.pad in crop_blobs work
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def crop_blobs(locs, image, area=0, place_holder=0, crop_width=80):
    '''
    input1: locs [n, 0:3], [y, x, r] or labels [n, 0:4], [y, x, r, L]
    input2: image, corresponding image
    plt: cropped images
    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers

    Algorithm:
    1. White padding
    2. Crop for each blob
    '''
    import numpy as np

    # White padding so that locs on the edge can get cropped image
    padder = max(np.max(image), 1)
    padded = np.pad(image, crop_width, pad_with, padder=padder)  # 1 = white padding, 0 = black padding

    # crop for each blob
    crops = []
    for i, blob in enumerate(locs):
        y, x, r = blob[0:3]  # conter-intuitive order

        if locs.shape[1] > 3:
            L = blob[3]
        else:
            L = -1  # unlabeled
        y_ = int(y + crop_width)
        x_ = int(x + crop_width)  # adj for padding

        cropped_img = padded[
            y_ - crop_width: y_ + crop_width, 
            x_ - crop_width: x_ + crop_width]  # x coordinates use columns to locate, vise versa

        flat_crop = np.insert(
            cropped_img.flatten(), [0, 0, 0, 0, 0, 0], 
            [y, x, r, L, area, place_holder])  # -1 unlabeled
        crops.append(flat_crop)
    return np.array(crops)

