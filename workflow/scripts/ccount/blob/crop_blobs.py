def pad_with(vector, pad_width, iaxis, kwargs):
    '''
    to make np.pad in crop_blobs work
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def crop_blobs(blobs, image, label = 5, area=0, place_holder=0, crop_width=80):
    '''
    input1: blobs, blob info [n, 0:3], [y, x, r]
    input2: image, corresponding image
    plt: cropped images
    return: cropped padded images in a flattened 2d-array, with meta data in the first 6 numbers

    Algorithm:
    1. White padding
    2. Crop for each blob
    '''
    import numpy as np

    # White padding so that blobs on the edge can get cropped image
    padder = max(np.max(image), 1)
    padded = np.pad(image, crop_width, pad_with, padder=padder)  # 1 = white padding, 0 = black padding

    # crop for each blob
    #flat_crops = np.empty((0, int(6 + 2 * crop_width * 2 * crop_width)))
    L = []
    n_total = blobs.shape[0]
    for i, blob in enumerate(blobs):
        y, x, r = blob[0:3]  # conter-intuitive order
        #print("cropping blob {}/{}  x:{} y:{} r:{}".format(i, n_total, x, y, r))
        y_ = int(y + crop_width)
        x_ = int(x + crop_width)  # adj for padding

        cropped_img = padded[
            y_ - crop_width: y_ + crop_width, 
            x_ - crop_width: x_ + crop_width]  # x coordinates use columns to locate, vise versa

        flat_crop = np.insert(
            cropped_img.flatten(), [0, 0, 0, 0, 0, 0], 
            [y, x, r, label, area, place_holder])  # -1 unlabeled
        L.append(flat_crop)
    return np.array(L)

